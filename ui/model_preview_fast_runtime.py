"""Request-local reuse over unchanged frozen adapters. No policy/controller logic.

File contents are rehashed on EVERY request. Only immutable receipt parsing is
reused between requests, after the complete dependency closure was reverified.
"""
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
from hashlib import sha256
from pathlib import Path
import stat
import threading
from time import perf_counter, thread_time

from runtime_v642.catalog import decode
from runtime_v642.contracts import PolicyBinding, digest, freeze, require, RuntimeUnavailable
from runtime_v642.legacy_domain import PongAdapter, KitchenAdapter, pong_nn
from .model_preview_warehouse import WarehouseAdapter

VERSION = 'manual-preview-runtime.request-scoped.v2.1'


def verified_request(function):
    @wraps(function)
    def call(self, *args, **kwargs):
        with self.runtime.request():
            return function(self, *args, **kwargs)
    return call


class RequestRuntime:
    def __init__(self, catalog):
        self.catalog = catalog
        self.current = ContextVar('manual_preview_request', default=None)
        self.completed = threading.local()
        self.parsed = {}
        try:
            quota,period=Path('/sys/fs/cgroup/cpu.max').read_text().split()
            self.compute=dict(cpu_quota_us=None if quota=='max' else int(quota),cpu_period_us=int(period))
        except (OSError,ValueError):
            self.compute=dict(cpu_quota_us=None,cpu_period_us=None)
        original_path, original_verify = catalog.assets.path, catalog.verify_assets

        def path(key):
            scope = self.current.get()
            if scope is None:
                return original_path(key)
            if key not in scope['paths']:
                require(key in catalog.assets.records,'required asset hash is absent')
                target = catalog.assets.root / catalog.assets.records[key].path
                # Manifest paths were validated lexically by AssetResolver.
                # Check each shared parent once in this request, rather than
                # resolving/readlink-ing the same prefix for every file.
                for parent in target.parents:
                    if not parent.is_relative_to(catalog.assets.root):break
                    if parent not in scope['directories']:
                        require(stat.S_ISDIR(parent.lstat().st_mode),'symlinked or missing asset parent')
                        scope['directories'].add(parent)
                before = target.lstat()
                require(stat.S_ISREG(before.st_mode) and not before.st_mode & 0o222,
                        'missing, writable or symlinked package asset')
                raw = target.read_bytes()
                after = target.lstat()
                require((before.st_dev,before.st_ino,before.st_mode,before.st_size,before.st_mtime_ns)==
                        (after.st_dev,after.st_ino,after.st_mode,after.st_size,after.st_mtime_ns)
                        and sha256(raw).hexdigest()==key,'package asset hash or identity mismatch')
                scope['paths'][key] = target
                scope['bytes'][key] = raw
            return scope['paths'][key]

        def verify():
            scope = self.current.get()
            if scope is None or not scope['verified']:
                ref = Path(catalog._catalog_ref['path'])
                require(ref.is_file() and not ref.is_symlink() and not ref.stat().st_mode & 0o222,
                        'catalog missing, symlinked or writable')
                original_verify()
                if scope is not None:
                    scope['verified'] = True

        def parsed_json(key):
            verified_path = path(key)  # still verifies when called outside a request
            if key not in self.parsed:
                scope = self.current.get()
                self.parsed[key] = freeze(decode(scope['bytes'][key] if scope else verified_path.read_bytes()))
            return self.parsed[key]

        # Per-instance wrappers preserve exact catalog type and all old callers.
        catalog.assets.path, catalog.assets.json, catalog.verify_assets = path, parsed_json, verify

    @contextmanager
    def request(self):
        if self.current.get() is not None:
            yield self.current.get(); return
        scope = dict(paths={}, bytes={}, directories=set(), verified=False, memo=OrderedDict(),
                     integrity=0., inference=0., physics=0., storage=0., total=0.)
        token = self.current.set(scope); began = perf_counter(); began_cpu = thread_time()
        try:
            with self.measure('integrity'):
                require(self.catalog.assets.root.resolve()==self.catalog.assets.root,
                        'package root or ancestor replaced by a symlink')
                self.catalog.verify_assets()
            yield scope
        except OSError as exc:
            raise RuntimeUnavailable('preview asset or storage unavailable') from exc
        finally:
            scope['total'] = (perf_counter() - began) * 1000
            scope['cpu'] = (thread_time() - began_cpu) * 1000
            self.completed.metrics = {k:scope[k] for k in ('integrity','inference','physics','storage','total','cpu')}
            self.current.reset(token)

    @contextmanager
    def measure(self, category):
        scope = self.current.get(); began = perf_counter()
        previous_inference = scope['inference'] if scope else 0
        try:
            yield
        finally:
            if scope is not None:
                duration = (perf_counter() - began) * 1000
                if category == 'physics':
                    duration -= scope['inference'] - previous_inference
                scope[category] += max(0, duration)

    def memo(self, key, compute, *, category='inference'):
        scope = self.current.get()
        if scope is None:
            return compute()
        cache = scope['memo']
        if key not in cache:
            with self.measure(category) if category else nullcontext():
                cache[key] = compute()
            if len(cache) > 8:  # replay never retains whole episode snapshots
                cache.popitem(last=False)
        return deepcopy(cache[key])


class FastWarehouseAdapter(WarehouseAdapter):
    def __init__(self, catalog, runtime):
        super().__init__(catalog); self.runtime = runtime

    def _selection(self, env, state, binding):
        # The caller still restores and verifies the entire state before reuse.
        return self.runtime.memo(('warehouse', binding.identity_sha256, state['state_sha256']),
                                 lambda: super(FastWarehouseAdapter,self)._selection(env,state,binding))

    def decide(self, state, binding):
        self._binding(binding)
        # Hash complete actual contents before reusing a validated decision;
        # the client-supplied state_sha256 alone is never a cache identity.
        return self.runtime.memo(('warehouse-decision',binding.identity_sha256,digest(state)),
            lambda: super(FastWarehouseAdapter,self).decide(state,binding),category=None)


class FastLegacyMixin:
    def __init__(self, catalog, runtime):
        super().__init__(catalog); self.runtime = runtime
        self._validated_receipts = set()

    def _core(self, binding):
        require(type(binding) is PolicyBinding and binding.domain == self.DOMAIN
                and binding == self.catalog.binding(self.DOMAIN, binding.variant), 'unregistered carried-domain binding')
        key = binding.identity_sha256
        if key not in self._validated_receipts:
            # Full historical receipt/schema/role validation once per immutable
            # binding. Each subsequent request rehashes ALL its dependencies.
            core = super()._core(binding)
            original_choose = core.choose
            if self.DOMAIN == 'pong':
                def choose(state):
                    core.validate_native(state)
                    require(not state['terminal'], 'no NN action after terminal frame')
                    choices, scores, masks = {}, {}, {}
                    for role in ('human','ai'):
                        scores[role] = pong_nn.logits_from_payload(state,role,core.actor)
                        masks[role] = list(pong_nn.legal_mask(state,role))
                        # Same physical mask and stable tie break as the frozen
                        # choose_from_payload, reusing its already computed logits.
                        choices[role] = pong_nn.ACTIONS[max((i for i in range(len(pong_nn.ACTIONS))
                            if masks[role][i]),key=lambda i:(scores[role][i],-i))]
                    return choices,scores,masks
                core.choose = choose
            else:
                core.choose = original_choose
            self._validated_receipts.add(key)
        core = self._cores[key]; core.verify()
        return core

    def decide(self, state, binding):
        self._core(binding)
        return self.runtime.memo((self.DOMAIN,binding.identity_sha256,digest(state)),
                                 lambda: super(FastLegacyMixin,self).decide(state,binding))


class FastPongAdapter(FastLegacyMixin, PongAdapter):
    pass


class FastKitchenAdapter(FastLegacyMixin, KitchenAdapter):
    pass
