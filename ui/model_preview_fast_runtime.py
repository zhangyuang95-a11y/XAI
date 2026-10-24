"""Request-local reuse over unchanged frozen adapters. No policy/controller logic.

File contents are rehashed on EVERY request. Only immutable receipt parsing is
reused between requests, after the complete dependency closure was reverified.
"""
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
from pathlib import Path
import threading
from time import perf_counter

from runtime_v642.catalog import decode
from runtime_v642.contracts import PolicyBinding, freeze, require, RuntimeUnavailable
from runtime_v642.legacy_domain import PongAdapter, KitchenAdapter, pong_nn
from .model_preview_warehouse import WarehouseAdapter

VERSION = 'manual-preview-runtime.request-scoped.v2'


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
        original_path, original_verify = catalog.assets.path, catalog.verify_assets

        def path(key):
            scope = self.current.get()
            if scope is None:
                return original_path(key)
            if key not in scope['paths']:
                scope['paths'][key] = original_path(key)
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
                self.parsed[key] = freeze(decode(verified_path.read_bytes()))
            return self.parsed[key]

        # Per-instance wrappers preserve exact catalog type and all old callers.
        catalog.assets.path, catalog.assets.json, catalog.verify_assets = path, parsed_json, verify

    @contextmanager
    def request(self):
        if self.current.get() is not None:
            yield self.current.get(); return
        scope = dict(paths={}, verified=False, memo=OrderedDict(),
                     integrity=0., inference=0., physics=0., storage=0., total=0.)
        token = self.current.set(scope); began = perf_counter()
        try:
            with self.measure('integrity'):
                self.catalog.verify_assets()
            yield scope
        except OSError as exc:
            raise RuntimeUnavailable('preview asset or storage unavailable') from exc
        finally:
            scope['total'] = (perf_counter() - began) * 1000
            self.completed.metrics = {k:scope[k] for k in ('integrity','inference','physics','storage','total')}
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

    def memo(self, key, compute):
        scope = self.current.get()
        if scope is None:
            return compute()
        cache = scope['memo']
        if key not in cache:
            with self.measure('inference'):
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
        # Validate mutable state and binding even on a repeated call.
        self.restore(state,binding)
        return self.runtime.memo((self.DOMAIN,binding.identity_sha256,state['state_sha256']),
                                 lambda: super(FastLegacyMixin,self).decide(state,binding))


class FastPongAdapter(FastLegacyMixin, PongAdapter):
    pass


class FastKitchenAdapter(FastLegacyMixin, KitchenAdapter):
    pass
