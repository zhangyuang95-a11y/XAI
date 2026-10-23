"""Explicit adapter injection for future Store/Server composition."""
from dataclasses import dataclass

from .contracts import DOMAINS, GROUP_VARIANTS, PolicyBinding, RuntimeUnavailable, freeze, require
from .catalog import PolicyCatalog

ADAPTER_METHODS = ('initial_state', 'demonstration', 'decide', 'step', 'restore', 'replay', 'counterfactual')


@dataclass(frozen=True, slots=True)
class BoundDomain:
    """An identity and injected dependency; no enrollment or execution is implied."""
    binding: PolicyBinding
    adapter: object
    catalog: PolicyCatalog
    group: str
    mode: str

    def verify(self):
        require(self.binding == self.catalog.binding(self.binding.domain, self.binding.variant), 'saved binding changed')
        require(getattr(self.adapter, 'DOMAIN', None) == self.binding.domain, 'injected adapter domain changed')
        if self.mode == 'pilot':
            self.catalog.require_pilot()
        return self.binding

    def explanation_program(self, *, task, role='ai'):
        self.verify()
        if self.group != 'A' or task != 2:
            return None
        require(role in ('human', 'ai'), 'unknown explanation role')
        if not any(p.role == role and p.independent_audit_sha256 is not None for p in self.binding.programs):
            return None
        return self.catalog.program(self.binding, role)


class RuntimeRegistry:
    def __init__(self, catalog, *, adapters, legacy_reader=None):
        require(type(catalog) is PolicyCatalog and set(adapters) == set(DOMAINS), 'explicit three-domain catalog and adapters required')
        for domain, adapter in adapters.items():
            require(getattr(adapter, 'DOMAIN', None) == domain and all(callable(getattr(adapter, name, None))
                for name in ADAPTER_METHODS), 'incomplete versioned domain adapter')
        self.catalog = catalog
        self.adapters = freeze(dict(adapters))
        self._legacy = legacy_reader
        if legacy_reader is not None:
            require(callable(getattr(legacy_reader, 'read_instance', None))
                    and callable(getattr(legacy_reader, 'export_instance', None)), 'legacy interface must expose read/export methods')

    def bind(self, domain, group, *, mode):
        require(domain in DOMAINS and group in GROUP_VARIANTS and mode in ('preview', 'test', 'pilot'), 'unknown runtime selection')
        if mode == 'pilot':
            self.catalog.require_pilot()
        binding = self.catalog.binding(domain, GROUP_VARIANTS[group])
        # group is held by the evidence gate; the future Store passes ONLY the
        # PolicyBinding into adapter methods, keeping group out of Actor input.
        return BoundDomain(binding, self.adapters[domain], self.catalog, group, mode)

    def archived_instance(self, instance_id, *, export=False):
        require(self._legacy is not None, 'no read-only legacy source registered')
        method = self._legacy.export_instance if export else self._legacy.read_instance
        return freeze(method(instance_id))
