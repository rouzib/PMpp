"""Utilities for defining lazy, import-order-independent package APIs."""

from importlib import import_module
import sys
from types import ModuleType


class _LazyAPIModule(ModuleType):
    """Module type that gives curated exports precedence over submodules."""

    def __getattribute__(self, name):
        namespace = ModuleType.__getattribute__(self, "__dict__")
        exports = namespace.get("_EXPORTS", {})
        if name in exports:
            cache = namespace["_API_CACHE"]
            if name not in cache:
                module_name, attribute = exports[name]
                module = import_module(module_name, ModuleType.__getattribute__(self, "__name__"))
                cache[name] = module if attribute is None else getattr(module, attribute)
            return cache[name]
        return ModuleType.__getattribute__(self, name)

    def __dir__(self):
        namespace = ModuleType.__getattribute__(self, "__dict__")
        return sorted(set(namespace) | set(namespace.get("_EXPORTS", {})))


def install_lazy_api(module_name, exports):
    """Install a stable lazy export map on an already-imported package."""
    module = sys.modules[module_name]
    module.__dict__["_EXPORTS"] = dict(exports)
    module.__dict__["_API_CACHE"] = {}
    module.__dict__["__all__"] = list(exports)
    module.__class__ = _LazyAPIModule


__all__ = ["install_lazy_api"]
