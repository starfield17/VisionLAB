"""Static composition binds trusted metadata and factory in one registration."""
from collections.abc import Callable

from contractcheck.package import AdapterRegistry, AdapterSpec

from .types import InferenceAdapter


class RuntimeRegistry:
    def __init__(self) -> None:
        self.descriptions = AdapterRegistry()
        self._factories: dict[tuple[str, str], Callable[[], InferenceAdapter]] = {}

    def register(self, spec: AdapterSpec, factory: Callable[[], InferenceAdapter]) -> None:
        self.descriptions.register(spec.id, spec.version, spec.config_schema, formats=spec.formats)
        self._factories[spec.id, spec.version] = factory

    def create(self, adapter_id: str, version: str) -> InferenceAdapter:
        return self._factories[adapter_id, version]()
