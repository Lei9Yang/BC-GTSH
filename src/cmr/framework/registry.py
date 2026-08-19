from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .contracts import HashMethod, MethodBuildContext, MethodCapabilities

MethodConfigLoader = Callable[[Mapping[str, Any]], Any]
MethodFactory = Callable[[Any, MethodBuildContext], HashMethod]


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    display_name: str
    config_loader: MethodConfigLoader
    factory: MethodFactory
    capabilities: MethodCapabilities = MethodCapabilities()

    def __post_init__(self) -> None:
        normalized = self.method_id.strip().casefold()
        if not normalized or normalized != self.method_id:
            raise ValueError("method_id must be a non-empty lowercase identifier")


class MethodRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, MethodSpec] = {}

    def register(self, spec: MethodSpec) -> None:
        if spec.method_id in self._specs:
            raise ValueError(f"Method already registered: {spec.method_id}")
        self._specs[spec.method_id] = spec

    def get(self, method_id: str) -> MethodSpec:
        key = method_id.strip().casefold()
        try:
            return self._specs[key]
        except KeyError as error:
            available = ", ".join(sorted(self._specs)) or "<none>"
            raise KeyError(f"Unknown method '{method_id}'. Available methods: {available}") from error

    def create(
        self,
        method_id: str,
        parameters: Mapping[str, Any],
        context: MethodBuildContext,
    ) -> HashMethod:
        spec = self.get(method_id)
        config = spec.config_loader(parameters)
        method = spec.factory(config, context)
        if method.method_id != spec.method_id:
            raise ValueError(
                f"Method factory for '{spec.method_id}' returned method_id '{method.method_id}'"
            )
        return method

    def list(self) -> tuple[MethodSpec, ...]:
        return tuple(self._specs[key] for key in sorted(self._specs))


registry = MethodRegistry()


def register_method(spec: MethodSpec) -> None:
    registry.register(spec)


def get_method(method_id: str) -> MethodSpec:
    return registry.get(method_id)


def list_methods() -> tuple[MethodSpec, ...]:
    return registry.list()

