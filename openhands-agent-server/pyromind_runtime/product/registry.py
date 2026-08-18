from __future__ import annotations

from pyromind_runtime.contracts.harness import HarnessDescriptor, HarnessProtocol


class HarnessRegistryError(RuntimeError):
    pass


class HarnessNotRegisteredError(HarnessRegistryError):
    pass


class HarnessRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, HarnessProtocol] = {}

    def register(self, harness_id: str, adapter: HarnessProtocol) -> None:
        if harness_id in self._adapters:
            raise HarnessRegistryError(f"harness is already registered: {harness_id}")
        self._adapters[harness_id] = adapter

    async def resolve(
        self,
        harness_id: str,
    ) -> tuple[HarnessProtocol, HarnessDescriptor]:
        adapter = self._adapters.get(harness_id)
        if adapter is None:
            raise HarnessNotRegisteredError(harness_id)
        descriptor = await adapter.describe()
        if descriptor.harness_id != harness_id:
            raise HarnessRegistryError(
                f"registered harness_id {harness_id} does not match "
                f"descriptor {descriptor.harness_id}"
            )
        return adapter, descriptor

    def registered_ids(self) -> tuple[str, ...]:
        return tuple(self._adapters)
