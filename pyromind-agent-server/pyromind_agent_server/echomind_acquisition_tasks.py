from __future__ import annotations

import asyncio
import logging

from openhands.agent_server.echomind_acquisition import AcquisitionSimulation
from openhands.agent_server.echomind_acquisition_runtime import (
    AcquisitionSimulationRunner,
)


logger = logging.getLogger(__name__)


class EchoMindAcquisitionTaskManager:
    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[AcquisitionSimulation]] = {}

    def start(
        self,
        *,
        simulation: AcquisitionSimulation,
        runner: AcquisitionSimulationRunner,
    ) -> bool:
        existing = self._tasks.get(simulation.simulation_id)
        if existing is not None and not existing.done():
            return False
        task = asyncio.create_task(
            asyncio.to_thread(runner.run, simulation),
            name=f"echomind-acquisition-{simulation.simulation_id}",
        )
        self._tasks[simulation.simulation_id] = task
        task.add_done_callback(
            lambda completed: self._discard(simulation.simulation_id, completed)
        )
        return True

    async def stop(self) -> None:
        tasks = list(self._tasks.values())
        self._tasks.clear()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def _discard(
        self,
        simulation_id: str,
        task: asyncio.Task[AcquisitionSimulation],
    ) -> None:
        self._tasks.pop(simulation_id, None)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception:
            logger.exception(
                "EchoMind acquisition simulation task crashed: %s",
                simulation_id,
            )
