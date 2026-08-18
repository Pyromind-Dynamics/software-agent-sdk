import signal
from types import FrameType

import uvicorn

from openhands.sdk.logger import get_logger


logger = get_logger(__name__)


class LoggingServer(uvicorn.Server):
    """Uvicorn server that logs the signal responsible for shutdown."""

    def handle_exit(self, sig: int, frame: FrameType | None) -> None:
        sig_name = signal.Signals(sig).name
        logger.info(
            "Received signal %s (%d), shutting down...",
            sig_name,
            sig,
        )
        super().handle_exit(sig, frame)
