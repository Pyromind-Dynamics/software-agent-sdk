from __future__ import annotations

import json
from typing import Any

from openhands.sdk.event import ActionEvent
from openhands.sdk.llm import MessageToolCall, TextContent
from openhands.sdk.security import ConfirmRisky, PatternSecurityAnalyzer


class TerminalPermissionPolicy:
    def __init__(self) -> None:
        self._analyzer = PatternSecurityAnalyzer()
        self._policy = ConfirmRisky()

    def requires_confirmation(
        self,
        tool_call_id: str,
        arguments: dict[str, Any],
    ) -> bool:
        action = ActionEvent(
            thought=[TextContent(text="")],
            tool_name="terminal",
            tool_call_id=tool_call_id,
            tool_call=MessageToolCall(
                id=tool_call_id,
                name="terminal",
                arguments=json.dumps(arguments),
                origin="completion",
            ),
            llm_response_id=tool_call_id,
        )
        return self._policy.should_confirm(self._analyzer.security_risk(action))
