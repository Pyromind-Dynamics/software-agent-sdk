from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, Field

from openhands.sdk.security import _shell_ast
from openhands.sdk.security.risk import SecurityRisk


class CommandPolicyAction(StrEnum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


class CommandExecutionSource(StrEnum):
    AGENT_TERMINAL = "agent_terminal"
    DIRECT_BASH_API = "direct_bash_api"
    HOOK = "hook"
    CONTROLLED_TOOL = "controlled_tool"
    MCP = "mcp"


class CommandPolicyDecision(BaseModel):
    action: CommandPolicyAction
    risk: SecurityRisk
    rule_id: str | None = None
    reason: str
    redacted_command: str


class CommandExecutionContext(BaseModel):
    source: CommandExecutionSource
    user_id: str | None = None
    tenant_id: str | None = None
    conversation_id: UUID | None = None
    workspace_dir: Path | None = None
    cwd: Path | None = None


class CommandPolicyConfig(BaseModel):
    enabled: bool = True
    allowed_roots: list[Path] = Field(default_factory=list)
    allowed_script_entrypoints: list[Path] = Field(default_factory=list)
    require_tenant_scope: bool = False
    deny_network_to_shell: bool = True
    deny_sensitive_file_reads: bool = True
    deny_host_escape_paths: bool = True
    deny_direct_script_execution: bool = True
    deny_cloud_writes: bool = True


_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password|passwd|bearer)\b"
    r"(\s*[:=]\s*)"
    r"([^\s'\";|&]+)"
)
_SENSITIVE_PATH_MARKERS = (
    ".env",
    "/.env",
    "~/.ssh",
    "/.ssh/",
    "~/.kube",
    "/.kube/",
    "/etc/passwd",
    "/etc/shadow",
)
_SCRIPT_SUFFIXES = (".py", ".sh", ".bash", ".js", ".rb", ".pl")
_SCRIPT_INTERPRETERS = {
    "python",
    "python3",
    "bash",
    "sh",
    "zsh",
    "node",
    "ruby",
    "perl",
}
_FETCH_COMMANDS = {"curl", "wget"}
_SHELL_EXEC_COMMANDS = {"bash", "sh", "zsh"}
_CLOUD_WRITE_COMMANDS = {
    "kubectl": {"apply", "delete", "replace", "scale", "rollout"},
    "terraform": {"apply", "destroy"},
}


def evaluate_command(
    command: str,
    context: CommandExecutionContext,
    config: CommandPolicyConfig | None = None,
) -> CommandPolicyDecision:
    config = config or CommandPolicyConfig()
    redacted_command = redact_command(command)

    if not config.enabled:
        return _allow(redacted_command)

    scope_decision = _evaluate_scope(context, config, redacted_command)
    if scope_decision is not None:
        return scope_decision

    try:
        program = _shell_ast.parse_shell_program(command)
    except UnicodeEncodeError:
        return _deny(
            "invalid-command-encoding",
            "Command contains invalid text encoding.",
            redacted_command,
        )

    for pipeline in _shell_ast.iter_pipelines(program):
        if config.deny_network_to_shell and _is_fetch_to_shell(pipeline):
            return _deny(
                "fetch-to-exec",
                "Remote download piped to a shell is disabled.",
                redacted_command,
            )

    for shell_command in _shell_ast.iter_commands(program):
        basename = _shell_ast.command_basename(shell_command)
        args = [word.text for word in shell_command.words if not word.opaque]
        raw_words = [word.text for word in shell_command.words]

        if basename is None:
            continue

        if _is_destructive_root_delete(basename, args):
            return _deny(
                "destructive-root-delete",
                "Destructive deletion outside the workspace is disabled.",
                redacted_command,
            )

        if _is_raw_disk_operation(basename, args):
            return _deny(
                "raw-disk-op",
                "Raw disk operations are disabled.",
                redacted_command,
            )

        if config.deny_sensitive_file_reads and _reads_sensitive_data(
            basename, raw_words
        ):
            return _deny(
                "secret-read",
                "Reading secrets or host credentials from shell is disabled.",
                redacted_command,
            )

        if config.deny_host_escape_paths and _uses_host_escape_path(
            raw_words, context, config
        ):
            return _deny(
                "host-path-access",
                "Accessing paths outside the allowed workspace roots is disabled.",
                redacted_command,
            )

        if _is_privilege_escalation(basename, args):
            return _deny(
                "privilege-escalation",
                "Privilege escalation commands are disabled.",
                redacted_command,
            )

        if config.deny_cloud_writes and _is_cloud_write(basename, args):
            return _deny(
                "direct-cloud-write",
                "Direct cloud or cluster write operations are disabled.",
                redacted_command,
            )

        if config.deny_direct_script_execution and _is_direct_script_execution(
            basename, args, context, config
        ):
            return _deny(
                "direct-script-execution",
                "Direct script execution is disabled; use a controlled tool or API.",
                redacted_command,
            )

        confirm = _confirm_rule(basename, args)
        if confirm is not None:
            return CommandPolicyDecision(
                action=CommandPolicyAction.CONFIRM,
                risk=SecurityRisk.MEDIUM,
                rule_id=confirm,
                reason="Command requires confirmation before execution.",
                redacted_command=redacted_command,
            )

    return _allow(redacted_command)


def redact_command(command: str) -> str:
    return _SECRET_ASSIGNMENT_RE.sub(r"\1\2[REDACTED]", command)


def _evaluate_scope(
    context: CommandExecutionContext,
    config: CommandPolicyConfig,
    redacted_command: str,
) -> CommandPolicyDecision | None:
    if config.require_tenant_scope:
        if (
            context.user_id is None
            or context.tenant_id is None
            or context.conversation_id is None
            or context.workspace_dir is None
            or context.cwd is None
        ):
            return _deny(
                "missing-execution-scope",
                "Command execution requires tenant, user, conversation, and workspace.",
                redacted_command,
            )

    allowed_roots = _resolved_allowed_roots(context, config)
    if context.cwd is not None and allowed_roots:
        cwd = context.cwd.resolve()
        if not _path_is_within_roots(cwd, allowed_roots):
            return _deny(
                "cwd-outside-workspace",
                "Command cwd must be inside an allowed workspace root.",
                redacted_command,
            )

    return None


def _resolved_allowed_roots(
    context: CommandExecutionContext,
    config: CommandPolicyConfig,
) -> tuple[Path, ...]:
    roots = list(config.allowed_roots)
    if context.workspace_dir is not None:
        roots.append(context.workspace_dir)
    return tuple(root.resolve() for root in roots)


def _path_is_within_roots(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def _is_fetch_to_shell(pipeline: _shell_ast.ShellPipeline) -> bool:
    if len(pipeline.commands) < 2:
        return False
    first = _shell_ast.command_basename(pipeline.commands[0])
    last = _shell_ast.command_basename(pipeline.commands[-1])
    return first in _FETCH_COMMANDS and last in _SHELL_EXEC_COMMANDS


def _is_destructive_root_delete(command: str, args: list[str]) -> bool:
    if command != "rm":
        return False
    recursive = any(arg.startswith("-") and "r" in arg for arg in args)
    if not recursive:
        return False
    return any(arg in {"/", "/home", "/Users", "/root"} for arg in args)


def _is_raw_disk_operation(command: str, args: list[str]) -> bool:
    if command.startswith("mkfs"):
        return True
    if command == "dd":
        return any(arg.startswith("of=/dev/") for arg in args)
    return False


def _reads_sensitive_data(command: str, words: list[str]) -> bool:
    if command in {"env", "printenv"} and not words:
        return True
    if command not in {"cat", "less", "more", "head", "tail", "grep", "rg"}:
        return False
    joined = " ".join(words)
    return any(marker in joined for marker in _SENSITIVE_PATH_MARKERS)


def _uses_host_escape_path(
    words: list[str],
    context: CommandExecutionContext,
    config: CommandPolicyConfig,
) -> bool:
    roots = tuple(root.resolve() for root in config.allowed_roots)
    for word in words:
        if word.startswith("-") and not word.startswith("--"):
            continue
        if _uses_home_variable_path(word):
            return True
        path = _path_arg_to_resolved_path(word, context)
        if path is None:
            continue
        if roots and _path_is_within_roots(path, roots):
            continue
        if _is_host_path(path):
            return True
        if roots:
            return True
    return False


def _uses_home_variable_path(word: str) -> bool:
    return word in {"$HOME", "${HOME}"} or word.startswith(("$HOME/", "${HOME}/"))


def _path_arg_to_resolved_path(
    word: str, context: CommandExecutionContext
) -> Path | None:
    if word.startswith("--") and "=" in word:
        option_value = word.partition("=")[2]
        if option_value:
            return _path_arg_to_resolved_path(option_value, context)
    if word.startswith(("http://", "https://")):
        return None
    if word in {"~", "~/"} or word.startswith("~/"):
        return Path(word).expanduser().resolve()
    path = Path(word)
    if path.is_absolute():
        return path.resolve()
    if context.cwd is not None and (
        "/" in word or word in {".", ".."} or word.startswith("../")
    ):
        return (context.cwd / path).resolve()
    return None


def _is_host_path(path: Path) -> bool:
    text = str(path)
    return text in {"/etc", "/root", "/home", "/Users"} or text.startswith(
        ("/etc/", "/root/", "/home/", "/Users/")
    )


def _is_privilege_escalation(command: str, args: list[str]) -> bool:
    if command == "sudo":
        return True
    return command == "chmod" and any(arg == "777" for arg in args)


def _is_cloud_write(command: str, args: list[str]) -> bool:
    write_args = _CLOUD_WRITE_COMMANDS.get(command)
    if write_args is not None:
        return any(arg in write_args for arg in args)
    return command == "aws" and any(arg == "iam" for arg in args)


def _is_direct_script_execution(
    command: str,
    args: list[str],
    context: CommandExecutionContext,
    config: CommandPolicyConfig,
) -> bool:
    if context.source == CommandExecutionSource.CONTROLLED_TOOL:
        return False

    if command in _SCRIPT_INTERPRETERS:
        if any(flag in args for flag in ("-c", "-e", "--eval", "--command")):
            return True
        if "-m" in args:
            return True
        script_arg = _first_non_option_arg(args)
        if script_arg is None:
            return False
        if script_arg.endswith(_SCRIPT_SUFFIXES):
            return not _is_registered_script(script_arg, config)
        return False

    if command == "npx":
        return True

    if command in {"uv", "poetry"} and len(args) >= 3:
        return args[0] == "run" and _is_direct_script_execution(
            args[1],
            args[2:],
            context,
            config,
        )

    return False


def _first_non_option_arg(args: list[str]) -> str | None:
    for arg in args:
        if not arg.startswith("-"):
            return arg
    return None


def _is_registered_script(script_arg: str, config: CommandPolicyConfig) -> bool:
    script_path = Path(script_arg)
    if not script_path.is_absolute():
        return False
    resolved = script_path.resolve()
    return any(
        resolved == entrypoint.resolve()
        for entrypoint in config.allowed_script_entrypoints
    )


def _confirm_rule(command: str, args: list[str]) -> str | None:
    if command in {"pip", "pip3"} and "install" in args:
        return "dependency-install"
    if command == "npm" and "install" in args:
        return "dependency-install"
    if command == "uv" and "sync" in args:
        return "dependency-install"
    if command == "git" and any(arg in {"commit", "push"} for arg in args):
        return "git-write"
    if command == "rm":
        return "delete-workspace-file"
    if command in {"curl", "wget"}:
        return "network-fetch"
    if command in {"tar", "zip"}:
        return "archive-directory"
    return None


def _allow(redacted_command: str) -> CommandPolicyDecision:
    return CommandPolicyDecision(
        action=CommandPolicyAction.ALLOW,
        risk=SecurityRisk.LOW,
        reason="Command allowed by policy.",
        redacted_command=redacted_command,
    )


def _deny(
    rule_id: str,
    reason: str,
    redacted_command: str,
) -> CommandPolicyDecision:
    return CommandPolicyDecision(
        action=CommandPolicyAction.DENY,
        risk=SecurityRisk.HIGH,
        rule_id=rule_id,
        reason=reason,
        redacted_command=redacted_command,
    )
