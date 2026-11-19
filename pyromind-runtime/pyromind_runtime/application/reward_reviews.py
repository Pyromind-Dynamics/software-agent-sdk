from __future__ import annotations

import hashlib
import secrets
from collections.abc import Awaitable, Callable
from pathlib import PurePosixPath

from pyromind_runtime.domain.content import TextContent
from pyromind_runtime.domain.reward_bundle import (
    APPROVE_REWARD,
    MAX_REWARD_FILE_BYTES,
    REWARD_QUESTION_ID,
    REWARD_WORKDIR,
    RewardBundle,
    RewardManifest,
    Rubric,
    StartRewardReview,
    freeze_reward_files,
)
from pyromind_runtime.domain.user_input import (
    UserInputOption,
    UserInputPresentation,
    UserInputQuestion,
    UserInputRequest,
)
from pyromind_runtime.domain.versions import SnapshotSource, fingerprint


async def load_reward_candidate(
    request: StartRewardReview, read: Callable[[str], Awaitable[str | None]]
) -> dict[str, str]:
    directory = PurePosixPath(request.candidate_directory)
    if (
        directory.is_absolute()
        or ".." in directory.parts
        or "\\" in request.candidate_directory
        or len(request.candidate_directory) > 256
        or "\x00" in request.candidate_directory
    ):
        raise ValueError("candidate must be a directory inside the rubric workspace")
    names = ["task_prompt.md", "rubric.json", "reward.py", "calibration_review.json"]
    if request.judge_kind != "code":
        names.append("calibration.json")
    files = {}
    for name in names:
        text = await read(str(PurePosixPath(REWARD_WORKDIR) / directory / name))
        if text is None:
            raise ValueError(f"Reward candidate is missing {name}")
        if len(text.encode("utf-8")) > MAX_REWARD_FILE_BYTES:
            raise ValueError(f"Reward candidate file exceeds 64 KiB: {name}")
        files[name] = text
    return files


def reward_confirmation(
    manifest: RewardManifest, rubric: Rubric, task_prompt: str, manifest_digest: str
) -> UserInputRequest:
    items = "\n".join(
        f"- {item.criterion} (权重 {item.weight:g}"
        f"{'，硬约束' if item.hard_constraint else ''})"
        for item in rubric.items
    )
    evidence = "\n".join(
        f"- {case.case_id} · {case.score:g} · {case.summary}"
        for case in manifest.calibration
    )
    return UserInputRequest(
        presentation=UserInputPresentation(
            title="确认并冻结 Reward 包",
            content=(
                TextContent(text=f"### 任务描述\n\n{task_prompt}"),
                TextContent(
                    text=f"### Checklist\n\n{items}\n\n"
                    f"**聚合方式**\n\n{rubric.aggregation}"
                ),
                TextContent(
                    text=f"### 校准依据\n\n{evidence}\n\n"
                    f"评分类型 `{manifest.judge_kind}`\n\n"
                    f"运行要求 {manifest.requirements}\n\n"
                    f"版本 `{manifest_digest}`"
                ),
            ),
        ),
        questions=(
            UserInputQuestion(
                id=REWARD_QUESTION_ID,
                prompt="Project 创建后将固定使用这版评分标准。确认这份 Reward 包吗？",
                options=(
                    UserInputOption(id=APPROVE_REWARD, label="确认并冻结"),
                    UserInputOption(id="revise", label="继续修改"),
                ),
            ),
        ),
    )


def prepare_reward_snapshot(
    request: StartRewardReview, files: dict[str, str]
) -> tuple[SnapshotSource, str, UserInputRequest]:
    frozen, manifest, rubric, prompt = freeze_reward_files(request, files)
    digest = hashlib.sha256(frozen["manifest.json"].encode("utf-8")).hexdigest()
    return (
        SnapshotSource(files=frozen),
        digest,
        reward_confirmation(manifest, rubric, prompt, digest),
    )


def reward_target(conversation_id: str, cluster: str, digest: str) -> str:
    return (
        f"/.pyromind-agent/{conversation_id}/prepared/{fingerprint([cluster, digest])}"
    )


def new_approval_id() -> str:
    return "reward-" + secrets.token_hex(16)


def same_reward(bundle: RewardBundle | None, cluster: str, digest: str) -> bool:
    return (
        bundle is not None
        and bundle.cluster == cluster
        and bundle.manifest_digest == digest
    )
