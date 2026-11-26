import subprocess
from pathlib import Path

from harness_adapter.pi_adapter import fixed_actor_resources
from harness_adapter.pi_adapter.actor_workspace import ActorTransport, ActorWorkspace
from harness_adapter.pi_adapter.fixed_actor_resources import prepare_actor_resources
from harness_adapter.pi_adapter.sandbox_runtime import resource_files


def test_actor_resources_follow_working_tree_and_reuse_mount_cache(
    tmp_path, monkeypatch
):
    mount = tmp_path / "mount"
    source = tmp_path / "knowledge"
    source.mkdir()
    obsolete = source / "old.md"
    obsolete.write_text("legacy tool")
    document = source / "sdk.md"
    document.write_text("first")
    writes = []

    def write_file(self, path, content):
        writes.append(path)
        Path(path).write_bytes(content)

    def checked_command(self, command, cwd=None, timeout=30):
        result = subprocess.run(
            ["/bin/sh", "-c", command],
            cwd=cwd or self.working_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode:
            raise RuntimeError(result.stderr)
        return result.stdout

    monkeypatch.setattr(ActorWorkspace, "storage_path", property(lambda _: str(mount)))
    monkeypatch.setattr(ActorWorkspace, "checked_command", checked_command)
    monkeypatch.setattr(ActorTransport, "write_file", write_file)
    image = tmp_path / "image"
    monkeypatch.setattr(fixed_actor_resources, "_IMAGE_RESOURCES", str(image))

    def prepare(session, local=source):
        root = mount / session
        root.mkdir(parents=True, exist_ok=True)
        workspace = ActorWorkspace(
            working_dir=str(root), transport=ActorTransport("http://localhost", "test")
        )
        prepare_actor_resources(
            workspace, resource_files([("knowledge", local)], str(root))
        )
        return root

    first = prepare("first")
    (first / "user-script.py").write_text("pass")
    second_checkout = tmp_path / "second-checkout"
    second_checkout.mkdir()
    for file in source.iterdir():
        (second_checkout / file.name).write_bytes(file.read_bytes())
    second = prepare("second", second_checkout)
    assert len(writes) == 1
    assert (second / "knowledge/sdk.md").read_text() == "first"

    Path(writes[0]).write_bytes(b"broken archive")
    prepare("third")
    assert writes == [writes[0], writes[0]]

    obsolete.unlink()
    document.write_text("current SDK helper")
    prepare("first")
    assert len(writes) == 3
    assert (first / "knowledge/sdk.md").read_text() == "current SDK helper"
    assert not (first / "knowledge/old.md").exists()
    assert (first / "user-script.py").read_text() == "pass"
    assert (second / "knowledge/old.md").is_file()

    document.write_text("shipped in the new image")
    (image / "knowledge").mkdir(parents=True)
    (image / "knowledge/sdk.md").write_bytes(document.read_bytes())
    fourth = prepare("fourth")
    assert (fourth / "knowledge/sdk.md").read_bytes() == document.read_bytes()
    assert len(writes) == 3
