"""Freeze the template helpers and an agent-authored driver into one Python file."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def bundle_pipeline(driver: Path, output: Path) -> str:
    helpers = [
        Path(__file__).with_name(name)
        for name in ("template_synthesis.py", "reference_synthesis.py")
    ]
    if output.resolve() in {driver.resolve(), *(p.resolve() for p in helpers)}:
        raise ValueError("bundle output must not overwrite its sources")
    driver_source = driver.read_text(encoding="utf-8")
    compile(driver_source, str(driver), "exec")
    driver_hash = hashlib.sha256(driver_source.encode()).hexdigest()
    source = (
        f"# driver_sha256: {driver_hash}\n"
        "import sys as _bundle_sys\n"
        "import types as _bundle_types\n"
    )
    for helper in helpers:
        helper_source = helper.read_text(encoding="utf-8")
        compile(helper_source, str(helper), "exec")
        helper_hash = hashlib.sha256(helper_source.encode()).hexdigest()
        source += (
            f"# {helper.stem}_sha256: {helper_hash}\n"
            f"_bundle_module = _bundle_types.ModuleType({helper.stem!r})\n"
            f"_bundle_module.__file__ = __file__ + '::{helper.stem}'\n"
            f"_bundle_sys.modules[{helper.stem!r}] = _bundle_module\n"
            f"exec(compile({helper_source!r}, _bundle_module.__file__, 'exec'), "
            "_bundle_module.__dict__)\n"
        )
    source += f"exec(compile({driver_source!r}, __file__, 'exec'), globals())\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(source, encoding="utf-8")
    return hashlib.sha256(source.encode()).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("driver", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(bundle_pipeline(args.driver, args.output))
