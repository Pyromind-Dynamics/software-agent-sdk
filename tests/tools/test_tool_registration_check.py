from pathlib import Path

from scripts.check_tool_registration import main


def test_product_tool_registration_uses_current_package_root():
    repo_root = Path(__file__).parents[2]
    browser_definition = (
        repo_root
        / "pyromind-common"
        / "pyromind_common"
        / "tools"
        / "sandbox"
        / "definition.py"
    )

    assert main([str(browser_definition)]) == 0
