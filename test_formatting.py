#!/usr/bin/env python3
"""Test file with intentionally bad formatting to test auto-commit functionality."""

from typing import Any, Dict, List, Optional


def badly_formatted_function(param1: str, param2: int, param3: Optional[Dict[str, List[int]]] = None) -> bool:
    """This function has terrible formatting on purpose."""
    if param3 is None:
        param3 = {}

    result = []
    for key, value in param3.items():
        if isinstance(value, list):
            for item in value:
                if item > param2:
                    result.append(f"{param1}_{key}_{item}")

    return len(result) > 0


class BadlyFormattedClass:
    def __init__(self, name: str, data: Dict[str, Any]):
        self.name = name
        self.data = data

    def process(self) -> None:
        print(f"Processing {self.name} with {len(self.data)} items")


if __name__ == "__main__":
    test_data = {"numbers": [1, 2, 3, 4, 5], "letters": ["a", "b", "c"]}
    obj = BadlyFormattedClass("test", test_data)
    obj.process()
    result = badly_formatted_function("prefix", 3, {"test": [1, 2, 3, 4, 5]})
    print(f"Result: {result}")
