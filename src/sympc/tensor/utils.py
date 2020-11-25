from typing import Any


def ispointer(obj: Any) -> bool:
    if type(obj).__name__.endswith("Pointer") and hasattr(obj, "id_at_location"):
        return True
    return False
