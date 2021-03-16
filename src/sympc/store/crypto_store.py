# stdlib
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List


class CryptoStore:
    _func_add_store: Dict[Any, Callable] = {}
    _func_get_store: Dict[Any, Callable] = {}

    def __init__(self):
        self.log_store: List[str] = []
        self.store: Dict[Any, Any] = {}

    def populate_store(
        self,
        op_str: str,
        primitives: Iterable[Any],
        *args: List[Any],
        **kwargs: Dict[Any, Any]
    ) -> None:
        populate_func = CryptoStore._func_add_store[op_str]
        populate_func(self.store, primitives, *args, **kwargs)

    def get_primitives_from_store(
        self, op_str: str, nr_instances: int = 1, *args, **kwargs
    ) -> List[Any]:
        retrieve_func = CryptoStore._func_get_store[op_str]
        primitives = retrieve_func(self.store, nr_instances, *args, **kwargs)
        return primitives
