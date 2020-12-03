from typing import Any
from typing import List
from typing import Dict
from typing import Union
from typing import Callable

from itertools import repeat
import functools
import operator

from concurrent.futures import ThreadPoolExecutor, wait


def ispointer(obj: Any) -> bool:
    if type(obj).__name__.endswith("Pointer") and hasattr(obj, "id_at_location"):
        return True
    return False


def isvm(party: Any) -> bool:
    party_type = party.class_name
    if "VirtualMachine" in party_type:
        return True
    return False


def parallel_execution(
    fn: Callable[..., Any], parties: Union[None, List[Any]] = None
) -> Callable[..., List[Any]]:
    @functools.wraps(fn)
    def wrapper(
        args: Union[None, List[List[Any]]] = None,
        kwargs: Union[None, Dict[Any, Dict[Any, Any]]] = None,
    ) -> List[Any]:

        # Each party has a list of args and a dictionary of kwargs
        nr_parties = len(args)

        if args is None:
            args = [[] for i in range(nr_parties)]

        if kwargs is None:
            kwargs = {}

        funcs = None
        if parties:
            func_name = f"{fn.__module__}.{fn.__qualname__}"
            attr_getter = operator.attrgetter(func_name)
            funcs = [attr_getter(party) for party in parties]
        else:
            funcs = list(repeat(fn, nr_parties))

        futures = []
        with ThreadPoolExecutor(
            max_workers=nr_parties, thread_name_prefix=fn.__name__
        ) as executor:

            for i in range(nr_parties):
                _args = args[i]
                _kwargs = kwargs.get(i, {})

                futures.append(executor.submit(funcs[i], *_args, **_kwargs))

        local_shares = [f.result() for f in futures]

        return local_shares

    return wrapper
