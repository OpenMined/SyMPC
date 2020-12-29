"""
In this file there would be defined utils functions that might be used
int any module
"""

# stdlib
import asyncio
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import functools
from itertools import repeat
import operator
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union


def ispointer(obj: Any) -> bool:
    """Check if a given obj is a pointer (is a remote object)

    :return: True (if pointer) or False (if not)
    :rtype: bool
    """
    if type(obj).__name__.endswith("Pointer") and hasattr(obj, "id_at_location"):
        return True
    return False


def islocal(obj: Any) -> bool:
    """Check if the object is on the local machine (in Duet or VM)

    :return: True if yes, else False
    :rtype: bool
    """
    party_type = obj.client.class_name
    return party_type in {"VirtualMachineClient", "DomainClient"}


def parallel_execution(
    fn: Callable[..., Any],
    parties: Union[None, List[Any]] = None,
    cpu_bound: bool = False,
) -> Callable[..., List[Any]]:
    """Wraps a function such that it can be run in parallel at multiple
    parties

    Arguments:
        fn (Callable): the function to run
        parties (Clients from Syft): if this is set, then the function should be
            run remotely
        cpu_bound (bool): because of the GIL (global interpreter lock) sometimes
            it makes more sense to use processes than threads
            if it is set then processes should be used since they really
              run in parallel
            if not then it makes sense to use threads since there is no bottleneck
              on the CPU side

        :return: a function that runs in parallel at multiple parties or not
        :rtype: a Callable that returns a list of results
    """

    def initializer(event_loop):
        """Initializer used to set the same event loop to other
        threads/processes

        This is needed because there are new threads/processes started with
        the Executor and they do not have have an event loop set

        It is set manually here, to be the same as the main thread
        """
        asyncio.set_event_loop(event_loop)

    @functools.wraps(fn)
    def wrapper(
        args: List[List[Any]],
        kwargs: Optional[Dict[Any, Dict[Any, Any]]] = None,
    ) -> List[Any]:
        """The wrapper function that does sanity checks and checks
        what executor should be used
        """

        Executor: Union[Type[ProcessPoolExecutor], Type[ThreadPoolExecutor]]
        if cpu_bound:
            Executor = ProcessPoolExecutor
        else:
            Executor = ThreadPoolExecutor

        # Each party has a list of args and a dictionary of kwargs
        nr_parties = len(args)

        if args is None:
            args = [[] for i in range(nr_parties)]

        if kwargs is None:
            kwargs = {}

        if parties:
            func_name = f"{fn.__module__}.{fn.__qualname__}"
            attr_getter = operator.attrgetter(func_name)
            funcs = [attr_getter(party) for party in parties]
        else:
            funcs = list(repeat(fn, nr_parties))

        futures = []
        loop = asyncio.get_event_loop()

        with Executor(
            max_workers=nr_parties, initializer=initializer, initargs=(loop,)
        ) as executor:
            for i in range(nr_parties):
                _args = args[i]
                _kwargs = kwargs.get(i, {})
                futures.append(executor.submit(funcs[i], *_args, **_kwargs))

        local_shares = [f.result() for f in futures]

        return local_shares

    return wrapper
