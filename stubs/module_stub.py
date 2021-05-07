"""File containing a stub of the Module that can be found in the PySyft library.

The goal of this stub is to make testing easier.
* Remove the dependency on the Module class that can be found in the PySyft library.
* Solve the "circular dependency" between both libraries while testing.
  In the past, some changes on the SyMPC required changes in the PySyft, so that they could
  be reflected in the Module class. For that reason, tests in SyMPC failed as well as PySyft.
  Requiring to merge PySyft changes in the SyMPC library manually.

If you want to know how a real Module should look like, please check the one in PySyft library.
"""

# stdlib
from collections import OrderedDict
import copy

# third party
import torch

import sympc.module as sympc_module
from sympc.module.nn import Conv2d
from sympc.module.nn import Linear

MAP_TORCH_TO_SYMPC = {
    "Linear": Linear,
    "Conv2d": Conv2d,
}


def full_name_with_qualname(klass: type) -> str:
    """Retrieve Fully Qualified Name.

    Arguments:
        klass: Type of which we want to know the Fully Qualified Name

    Returns:
        The Fully Qualified Name
    """
    return f"{klass.__module__}.{klass.__qualname__}"


class Module:
    """Stub of the Module class that can be found in PySyft."""

    def __init__(self, torch_ref):
        """Call setup method to initialize the basic attributes."""
        self.setup(torch_ref=torch_ref)

    def setup(self, torch_ref):
        """Initialize the basic attributes this module requires to work.

        Arguments:
            torch_ref: Reference to the torch library.
        """
        self.torch_ref = torch_ref
        self._modules = OrderedDict()
        if "syft" in full_name_with_qualname(klass=type(torch_ref)):
            self.is_local = False
        else:
            self.is_local = True

    def __call__(self, *args, **kwargs):
        """Proxy method that will call the forward net method.

        Arguments:
            args: Python args
            kwargs: Python kwargs

        Returns:
            Result of forward function.
        """
        return self.forward(*args, **kwargs)

    def share(_self, session):
        """Share the module between the clients of the session.

        Arguments:
            session: Session where the module will be shared.

        Returns:
            Module to share.
        """
        mpc_module = copy.copy(_self)

        mpc_module._modules = OrderedDict()
        mpc_module.torch_ref = sympc_module

        for name, module in _self.modules.items():
            state_dict = module.state_dict()
            name_layer = type(module).__name__
            sympc_type_layer = MAP_TORCH_TO_SYMPC[name_layer]
            sympc_layer = sympc_type_layer(session=session)
            sympc_layer.share_state_dict(state_dict)
            mpc_module._modules[name] = sympc_layer

        return mpc_module

    def eval(self):
        """Set train to false.

        In this stub is not required to actually implement it.
        """
        return

    def __setattr__(self, name, value):
        """Catch modules being set during subclass init.

        We must catch moduels due to a bug where torch.nn.modules is not the full name on some
        imports.

        Arguments:
            name: Module name
            value: Module value
        """
        if "torch.nn" not in full_name_with_qualname(klass=type(value)):
            object.__setattr__(self, name, value)
            return

        modules = self.__dict__.get("_modules")
        if modules is not None:
            modules[name] = value

        real_module = self.__dict__.get("real_module")
        if real_module is not None:
            real_module.add_module(name, value)

    def __getattr__(self, name):
        """Get module value.

        Arguments:
            name: Module name

        Returns:
            Module value
        """
        modules = self.__dict__.get("_modules")
        if modules is not None and name in modules:
            return modules[name]

        return object.__getattribute__(self, name)

    @property
    def modules(self):
        """Check if modules exist and return it.

        Returns:
            Dicitonary with modules
        """
        modules = self.__dict__.get("_modules")
        return modules or OrderedDict()

    def send(self, client):
        """Send the model to one client of the session.

        Arguments:
            client: Client that will receive the model.

        Returns:
            Remote model
        """
        if not self.is_local:
            return

        remote_model = copy.copy(self)
        remote_model.setup(torch_ref=client.torch)
        remote_model.duet = client

        return remote_model

    def get(self):
        """Get the shared model.

        Returns:
            Shared model.
        """
        if self.is_local:
            return None

        local_model = copy.copy(self)
        local_model.setup(torch_ref=torch)
        local_model.duet = self.duet

        self.local_model = local_model
        return self.local_model

    def reconstruct(_self):
        """Get back the shares from all the parties and reconstruct the underlying value.

        Returns:
            Neural network module with the secret reconstructed
        """
        syft_module = copy.copy(_self)

        syft_module.real_module = None
        syft_module.torch_ref = torch

        for name, module in _self.modules.items():
            state_dict = module.reconstruct_state_dict()
            torch_module = type(module).get_torch_module(module)
            torch_module.load_state_dict(state_dict)

            setattr(syft_module, name, torch_module)

        return syft_module
