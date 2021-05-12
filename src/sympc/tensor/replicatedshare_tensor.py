
from .tensor import SyMPCTensor
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Union


PROPERTIES_NEW_SHARE_TENSOR: Set[str] = {"T"}
METHODS_NEW_SHARE_TENSOR: Set[str] = {"unsqueeze", "view", "t", "sum", "clone"}

class ReplicatedShareTensor(metaclass=SyMPCTensor):
    """Replicated share tensor is used when a party  holds more than a single shares. 
       Protocols such as Falcon require it.
       
       Arguments:
          session (Session): the session
          shares: The shares held by the party
          
       Attributes:
          shares: The shares held by the party

    """
       
    
    AUTOGRAD_IS_ON: bool = True

    # Used by the SyMPCTensor metaclass
    METHODS_FORWARD = {"numel", "t", "unsqueeze", "view", "sum", "clone"}
    PROPERTIES_FORWARD = {"T"}
    
    def __init__(self,shares=None,session=None):
        
        self.shares=shares
    
        
    def add(self,y):
        
        pass
    
    def radd(self,y):
        
        pass
    
    def sub(self,y):
        
        pass
    
    def rsub(self,y):
        
        pass
    
    def mul(self,y):
        
        pass
    
    def div(self,y):
        
        pass
    
    def matmul(self,y):
        
        pass
    
    def rmatmul(self,y):
        
        pass
    
    def xor(self,y):
        
        pass
    
    
    def le(self,y):
        
        pass
    
    def ge(self,y):
        
        pass
    
    def eq(self,y):
        
        pass
    
    def ne(self,y):
        
        pass

        
    @staticmethod
    def hook_property(property_name: str) -> Any:
        """Hook a framework property (only getter).
        Ex:
         * if we call "shape" we want to call it on the underlying tensor
        and return the result
         * if we call "T" we want to call it on the underlying tensor
        but we want to wrap it in the same tensor type
        Args:
            property_name (str): property to hook
        Returns:
            A hooked property
        """

        def property_new_share_tensor_getter(_self: "ReplicatedShareTensor") -> Any:
            tensor = getattr(_self.tensor, property_name)
            res = ReplicatedShareTensor(session=_self.session)
            res.tensor = tensor
            return res

        def property_getter(_self: "ReplicatedShareTensor") -> Any:
            prop = getattr(_self.tensor, property_name)
            return prop

        if property_name in PROPERTIES_NEW_SHARE_TENSOR:
            res = property(property_new_share_tensor_getter, None)
        else:
            res = property(property_getter, None)

        return res

    @staticmethod
    def hook_method(method_name: str) -> Callable[..., Any]:
        """Hook a framework method such that we know how to treat it given that we call it.
        Ex:
         * if we call "numel" we want to call it on the underlying tensor
        and return the result
         * if we call "unsqueeze" we want to call it on the underlying tensor
        but we want to wrap it in the same tensor type
        Args:
            method_name (str): method to hook
        Returns:
            A hooked method
        """

        def method_new_RS_tensor(
            _self: "ReplicatedShareTensor", *args: List[Any], **kwargs: Dict[Any, Any]
        ) -> Any:
            method = getattr(_self.tensor, method_name)
            tensor = method(*args, **kwargs)
            res = ReplicatedShareTensor(session=_self.session)
            res.tensor = tensor
            return res

        def method(
            _self: "ReplicatedShareTensor", *args: List[Any], **kwargs: Dict[Any, Any]
        ) -> Any:
            method = getattr(_self.tensor, method_name)
            res = method(*args, **kwargs)
            return res

        if method_name in METHODS_NEW_SHARE_TENSOR:
            res = method_new_RS_tensor
        else:
            res = method

        return res

    __add__ = add
    __radd__ = add
    __sub__ = sub
    __rsub__ = rsub
    __mul__ = mul
    __rmul__ = mul
    __matmul__ = matmul
    __rmatmul__ = rmatmul
    __truediv__ = div
    __xor__ = xor

    
    