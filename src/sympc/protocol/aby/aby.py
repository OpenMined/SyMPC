
from sympc.protocol import Protocol
from sympc.tensor import MPCTensor


class ABY(metaclass=Protocol):


    @staticmethod
    def A2B(x: MPCTensor) -> MPCTensor:
        """
        Convert an Arithmetic Shared Value to a Binary Shared Value
        """



