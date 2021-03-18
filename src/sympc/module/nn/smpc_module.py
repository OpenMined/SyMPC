"""MPC Class that would serve as a base for the other layers."""

# third party
import torch


class SMPCModule:
    @staticmethod
    def eq_close(
        layer1: "SMPCModule",
        layer2: "SMPCModule",
        rtol: float = 1e-05,
        atol: float = 1e-08,
    ) -> bool:
        """Compare two SMPC modules.

        Args:
            layer1 (SMPCModule): the first operand for the comparison
            layer2 (SMPCModule): the second operand for the comparison
            rtol (float): relative tolerance used
            atol (float): the absolute tolerance

        Returns:
            True if the layers parameters are nearly equal or False if they are not
        """
        if (layer1.bias is None) != (layer2.bias is None):
            return False

        if not torch.allclose(layer1.weight, layer2.weight, rtol=rtol, atol=atol):
            return False

        if not torch.allclose(layer1.bias, layer2.bias, rtol=rtol, atol=atol):
            return False

        return True
