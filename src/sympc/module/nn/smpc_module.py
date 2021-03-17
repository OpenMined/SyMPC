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
        if (layer1.bias is None) != (layer2.bias is None):
            return False

        if not torch.allclose(layer1.weight, layer2.weight, rtol=rtol, atol=atol):
            return False

        if not torch.allclose(layer1.bias, layer2.bias, rtol=rtol, atol=atol):
            return False

        return True
