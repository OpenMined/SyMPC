
class Linear:

    __slots__ = ["in_features", "out_features", "bias"]

    def __init__(self, torch_ref, in_features: int, out_features: int, bias: bool = True) -> None:
        self.torch_ref = torch_ref
        self.linear_module = torch_ref.Linear(

