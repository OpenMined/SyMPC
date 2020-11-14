from dataclasses import dataclass, field

@dataclass(order=True, frozen=True)
class Config:
    fp_encoder_base: int = field(default=10)
    fp_encoder_precision: int = field(default=4)

    ring_size: int = field(default=2 ** 32)
    min_value: int = field(init=False)
    max_value: int = field(init=False)

    def __post_init__(self):
        super().__setattr__("min_value", -(self.ring_size // 2))
        super().__setattr__("max_value", (self.ring_size - 1) // 2)
