from dataclasses import dataclass, field


@dataclass
class Config:
    ring_size: int = field()
    min_value: int = field()
    max_value: int = field()

    enc_precision: int = field()
    enc_base: int = field()

    def __init__(
        self, ring_size: int = 2 ** 62, enc_precision: int = 4, enc_base: int = 10
    ):
        self.ring_size = ring_size
        self.min_value = -(ring_size // 2)
        self.max_value = (ring_size - 1) // 2

        self.enc_precision = enc_precision
        self.enc_base = enc_base
