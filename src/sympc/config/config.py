from dataclasses import dataclass, field

@dataclass
class Config:
    min_value: int = field()
    max_value: int = field()
    ring_size: int = field()

    def __init__(self, ring_size: int = 2**62):
        self.ring_size = ring_size
        self.min_value = -(self.ring_size // 2)
        self.max_value = (self.ring_size - 1) // 2
