from dataclasses import dataclass, field


@dataclass
class Config:
    """
    Config used for the Fixed Precision Encoder
    TODO: This might get populated with other values
    """

    encoder_precision: int = field()
    encoder_base: int = field()

    def __init__(self, encoder_precision: int = 4, encoder_base: int = 10) -> None:
        self.encoder_precision = encoder_precision
        self.encoder_base = encoder_base
