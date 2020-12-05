from dataclasses import dataclass, field


@dataclass
class Config:
    """
    Config is a class used inside a Session (see Session) that specifies
    diffenret options that can be used for the Fixed Precision Encoder

    Arguments:
        encoder_precision (int): the precision for the encoder
        encoder_base (int): the base for the encoder
    """

    encoder_precision: int = field()
    encoder_base: int = field()

    def __init__(self, encoder_precision: int = 16, encoder_base: int = 2) -> None:
        self.encoder_precision = encoder_precision
        self.encoder_base = encoder_base
