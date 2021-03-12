"""Configuration information used for FixedPrecisionEncoder and ShareTensor."""

# stdlib
from dataclasses import dataclass
from dataclasses import field


@dataclass
class Config:
    """Config is a class used inside a Session (see Session) that specifies
    diffenret options that can be used for the Fixed Precision Encoder.

    Arguments:
        encoder_base (int): the base for the encoder
        encoder_precision (int): the precision for the encoder
    """

    encoder_precision: int = field()
    encoder_base: int = field()

    def __init__(self, encoder_base: int = 2, encoder_precision: int = 16) -> None:
        """Initializer for the Config."""
        self.encoder_base = encoder_base
        self.encoder_precision = encoder_precision
