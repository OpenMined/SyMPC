"""Configuration information used for FixedPrecisionEncoder and ShareTensor."""

# stdlib
from dataclasses import dataclass
from dataclasses import field


@dataclass
class Config:
    """Session configuration.

    Attributes:
        encoder_base (int): Base for the encoder.
        encoder_precision (int): Precision for the encoder.
    """

    encoder_precision: int = field()
    encoder_base: int = field()

    def __init__(self, encoder_base: int = 2, encoder_precision: int = 16) -> None:
        """Session configuration.

        Config can be used inside a Session (see Session) to
        specify different options that can be used for the Fixed Precision Encoder.

        Args:
            encoder_base (int): Base for the encoder.
            encoder_precision (int): Precision for the encoder.
        """
        self.encoder_base = encoder_base
        self.encoder_precision = encoder_precision
