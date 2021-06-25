"""Export the encoders that are currently implemented.

For the moment, there is only:
- FixedPointEncoder: convert a float precision value
  to a fixed precision one
"""

from .fp_encoder import FixedPointEncoder

__all__ = ["FixedPointEncoder"]
