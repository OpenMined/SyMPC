# -*- coding: utf-8 -*-
"""
    This package represents the MPC component for Syft
    For the moment it has some basic functionality, but more would come in the
    following weeks
"""

from pkg_resources import DistributionNotFound, get_distribution

from . import config  # noqa: 401
from . import encoder  # noqa: 401
from . import protocol  # noqa: 401
from . import session  # noqa: 401
from . import tensor  # noqa: 401

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound
