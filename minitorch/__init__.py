"""MiniTorch is a minimalistic deep learning framework for educational purposes.
It provides basic functionalities for tensor operations, automatic differentiation,
and building neural networks.
"""

from .testing import MathTest, MathTestVariable  # type: ignore # noqa: F401,F403
from .module import *  # noqa: F401,F403
from .testing import *  # noqa: F401,F403
from .datasets import *  # noqa: F401,F403
