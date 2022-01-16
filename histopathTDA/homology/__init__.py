"""
Diagram
=====

API for generating persistence diagrams

Provides
  1. Cubical complex persistence diagrams of BW-images
"""
# License: TODO


from .diagram import *
from .landscape import *
from .pif import *

from .diagram import Diagram
from .landscape import Landscape
from .pif import PIF
from .distances import wassertein
from .distances import l_norm
from .histogram import color_hist
from .histogram import avg_hist

__all__ = [
    'Diagram',
    'Landscape',
    'PIF',
    'wassertein',
    'l_norm',
    'color_hist',
    'avg_hist']
