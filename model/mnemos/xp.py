try:
    import cupy as xp
    from cupy import ndarray
    GPU_ENABLED = True
except ImportError:
    import numpy as xp
    from numpy import ndarray
    GPU_ENABLED = False
