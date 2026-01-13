try:
    import cupy as _xp
    from cupy import ndarray
    print("Using GPU with CuPy")
    GPU_ENABLED = True
except ImportError:
    print("Using CPU with NumPy")
    import numpy as _xp
    from numpy import ndarray
    GPU_ENABLED = False

xp = _xp