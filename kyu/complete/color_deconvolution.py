import numpy as np
from skimage.util import dtype

def color_deconvolution(rgb, conv_matrix):
    rgb = _prepare_colorarray(rgb, force_copy=True)
    np.maximum(rgb, 1E-6, out=rgb)  # avoiding log artifacts
    log_adjust = np.log(1E-6)  # used to compensate the sum above
    stains = (np.log(rgb) / log_adjust) @ conv_matrix
    return stains

def _prepare_colorarray(arr, force_copy=False):
    arr = np.asanyarray(arr)
    if arr.shape[-1] != 3:
        raise ValueError("Input array must have a shape == (..., 3)), "
                         f"got {arr.shape}")
    return dtype.img_as_float(arr, force_copy=force_copy)