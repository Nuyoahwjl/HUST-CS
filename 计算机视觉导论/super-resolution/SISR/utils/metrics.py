import numpy as np
from math import log10
from PIL import Image
from .img import rgb2y
from skimage.metrics import structural_similarity as ssim_func


def psnr_y(img_sr: Image.Image, img_hr: Image.Image, shave=4) -> float:
    y_sr = rgb2y(img_sr)
    y_hr = rgb2y(img_hr)
    if shave > 0:
        y_sr = y_sr[shave:-shave, shave:-shave]
        y_hr = y_hr[shave:-shave, shave:-shave]
    mse = np.mean((y_sr - y_hr) ** 2)
    return 99.0 if mse < 1e-10 else 10 * log10((255.0 ** 2) / mse)

def ssim_y(img_sr: Image.Image, img_hr: Image.Image, shave=4) -> float:
    y_sr = rgb2y(img_sr)
    y_hr = rgb2y(img_hr)
    if shave > 0:
        y_sr = y_sr[shave:-shave, shave:-shave]
        y_hr = y_hr[shave:-shave, shave:-shave]
    return float(ssim_func(y_hr, y_sr, data_range=255.0, channel_axis=None))
