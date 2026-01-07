from math import log10
def PSNR(mse, max_val):
    return 10*log10(max_val**2/mse)

