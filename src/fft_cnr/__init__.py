"""FFT-based contrast-to-noise ratio estimation from single frames."""

from fft_cnr.core import CNREstimate, fft_cnr

__all__ = ["CNREstimate", "fft_cnr"]
