"""Decode BCM4366c0 packed CSI words into complex CSI and amplitude.

The 4366c0 reports each subcarrier as one 32-bit word in a block-floating-point form:

    bit 29        sign of real part
    bits 28..18   real mantissa (11 bits, magnitude)
    bit 17        sign of imaginary part
    bits 16..6    imaginary mantissa (11 bits, magnitude)
    bits 5..0     shared exponent (6-bit two's complement)

    H = (+/-real_mantissa + j * +/-imag_mantissa) * 2**exponent

We keep the exponent exactly (no per-packet rescaling), so amplitudes are on one
physical scale within a capture. Words arrive in FFT order (DC first); fftshift puts
them in frequency order (lowest subcarrier first, DC null in the middle), which is the
order used by the Experiment-3 CSV files.
"""
import numpy as np

MANT_MASK = (1 << 11) - 1


def decode_complex(words):
    """(n, 256) uint32 -> (n, 256) complex128 in frequency order."""
    w = words.astype(np.int64)
    re = (w >> 18) & MANT_MASK
    im = (w >> 6) & MANT_MASK
    re = np.where(w & (1 << 29), -re, re)
    im = np.where(w & (1 << 17), -im, im)
    exp = w & 0x3F
    exp = np.where(exp >= 32, exp - 64, exp)
    h = (re + 1j * im) * np.exp2(exp)
    return np.fft.fftshift(h, axes=-1)


def amplitude(h):
    return np.abs(h)
