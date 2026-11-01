"""Sanity checks for the preprocessing pipeline. Run: python -m pytest tests -q
The data-dependent tests need raw/ and a completed scripts/build_dataset.py run."""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import cleaning, csi_decode, windowing  # noqa: E402

INDEX = ROOT / "windows/index.csv"
needs_data = pytest.mark.skipif(not INDEX.exists(), reason="run scripts/build_dataset.py first")


def pack(re, im, exp):
    """Inverse of the documented 4366c0 word layout, for a round-trip test."""
    w = (abs(re) << 18) | (abs(im) << 6) | (exp & 0x3F)
    w |= (re < 0) << 29
    w |= (im < 0) << 17
    return w


def test_decode_roundtrip():
    rng = np.random.default_rng(0)
    re = rng.integers(-2047, 2048, 256)
    im = rng.integers(-2047, 2048, 256)
    exp = rng.integers(-20, 10, 256)
    words = np.array([pack(int(a), int(b), int(e)) for a, b, e in zip(re, im, exp)], dtype=np.uint32)
    h = csi_decode.decode_complex(words[None])[0]
    np.testing.assert_allclose(h, np.fft.fftshift((re + 1j * im) * np.exp2(exp)))


def test_qc_rejects_each_rule():
    good = np.ones((1, 256))
    narrow = np.zeros((1, 256))
    narrow[0, :64] = 1.0
    bad = np.vstack([good, good, np.zeros((1, 256)), np.full((1, 256), np.nan), narrow])
    keep, counts = cleaning.qc_mask(bad, 0.005)
    assert keep.tolist() == [True, False, False, False, False]
    assert counts == {"nonfinite": 1, "all_zero": 1, "duplicate": 1, "partial_band": 1}


@pytest.mark.parametrize("n", [11, 59, 150, 500])
def test_window_shape(n):
    w = windowing.capture_window(np.random.default_rng(n).random((n, 256)))
    assert w.shape == (150, 256) and w.dtype == np.float32 and (w >= 0).all()


@needs_data
def test_windows_and_splits():
    rows = list(csv.DictReader(open(INDEX)))
    by_capture = {}
    for r in rows:
        by_capture.setdefault(r["capture_id"], set()).add(r["split"])
    assert all(len(s) == 1 for s in by_capture.values()), "a capture appears in both splits"
    assert len({r["window_id"] for r in rows}) == len(rows)
    for r in rows[:: max(1, len(rows) // 200)]:
        w = np.load(ROOT / r["path"])
        assert w.shape == (150, 256) and w.dtype == np.float32 and np.isfinite(w).all()
        assert (r["split"] == "validation") == ("local_validation" in r["path"])


@needs_data
def test_room_B_validation_frozen():
    names = (ROOT / "splits/room_B_validation.txt").read_text().split()
    assert len(names) == len(set(names)) == 125
