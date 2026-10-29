"""How far the observatory air is above its dew point (2026-10-29).

Pure, and kept out of pegasus.py (which imports requests and the private
config at load) so it can be tested anywhere.
"""
import math
from typing import Optional


def dew_margin_c(temp_c: Optional[float], humidity: Optional[float]) -> Optional[float]:
    """Air temperature minus dew point (C), Magnus formula; None for a bad reading.

    The web chat ticker shows it beside the humidity (owner 2026-10-29): the
    night of 10-28 went from 3.7 C at dusk to 0.4 C at dawn."""
    if humidity is None or temp_c is None or not 0 < humidity <= 100:
        return None
    a, b = 17.62, 243.12
    g = math.log(humidity / 100.0) + a * temp_c / (b + temp_c)
    return temp_c - b * g / (a - g)
