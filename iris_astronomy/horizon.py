"""The observatory's tree line (configs/my.hrz), read the way N.I.N.A reads it.

Pure: no astropy, no config, so the reading itself is testable anywhere (the
CI runner installs only pytest). astro_dso_visibility.get_horizon_from_azimuth
delegates here.
"""


def altitude_at(this_az: float, az, al) -> float:
    """Horizon altitude at *this_az*, interpolated LINEARLY between the my.hrz
    points and wrapping 360 -> 0, which is how N.I.N.A reads the same file for
    its WaitUntilAboveHorizon.

    Until 2026-10-19 the planner returned the LEFT point's altitude (a
    staircase). Where the tree line climbs with azimuth that reads it low: at
    NGC 1961's azimuth 28 deg the staircase gave 44.1 deg against N.I.N.A's
    50.6, so the planner counted the 00:00 hour as clear, the generator
    measured no wait and did not park, and N.I.N.A then waited behind the
    trees until ~01:00 with the mount tracking the previous target.
    """
    if not az:
        return 0.0
    a = float(this_az) % 360.0
    pts = sorted(zip((float(x) for x in az), (float(y) for y in al)))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    # Wrap: the segment from the last point through 360 to the first.
    xs = [xs[-1] - 360.0] + xs + [xs[0] + 360.0]
    ys = [ys[-1]] + ys + [ys[0]]
    for i in range(len(xs) - 1):
        if xs[i] <= a <= xs[i + 1]:
            span = xs[i + 1] - xs[i]
            if span <= 0:
                return ys[i + 1]
            return ys[i] + (ys[i + 1] - ys[i]) * (a - xs[i]) / span
    return ys[-1]
