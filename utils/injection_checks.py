"""Shared invariant for the synthetic-injection acceptance tests.

Both injection tests — ``transient_search/test_injection.py`` (supernovae,
brightening) and ``transit_search/test_injection.py`` (exoplanets, dimming) —
measure recovery against injected ground truth. The number they produce is a
completeness curve, and the useful assertion is NOT that the curve equals any
particular value.

Pinning values would be actively harmful. A regression test written against the
transient search on 2026-08-04 would have asserted "recovery = 2/21", and then
failed when the shape window was fixed — locking in a bug and calling the fix a
regression. Completeness is supposed to improve.

What cannot legitimately change is the SHAPE: a stronger signal must not be
recovered less often than a weaker one. That is physics, not tuning, so it
survives any amount of legitimate improvement. Every bug found on 2026-08-05
announced itself as a violation of it:

  * transient: 6400 ADU recovered 0/3 while 400 ADU recovered 2/3 — the shape
    window was rejecting real point sources and the bright-source mask was
    deleting the brightest transients;
  * transit: 0% at every depth including 4% — injections were landing on stars
    outside the searched set;
  * transit: 2% beating 4% — the test itself was injecting every transit at one
    shared epoch, which the common-mode detrending correctly removed.

Note the third: the invariant caught a bug in the test, not in the pipeline.
That is the point of asserting a physical law rather than a remembered number.
"""

from typing import Sequence


def check_monotonic_recovery(
    levels: Sequence[float],
    recovered: Sequence[int],
    per_level: int,
    label: str = "signal",
    fmt: str = "{:g}",
) -> tuple[bool, list[str]]:
    """Check that recovery does not fall as the injected signal strengthens.

    ``levels`` must be ordered weakest first, ``recovered[i]`` is the count
    recovered out of ``per_level`` at ``levels[i]``.

    Each level is compared against the best recovery achieved at any *weaker*
    level, allowing one injection of slack: at these sample sizes (typically 3
    per level) a single miss is ordinary binomial noise, whereas a genuine
    inversion — the kind produced by all three bugs above — is far larger.
    Widening ``per_level`` is what buys real statistical power; this tolerance
    only avoids crying wolf at the sample size the tests actually use.

    Returns ``(ok, messages)``. Messages are empty when the curve is sound.
    """
    if len(levels) != len(recovered):
        return False, [f"levels/recovered length mismatch: "
                       f"{len(levels)} vs {len(recovered)}"]
    if per_level <= 0:
        return False, ["per_level must be positive"]

    problems: list[str] = []

    # A curve that is flat zero everywhere is not a sensitivity floor. It means
    # nothing reached the detector at all — a plumbing fault, and the loudest
    # thing this check can tell you.
    if not any(recovered):
        problems.append(
            f"nothing recovered at ANY {label} level — this is a broken pipeline "
            f"or a broken test, not a sensitivity limit")
        return False, problems

    # Compared as integer counts, never as fractions. In floating point
    # 2/3 < 1.0 - 1/3 is True, which failed a legitimate one-injection noise
    # dip the first time this was written.
    best_i = 0
    for i in range(1, len(levels)):
        if recovered[i] < recovered[best_i] - 1:
            problems.append(
                f"non-monotonic: {label} {fmt.format(levels[i])} recovered "
                f"{recovered[i]}/{per_level} but the weaker "
                f"{fmt.format(levels[best_i])} recovered "
                f"{recovered[best_i]}/{per_level} — a stronger signal cannot be "
                f"harder to find")
        if recovered[i] > recovered[best_i]:
            best_i = i

    # Total failure at the strongest signal is never binomial noise, so it is
    # checked without slack. This is the case that matters most and the one the
    # one-injection tolerance above would otherwise wave through: the transit
    # test's shared-epoch bug produced [0,0,0,1,0], where the deepest transit
    # was found zero times while a shallower one was found once.
    if recovered[-1] == 0 and any(recovered[:-1]):
        strongest_found = max(range(len(levels) - 1), key=lambda j: recovered[j])
        problems.append(
            f"the strongest {label} tested, {fmt.format(levels[-1])}, was recovered "
            f"0/{per_level} while the weaker {fmt.format(levels[strongest_found])} "
            f"was recovered {recovered[strongest_found]}/{per_level} — the easiest "
            f"case failing outright is a defect, not scatter")

    if len(problems) > 1:
        problems.append(
            f"note: {per_level} injections per level gives little statistical "
            f"power; raise per-level counts before reading much into a single "
            f"one-injection wobble")
    return not problems, problems


def report(ok: bool, problems: list[str]) -> int:
    """Print the verdict and return a process exit code."""
    if ok:
        print("\n  MONOTONICITY: ok — recovery never falls as the signal strengthens")
        return 0
    print("\n  MONOTONICITY: FAILED")
    for p in problems:
        print(f"    - {p}")
    print("    A completeness curve that inverts is evidence of a defect in the\n"
          "    pipeline or in the injection itself. Do not read a detection limit\n"
          "    off it until it is monotonic.")
    return 1
