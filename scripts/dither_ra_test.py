"""
dither_ra_test.py
Find out why the L-500 does not keep an RA dither, by commanding an RA offset
two different ways and measuring whether either one sticks.

THE OBSERVATION THIS EXPLAINS (2026-08-23, trunk, 46 subs):
NINA is configured correctly -- Direct_Guider, DitherPixels 20,
DitherRAOnly false -- and its log shows RA pulses being issued on every dither
("Dither target from (0,0) to (3.27, 19.19) using guide durations of 0.115 and
0.674 seconds"), with RA targets spanning +/-17.8 px. Yet the achieved RA in
the FITS headers stays within +/-0.23 px on every frame, while Dec achieves its
full commanded +/-23 px. Nothing re-centres mid-run. So the RA command leaves
NINA and the offset does not survive at the mount.

THE HYPOTHESIS: NINA's Direct Guider dithers with ASCOM PulseGuide, which is a
temporary RATE change. The L-500 is a direct-drive mount servoing to an
absolute encoder position against its tracking model, so in RA the servo
returns the axis to the model-predicted position and the nudge is erased. Dec
has no moving target to be pulled back toward, so a Dec nudge simply persists.

If that is right, PWI4's OWN offset API should behave differently:
mount_offset(ra_add_arcsec=...) moves the TARGET COORDINATE rather than the
rate, so the servo tracks to the offset position and holds it. That is the
thing worth knowing, because it is also the fix -- dither through PWI4's offset
API instead of ASCOM pulse guiding.

So this runs two tests and a control:
    A  PWI4 mount_offset(ra_add_arcsec)    -- expect: offset HOLDS
    B  PWI4 mount_offset(dec_add_arcsec)   -- control, known to work
    C  ASCOM PulseGuide in RA              -- what NINA does; expect: erased
       (skipped unless --ascom; needs the driver free, so not while NINA holds it)

SAFETY. This MOVES THE MOUNT, so the absolute rule applies: the roof must be
confirmed OPEN first, or a slewing scope can intersect the roof's travel path.
The check is enforced below and the script aborts if the roof cannot be
confirmed -- `--force` exists but should only be used when you are standing
there looking at an open roof. Offsets are tiny (30 arcsec, 0.008 deg) and
every one is reset in a finally block, so the mount ends where it started.

Usage (roof open, mount powered, PWI4 connected and TRACKING a target):
    python scripts/dither_ra_test.py
    python scripts/dither_ra_test.py --arcsec 60 --ascom
"""
import argparse
import os
import sys
import time

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# 30 arcsec is ~6x the actual 20 px dither (20 px x 0.26"/px = 5.2") so the
# result is unambiguous against pointing jitter, while still being a motion of
# 0.008 deg -- far below anything that changes the collision picture.
DEFAULT_ARCSEC = 30.0
SETTLE_S = 6.0
SAMPLES = 5


def _read(pwi4, n=SAMPLES, gap=0.4):
    """(ra_deg, dec_deg, scatter_arcsec) averaged over n samples."""
    import statistics as st
    ras, decs = [], []
    for _ in range(n):
        s = pwi4.status()
        ras.append(s.mount.ra_j2000_hours * 15.0)
        decs.append(s.mount.dec_j2000_degs)
        time.sleep(gap)
    import math
    dec0 = math.radians(st.mean(decs))
    sc = (st.pstdev(ras) * math.cos(dec0) * 3600 if len(ras) > 1 else 0.0)
    return st.mean(ras), st.mean(decs), sc


def _sep_arcsec(ra0, dec0, ra1, dec1):
    """Separation split into the RA and Dec components, in arcsec."""
    import math
    d0 = math.radians((dec0 + dec1) / 2.0)
    return ((ra1 - ra0) * math.cos(d0) * 3600.0, (dec1 - dec0) * 3600.0)


def roof_is_open(force=False):
    """Confirm the roof is OPEN before any mount motion. Two independent reads."""
    if force:
        print("!! --force: skipping the roof check. Only valid if you can SEE "
              "the roof is open.")
        return True
    ok = False
    try:
        from sentry import kasa_state
        safe, closed, is_open, when = kasa_state.kasa_status()
        print("  inside Kasa cam: scope_safe=%s roof_open=%s roof_closed=%s"
              % (safe, is_open, closed))
        ok = bool(is_open)
    except Exception as e:
        print("  inside Kasa cam check failed: %s" % e)
    if not ok:
        try:
            from sentry import vision_safety
            parked, closed, is_open, _ = vision_safety.visual_status()
            print("  safety cam: parked=%s roof_open=%s roof_closed=%s"
                  % (parked, is_open, closed))
            ok = bool(is_open)
        except Exception as e:
            print("  safety cam check failed: %s" % e)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arcsec", type=float, default=DEFAULT_ARCSEC)
    ap.add_argument("--ascom", action="store_true",
                    help="also test ASCOM PulseGuide (needs the driver free; "
                         "close NINA first)")
    ap.add_argument("--ascom-progid", default=None,
                    help="ASCOM telescope ProgID, e.g. ASCOM.PWI4.Telescope")
    ap.add_argument("--force", action="store_true",
                    help="skip the roof-open check -- only with eyes on the roof")
    args = ap.parse_args()

    print("Roof check (the mount must not move with the roof shut):")
    if not roof_is_open(args.force):
        print("\nABORT: roof is not confirmed OPEN. Nothing was moved.")
        return 2
    print("  roof confirmed open.\n")

    from hardware_control.pwi4_client import PWI4
    pwi4 = PWI4()
    s = pwi4.status()
    if not s.mount.is_connected:
        print("PWI4 mount not connected -- connect it in PWI4 first.")
        return 2
    if not s.mount.is_tracking:
        # The whole question is whether an RA offset survives while the servo
        # is following a moving target. Parked or idle, it proves nothing.
        print("Mount is NOT TRACKING. Start tracking a target first -- an RA "
              "offset only means something while the mount is following the sky.")
        return 2
    print("mount tracking, alt %.2f az %.2f" % (s.mount.altitude_degs,
                                                s.mount.azimuth_degs))

    off = args.arcsec
    results = {}
    try:
        ra0, dec0, scatter = _read(pwi4)
        print("baseline RA %.6f deg  Dec %.6f deg   (jitter %.2f arcsec)\n"
              % (ra0, dec0, scatter))

        # --- A: PWI4 native RA offset (moves the TARGET, should hold)
        print("A) PWI4 mount_offset(ra_add_arcsec=%+.0f) ..." % off)
        pwi4.mount_offset(ra_add_arcsec=off)
        time.sleep(SETTLE_S)
        ra1, dec1, _ = _read(pwi4)
        dra, ddec = _sep_arcsec(ra0, dec0, ra1, dec1)
        results["pwi4_ra"] = dra
        print("   achieved dRA %+.1f arcsec (commanded %+.0f), dDec %+.1f\n"
              % (dra, off, ddec))
        pwi4.mount_offset(ra_reset=0)
        time.sleep(SETTLE_S)

        # --- B: control. Dec is known to work from the imaging data.
        print("B) PWI4 mount_offset(dec_add_arcsec=%+.0f)  [control] ..." % off)
        rab, decb, _ = _read(pwi4)
        pwi4.mount_offset(dec_add_arcsec=off)
        time.sleep(SETTLE_S)
        ra2, dec2, _ = _read(pwi4)
        dra2, ddec2 = _sep_arcsec(rab, decb, ra2, dec2)
        results["pwi4_dec"] = ddec2
        print("   achieved dDec %+.1f arcsec (commanded %+.0f), dRA %+.1f\n"
              % (ddec2, off, dra2))
        pwi4.mount_offset(dec_reset=0)
        time.sleep(SETTLE_S)

        # --- C: what NINA actually does.
        if args.ascom:
            print("C) ASCOM PulseGuide in RA (the path NINA's Direct Guider uses) ...")
            try:
                import win32com.client
                progid = args.ascom_progid or "ASCOM.PWI4.Telescope"
                tel = win32com.client.Dispatch(progid)
                if not tel.Connected:
                    tel.Connected = True
                print("   driver %s, CanPulseGuide=%s, guide rates RA %.4f Dec %.4f deg/s"
                      % (progid, tel.CanPulseGuide,
                         tel.GuideRateRightAscension, tel.GuideRateDeclination))
                rac, decc, _ = _read(pwi4)
                rate = tel.GuideRateRightAscension * 3600.0     # arcsec/s
                ms = int(1000.0 * off / max(rate, 1e-6))
                print("   PulseGuide east for %d ms (rate %.1f arcsec/s)" % (ms, rate))
                tel.PulseGuide(2, ms)          # 2 = guideEast
                t0 = time.time()
                while tel.IsPulseGuiding and time.time() - t0 < 30:
                    time.sleep(0.2)
                time.sleep(SETTLE_S)
                ra3, dec3, _ = _read(pwi4)
                dra3, ddec3 = _sep_arcsec(rac, decc, ra3, dec3)
                results["ascom_ra"] = dra3
                print("   achieved dRA %+.1f arcsec (commanded %+.0f), dDec %+.1f\n"
                      % (dra3, off, ddec3))
            except Exception as e:
                print("   ASCOM test skipped: %s: %s\n" % (type(e).__name__, e))
    finally:
        # Always hand the mount back where it started, even on Ctrl+C.
        try:
            pwi4.mount_offset(ra_reset=0, dec_reset=0)
            print("offsets reset.")
        except Exception as e:
            print("!! could not reset offsets: %s -- check PWI4" % e)

    print("\n---------------- verdict ----------------")
    tol = 0.4 * off
    pr, pd = results.get("pwi4_ra"), results.get("pwi4_dec")
    if pr is not None:
        held = abs(pr) >= tol
        print("PWI4 RA offset:  %s (%.1f of %.0f arcsec)"
              % ("HELD" if held else "ERASED", pr, off))
        if held:
            print("  => the mount CAN hold an RA offset. NINA's pulse-guide path\n"
                  "     is what fails, so dithering via PWI4's offset API would work.")
        else:
            print("  => even PWI4's own target offset does not survive in RA.\n"
                  "     That points at the mount/servo, not at how it is commanded.")
    if pd is not None:
        print("PWI4 Dec offset: %s (%.1f of %.0f arcsec)  [control]"
              % ("HELD" if abs(pd) >= tol else "ERASED", pd, off))
    if "ascom_ra" in results:
        a = results["ascom_ra"]
        print("ASCOM RA pulse:  %s (%.1f of %.0f arcsec)"
              % ("HELD" if abs(a) >= tol else "ERASED", a, off))
        print("  => this is the exact path NINA uses; ERASED here confirms the\n"
              "     imaging-data finding directly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
