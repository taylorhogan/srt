# Iris Maintenance Manual

*v1, 2026-08-30. The inspection manual and dated schedule called for by
ARCHITECTURE_PLAN.md §6. Print this document; check boxes on the wall copy; record what
was actually done (and anything found) in `docs/upkeep.md`, which remains the running log.*

Every task is either **MANUAL** (hands and eyes in the observatory) or **AUTONOMOUS**
(a monitor that already runs by itself — your job is only to verify it is alive and act
when it flags). Most roof problems share one root cause: **the roof is heavy**. Adding
more wheels would lessen the strain on each wheel and reduce most of the track/guide
issues below — that remains the standing capital improvement.

---

## Schedule at a glance — Sep 2026 to Sep 2027

Work day is the **first Saturday of the month**. Monthly tasks every visit; the others
stack onto the months shown.

| Date | Monthly | Also due |
|---|---|---|
| **2026-09-05** | M1 M2 M7 M8 M11 A-check | **Quarterly:** M3 M9 M10 |
| **2026-10-03** | M1 M2 M7 M8 M11 A-check | **Semi-annual:** M5 M6 M12 · **Winter prep:** M4 |
| **2026-11-07** | M1 M2 M8 M11 A-check | Ice watch begins (M4 after every ice event) |
| **2026-12-05** | M1 M2 M8 M11 A-check | **Quarterly:** M3 M9 M10 |
| **2027-01-02** | M1 M2 M8 M11 A-check | |
| **2027-02-06** | M1 M2 M8 M11 A-check | |
| **2027-03-06** | M1 M2 M8 M11 A-check | **Quarterly:** M3 M9 M10 |
| **2027-04-03** | M1 M2 M7 M8 M11 A-check | **Semi-annual:** M5 M6 M12 · Bug season begins (M7 monthly thru Oct) |
| **2027-05-01** | M1 M2 M7 M8 M11 A-check | |
| **2027-06-05** | M1 M2 M7 M8 M11 A-check | **Quarterly:** M3 M9 M10 |
| **2027-07-03** | M1 M2 M7 M8 M11 A-check | |
| **2027-08-07** | M1 M2 M7 M8 M11 A-check | |
| **2027-09-04** | M1 M2 M7 M8 M11 A-check | **Quarterly:** M3 M9 M10 · renew this schedule for the next year |

**Event-driven (no date):** M4 after every ice storm · M10 flats refresh after *any*
disturbance of the optics · M6 collimation whenever the optics-trend monitor (A1) flags
· A2 audio labeling whenever a roof move is filed as anomalous.

---

## Manual tasks

### M1 — Roof track inspection  *(monthly, ~10 min)*

The track sometimes comes loose — the treads shift away from the gear, which causes the
motor to skip.

- ☐ Check the tightness of the track screws.
- ☐ Check the track keeps a uniform distance from the center beam along its length.
- ☐ Continue the ongoing program: replace self-tapping screws with 2.5" bolts and nuts
  (a few per visit).

*Symptoms of neglect:* motor skip during moves — which A2 (audio) and A3 (current
watchdog) will hear/feel before you do. An anomalous-move flag from either is a reason to
do M1 early.

### M2 — Roof guide wheels and brackets  *(monthly, ~10 min)*

The wheels sometimes catch, pull in, and can bend the brackets.

- ☐ Visually check that no bracket appears bent.
- ☐ Check the thick tape on the track screw heads — the screws can catch the wheels; the
  tape is the mitigation and it wears.
- ☐ Spin/inspect each wheel for flat spots or binding.

### M3 — Roof drive and lubrication  *(quarterly: Sep, Dec, Mar, Jun)*

- ☐ Inspect gear/tread engagement along the full travel.
- ☐ Lubricate the drive gear and wheel axles.
- ☐ Cycle the roof once (daytime, mount confirmed parked — `cycle_roof`) and listen; the
  move is auto-scored by A2/A3, so a clean pass here is a recorded baseline.

### M4 — Gasket, sill, and ice  *(winter prep in Oct; then after every ice event Nov–Mar)*

In winter, ice builds up on the ledge and pulls away the gasket.

- ☐ Inspect the gasket seat along the ledge; press back / re-adhere any lifted section.
- ☐ Clear ice from the sill after each storm before it freezes to the gasket.
- ☐ Standing to-do: build an overhang so water does not collect on the sill.

### M5 — Mirror inspection  *(semi-annual: Oct, Apr — clean only if needed)*

Inspect on schedule; **clean only when the inspection demands it** — cleaning risk
exceeds dust risk. When cleaning, in order, stopping at the first step that suffices:

1. Gently blow away any dust.
2. If there is residue, wipe with very wet towels.
3. Use alcohol with cotton swabs on stubborn areas only.

### M6 — Collimation check  *(semi-annual: Oct, Apr — and whenever A1 flags)*

- ☐ Review the optics-trend numbers first: `field_excess` above **0.45** is the one real
  drift signal (the sweet-spot clamp fakes trends; trust field_excess, not r).
- ☐ Star-test / check collimation only if flagged or if the semi-annual review shows a
  worsening trend. Never compare star counts across targets — they vary by field.
- ☐ After *any* adjustment of the optics: refresh flats (M10) and note the date here and
  in upkeep.md — older lights lose epoch-matched flats from that day forward.

### M7 — Bug control  *(monthly, Apr–Oct)*

- ☐ Check the film on the bug light.
- ☐ Put down the white powder outside.
- ☐ Spray the mint spray inside.
- ☐ Check for webs (camera windows, OTA opening, roof corners).

### M8 — Humidity control  *(monthly)*

- ☐ Confirm the dehumidifier runs (the end sequence turns it on) and its tank/drain is
  clear; clean its filter.
- ☐ Check/recharge desiccant where fitted.

### M9 — Power, cables, USB  *(quarterly: Sep, Dec, Mar, Jun)*

- ☐ Every Kasa plug answers discovery — if one vanishes, check TP-Link's *"allow
  third-party apps to control"* toggle **first**; firmware updates have silently turned
  it off before blaming wiring.
- ☐ Reseat check on camera USB runs; lesson learned: a changed port *number* does not
  mean a moved connector (USB2 vs SuperSpeed tiers share a socket).
- ☐ Visual pass over cable runs for chafe, strain, and rodent interest.

### M10 — Flats and calibration hygiene  *(quarterly: Sep, Dec, Mar, Jun — plus after any optics disturbance)*

- ☐ Run `purge` (dry-run), review, then `purge go` — masters across nights agree to
  0.2–0.3% RMS, so only the newest set per filter earns its ~1.2 GB/filter/night.
  This is the only command that deletes data outright; do it deliberately.
- ☐ If the optics were touched since the last flats (M5/M6, camera reseat, filter wheel
  work): shoot a fresh flat set **before** purging anything.
- ☐ Verify darks/bias masters still match the running sensor temperature (−20 °C).

### M11 — Disks and backups  *(monthly)*

- ☐ A dated `IrisBackupState` zip from within the last day exists on OneDrive and opens.
  (Dropbox does **not** sync on iris-pc — OneDrive is the one that counts.)
- ☐ Disk free on the image drive is sufficient for the coming month (flats are the
  usual eater — see M10).

### M12 — Cameras and sensors  *(semi-annual: Oct, Apr)*

- ☐ Inspect camera windows for dust/dew residue (main camera, guide, all-sky, Iris cam).
- ☐ **Never move the Kasa sky camera** — the plate solution depends on its pointing.
- ☐ If the safety/Iris camera has moved: the vision template positions must be
  re-measured before trusting roof/park detection — never separate a capture from its
  label by operator motion.

---

## Autonomous inspections (the "A-check": verify these are alive, monthly)

These run without you. The monthly **A-check** is one pass confirming each still reports —
a silent monitor is the failure mode to catch.

| ID | Monitor | Runs | Flags mean | Your action |
|---|---|---|---|---|
| **A1** | Optics drift — nightly FWHM trend, `field_excess` vs **0.45** | every imaging night | collimation drifting | do M6 now |
| **A2** | Roof audio anomaly — spectrogram MSE vs the good library | every roof move | mechanical change: skip, catch, chatter | do M1/M2 now; label leftovers with `audio <open\|close> <good\|bad>` |
| **A3** | Roof current stall watchdog — >20 W past 30 s cuts power + priority-2 push | every roof move | gear not engaged / obstruction | roof state is UNKNOWN — resolve on site before any further motion |
| **A4** | Vision safety — parked/roof template confirmation | every hardware action | contradiction or dark frame | check camera, lighting, and whether the camera moved (M12) |
| **A5** | Per-filter convergence (`snr`) — tail slope / RMSE per DSO per filter | post-night | which filters still need frames | feeds the night planner (§2c); nothing manual |
| **A6** | Sky monitors — rain detector, live skymap watchdog | continuous | weather in / feed stale | a bare "feed recovered" push = local outage; check IrisLiveSkymap exit codes before blaming the host |
| **A7** | State backup task — daily zip to OneDrive | daily | (verified by M11) | M11 |

**A-check procedure (monthly, ~5 min):** confirm each of A1–A7 has produced output since
the last work day — a log line, a journal note, a chat card, or a dated zip. Any monitor
with nothing to show is treated as **failed**, not as "nothing happened."

---

## Log

Record completed work days and findings in `docs/upkeep.md` (date, tasks done, anything
found, anything deferred). When a finding changes a procedure, edit the task card here in
the same commit.
