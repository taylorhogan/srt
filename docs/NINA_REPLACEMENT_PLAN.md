# Plan: replace N.I.N.A (and move the observatory to Linux)

Status 2026-10-16: research done, nothing built. Owner's project 2 of 3
(auto-stretch, NINA→Linux, autonomy gaps). Next decisions for the owner:
ask PlaneWave for a current PWI4 Linux build + L-500 support; confirm the
Alnitak model and how the Gemini is connected; then Stage 0/1 below.
**Owner decision 2026-10-16: no second Windows box.** A PWI4-only Windows
PC or VM beside a Linux computer is not an acceptable fallback ("the system
is only stable as its weakest link"). The Linux move therefore depends on
PWI4 itself running on Linux; that test is the first gate, before Stage 4
work. Stages 0-3 (N.I.N.A out, on Windows) stand on their own either way.
The autofocus finding (built-in, not Hocus Focus) was spot-checked on
2026-10-16: 718 reports in %LOCALAPPDATA%\NINA\AutoFocus, AutoFocuserName "NINA".

Research date 2026-10-16. Read-only: no repo, N.I.N.A or hardware changes were made.

## Headline findings

1. **Every device except one has a working Linux path.** QHY600M with the camera-port wheel, the Optec Gemini focuser/rotator, the Optec Alnitak panel, the Pegasus UPBv3 and ASTAP all have INDI drivers, simple serial protocols, or native Linux builds.
2. **The one real risk is PWI4.** PlaneWave documents PWI4 running on Linux under Mono, and its changelog carries Linux fixes up to 4.1.3. But the current download page shows only one installer (4.1.6) and never mentions Linux. Since 4.1.5, PWI4 targets .NET Framework 4.8, and Mono is now community-maintained by WineHQ.
   - The INDI PlaneWave mount driver is not an alternative: it talks to PWI4's HTTP API, so PWI4 is still needed.
   - Worst-case fallback: keep a small Windows host or VM that runs only PWI4, and reach it over the LAN through its HTTP API, which the repo already uses.
3. **N.I.N.A's built-in autofocus runs, not Hocus Focus.** All 716 autofocus reports in `%LOCALAPPDATA%\NINA\AutoFocus` say `"AutoFocuserName":"NINA","StarDetectorName":"NINA"`. ADR 0004 (`docs/adr/0004-capture-interface.md`) says autofocus stays N.I.N.A's because "HocusFocus is genuinely good". The Hocus Focus plugin is installed but has never produced a report. That weakens the ADR's reason, and the 716 reports give a ready set to validate a Python V-curve against.
4. **No guiding is used.** The profile's guider is `Direct_Guider`: ASCOM PulseGuide through PWI4, with no guide camera and no PHD2. `scripts/dither_now.py` already replaces the dither through PWI4 offsets.
5. **Two devices can't be opened by two programs at once (QHY camera, Pegasus box).** So the practical strangle is mostly **per night** (a Python night or a N.I.N.A night), not per instruction inside one N.I.N.A night.
   - Anything that goes through PWI4 HTTP or Pegasus Unity HTTP can move out of N.I.N.A now.
   - The camera, wheel, autofocus, centering and flats move together.
6. The repo already has a strangle plan: `docs/ARCHITECTURE_PLAN.md` §2e, which says centering first, autofocus next, camera last. This report agrees on direction but changes the order to fit the exclusive devices (Part C).

---

## Part A. What N.I.N.A does for us

### A1. Files in use

- **Config:** `configs/config_public.py:287-334`, the `"nina"` block.
  - `sequence_input`: `C:/Users/iriso/Documents/N.I.N.A/Sequences/cdk_full_sequence_4block.json`
  - `sequence_output`: `.../Sequences/full_for_tonight.json`
  - `image_dir`: `.../N.I.N.A/Targets`
  - `autofocus_reports`: `%LOCALAPPDATA%/NINA/AutoFocus`
  - `focus_model`: `local/focus_model.json`
  - `filter_wheel`: L0 R1 G2 B3 S-II4 O-III5 Ha6
  - `arc_sec_per_pixel`: 0.26
  - `max_slots`: 3
  - `configs/config.py:15` maps `image_dir` per machine.
- **Launchers** (all use profile `873cfc5d-...` "cdk17", run with `-r -x`):
  - `scripts/on_nina.bat`: prelude
  - `scripts/image_nina1.bat`: main sequence
  - `scripts/nina_flats.bat`: flats
  - `scripts/kill_nina_pwi4.bat`: taskkill
- **Sequences actually launched**, in `C:\Users\iriso\Documents\N.I.N.A\Sequences\`:
  - `cdk_prelude_sequence.json`
  - `full_for_tonight.json`, generated from `cdk_full_sequence_4block.json`
  - `flat_sequence.json`
  - `darks_and_bias.json`
  - `no_imaging.json`, used on a no-imaging night
- **Generator:** `nina_gen/nina_sequence_gen.py` (1,400 lines) plus `nina_gen/sequence_lint.py`. It writes:
  - target name and RA/Dec
  - the filter plan, with name *and* wheel position, into 4 SmartExposure blocks
  - a `MoveFocuserByTemperature` seed before each block, from `fits_processing/focus_model.py`
  - an L filter before Center (`ensure_center_filter`)
  - a hard end per slot (TimeCondition)
  - multi-slot cloning (`generate_slots_sequence`), rotation, and park across a gap
  - a read-back check of the written file
- **Active profile:** `%LOCALAPPDATA%\NINA\Profiles\873cfc5d-1c2c-4fb4-9cac-74bc2564662f.profile`
- **Installed versions:** N.I.N.A 3.1.2.9001; PWI4 4.1.4.99.
- **Plugins installed:** Advanced API, Autofocus Report Analysis, Connector, ExoPlanets, Exposure Calculator, Ground Station, Hocus Focus, Lightbucket, Orbitals, Phd2 Tools, PlaneWave Tools, Remote Copy, Sequencer Powerups, Session Metadata, Web Session History Viewer.

### A2. Instruction types used, with counts

Counts are per file: full = `full_for_tonight.json` (the 4block template is the same except one fewer SwitchFilter); prelude; flats; dark = `darks_and_bias`; noimg = `no_imaging`. Containers, strategies and value types are left out.

| Type (short) | full | prelude | flats | dark | noimg | What it does for us | Device |
|---|---|---|---|---|---|---|---|
| Connector.ConnectEquipment | 9 | 2 | 6 | 7 | 9 | Connects camera, switch, rotator, wheel, focuser, telescope, guider, safety monitor, weather | all |
| Connector.DisconnectAllEquipment | – | – | 1 | – | – | Releases devices after flats | all |
| Switch.SetSwitchValue | 4 | 3 | 6 | 6 | 4 | Pegasus power ports on/off. Index 0-2 are camera, Gemini and fan; index 3 is also set off at the end | Pegasus UPBv3 (Unity ASCOM switch) |
| Telescope.FindHome / ParkScope / UnparkScope | 1/2/1 | – | – | – | 1/2/1 | Home, then park/unpark at start; park at end | L-500 via ASCOM.PWI4 |
| PlaneWaveTools.StartPwi4 / StopPwi4 | –/1 | 1/– | 1/1 | 1/1 | –/1 | Starts and stops the PWI4 process | PWI4 |
| PlaneWaveTools.HeaterControl | 4 | – | 4 | 4 | 4 | Heaters 1 and 2 at 50% at start, 0 at end | PWI4 (OTA heaters) |
| PlaneWaveTools.Tracking | 1 | – | – | – | – | Tracking on | PWI4 |
| Camera.CoolCamera | 1 | – | 1 | 1 | 1 | Cools to −10 °C | QHY600M |
| Rotator.MoveRotatorMechanical | 1 | – | 1 | 1 | – | Rotator to mechanical 0 (the generator can set other angles) | Optec Gemini rotator |
| Focuser.MoveFocuserAbsolute | 1 | – | 7 | 1 | 1 | Start focus 72325; per-filter focus for flats | Optec Gemini focuser |
| Focuser.MoveFocuserByTemperature | 4 | – | – | – | – | Per-filter seed, slope·T + intercept (written by `nina_gen`) | Gemini (reads its temperature) |
| Utility.WaitForTime | 2 | – | – | – | – | Waits for sunset/nautical dusk and the slot start | – |
| Utility.WaitForTimeSpan | 3 | 2 | 4 | 3 | 3 | Power-up settles | – |
| Utility.WaitUntilAboveHorizon | 1 | – | – | – | – | Waits for the target to rise | – |
| Utility.Annotation | 1 | 1 | – | – | 1 | Comment only | – |
| Platesolving.Center | 1 | – | – | – | – | Solve with ASTAP and re-slew until within 1′ (30 s exposure, 10 tries, blind failover) | camera + mount + ASTAP |
| Autofocus.RunAutofocus | 4 | – | – | – | – | One autofocus per filter block | camera + wheel + Gemini |
| Trigger AutofocusAfterFilterChange | 1 | – | – | – | – | Refocus on filter change | same |
| Trigger AutofocusAfterHFRIncreaseTrigger | 1 | – | – | – | – | Refocus at +15% HFR over 10 frames | same |
| Trigger AutofocusAfterTemperatureChangeTrigger | 1 | – | – | – | – | Refocus after a 2 °C change | same |
| Imaging.SmartExposure (+TakeExposure, LoopCondition) | 4 | – | – | – | – | Per-filter block: switch filter, N × 300 s LIGHT, dither | camera + wheel |
| FilterWheel.SwitchFilter | 6 | – | 7 | 2 | 1 | Selects the filter (name + position) | QHY wheel on the camera's 4-pin port |
| Trigger Guider.DitherAfterExposures + Guider.Dither | 4+4 | – | – | – | – | Dithers after each sub (20 px), but in effect only in Dec; see `scripts/dither_now.py` | Direct_Guider → PWI4 PulseGuide |
| Conditions TimeCondition / AboveHorizonCondition / LoopCondition | 1/1/4 | – | 0/0/7 | 0/0/2 | – | Slot hard end, horizon stop, sub counts | – |
| Utility.WaitLoopData | 2 | – | – | – | – | Data only | – |
| FlatDevice.ToggleLight / SetBrightness / TrainedFlatExposure / OpenCover / CloseCover | – | – | 15/8/7/7/7 | – | – | Panel flats per filter using N.I.N.A's trained exposure times (in the profile) | Optec Alnitak CoverCalibrator |
| Imaging.TakeExposure (FLAT/BIAS/DARK) | (4 LIGHT) | – | 7 FLAT | 2 BIAS/DARK | – | Calibration frames | camera |
| Imaging.TakeManyExposures | – | – | – | 2 | – | 50 bias, 20 × 300 s darks | camera |
| WhenPlugin.ExternalScript (Sequencer Powerups) | 3 | 4 | – | – | 3 | Runs our scripts: `start.bat` (mount power, which asks the conductor), `goforimagecheck.bat`, `set_imaging_state.bat DONE_PRELUDE`, `end.bat` (park check, roof close), `smessage.bat` | our Python |
| Utility.ExternalScript (core) | – | – | 1 | – | – | `set_imaging_state.bat DONE_FLATS` | our Python |
| WhenPlugin.EndSequence | 1 | – | – | – | 1 | Ends the run | – |
| GroundStation.SendToPushover | 2 | 2 | 2 | 2 | 1 | Pushover notices | – |
| RemoteCopy.RobocopyStart | – | – | – | – | 1 | Robocopy, only on the no-imaging night | – |

The main sequence, in order:

1. Start area: connect, heaters on, home, park/unpark, wait for dusk, rotator to 0, focuser to 72325, cool to −10 °C.
2. Target: wait until above the horizon, Pushover, tracking on, Center, then 4 × (temperature focus seed, autofocus, SmartExposure with dither).
3. Triggers: autofocus on filter change, HFR rise or temperature change.
4. Conditions: TimeCondition and above-horizon.
5. End: heaters off, park, Pegasus ports 0-3 off, `end.bat`, StopPwi4, Pushover, EndSequence.

### A3. Hardware chain (from the active profile, the ASCOM registry and the repo)

| Role | Device | How N.I.N.A reaches it | Already driven from Python without N.I.N.A? |
|---|---|---|---|
| Mount | PlaneWave **L-500** (`MountName=l500`) under a CDK17, FL 2939 mm, f/6.8 | `ASCOM.PWI4.Telescope` → PWI4 | **Yes.** `hardware_control/pwi4_client.py` / `pwi4_utils.py` use HTTP :8220 for goto, park, home, offsets, tracking, model, fans, heaters |
| Camera | **QHY600M** (`QHY600M-1fed9c4a932a3b979`), native QHY SDK in N.I.N.A, gain 0, offset 10, −10 °C, 3.76 µm | native SDK | No |
| Filter wheel | QHY wheel on the camera (profile wheel Id = the camera Id), 7 positions L,R,G,B,S-II,O-III,Ha | through the camera's SDK | No |
| Focuser | **Optec Gemini** focusing rotator (`ASCOM.OptecGemini.Focuser`). Not a PlaneWave EFA/Hedrick; `focus_model.py` docstring confirms | ASCOM | No (only the seed model) |
| Rotator | Optec Gemini (`ASCOM.OptecGemini.Rotator`), half range, reversed | ASCOM | No |
| Flat panel / cover | Optec Alnitak (`ASCOM.OptecAlnitak.CoverCalibrator`). Exact model (Flip-Flat, Flat-Man XL or remote dust cover) **not verified** | ASCOM | No |
| Power | **Pegasus Ultimate Powerbox v3** (registry `ASCOM.Device1.Switch` = "PegasusAstro Switch 1", `UPBv30008E29A`, firmware 1.4), through Pegasus **Unity** | Unity's ASCOM switch | **Yes.** `hardware_control/pegasus.py` uses Unity HTTP :32000 and already sends **raw UPB serial commands** (`P1:0`), reads ports and environment |
| Plate solver | **ASTAP** (`C:\Program Files\astap\astap.exe`, D50 database), blind failover to nova.astrometry.net | CLI | **Yes.** `transit_search/transit.py`, `photometry/lightcurve.py`, `photometry/cmd_diagram.py` and `transient_search/difference.py` shell out to ASTAP. `sentry/plate_solve.py` is our own solver for the Kasa sky camera |
| Guiding | none (`Direct_Guider`). PHD2 path is set but unused; no guide camera | ASCOM PulseGuide | Dither replacement exists (`scripts/dither_now.py`); per memory, switching to it is deferred |
| Safety monitor / weather | `ASCOM.GenericFile.SafetyMonitor`, OpenWeatherMap | – | Our own safety and weather logic owns this anyway |
| Roof, lights, dehumidifier, cameras | Shelly / Kasa | not N.I.N.A | Yes, all Python |

### A4. What our Python already does that N.I.N.A would otherwise own

- **Mount control and dither:** full PWI4 HTTP client. `dither_now.py` dithers with `mount_offset` and showed 100% held offsets, where N.I.N.A's pulse dither did not move RA.
- **Focus:** `fits_processing/focus_model.py` fits per-filter slope·T + intercept from 571+ N.I.N.A autofocus reports. Every report keeps its measure points (position, HFR, error) plus the hyperbolic, quadratic and trend fits, which is a ready replay set for a Python V-curve. `fits_processing/fitsfwhm.py` has Moffat FWHM checked against PixInsight.
- **Plate solving:** ASTAP CLI wrappers (above) and a Gaia point-match fallback. Centering would also need a loop: solve, `mount_offset` or re-goto, repeat.
- **Frame quality:** `fits_processing/frame_watcher.py` watches the image directory and writes FWHM, eccentricity and sky to `frame_stats.json`. This could replace N.I.N.A's HFR trigger.
- **Calibration:** `stacking/stacker.py` handles flats grouped by FITS FILTER, cached masters and purge. Only the *capture* of flats, darks and bias is N.I.N.A's.
- **Planning and sequencing:** scheduler, conductor (`iris/conductor/*`), `control/night_plan.py`, `control/slot_plan.py`, slot hard ends and safety guards. N.I.N.A is already only an executor.
- **Power:** Pegasus port read and write with read-back; Kasa and Shelly.
- **Notifications:** `utils/pushover.py` covers the Ground Station Pushover items.
- **Capture seam:** ADR 0004 and ARCHITECTURE_PLAN Phase 6, `CaptureSession` (cool/filter/expose/focus/dither).

**Windows-only items outside N.I.N.A** that a Linux move must also handle (about 108 hits in 35 .py files):
- `tasklist`/`taskkill` for NINA.exe (super_user_commands, conductor shadow, morning_check, conductor_drill)
- `.bat`/`.cmd` launchers
- Scheduled Tasks (S4U) → systemd timers
- `%LOCALAPPDATA%` paths
- the audio detector on WDM-KS (`sentry/audio_classify.py`, `kasa_audio.py`)
- hard-coded `C:/Users/...` paths in `config_public.py`

---

## Part B. Linux support (2026)

| Component | Linux status | Notes / sources |
|---|---|---|
| **PWI4 (L-500 mount)** | **Officially runs under Mono, but whether a current build exists is unconfirmed.** | PlaneWave's "Running PWI4 under Linux" (Jan 2022) gives tar.gz + `./run-pwi4`, Mono on CentOS 7 / Ubuntu 18.04, `dialout` for `/dev/ttyACM*` ([PDF](https://indilib.org/media/kunena/attachments/7104/RunningPWI4underLinux.pdf)). The changelog has Linux fixes (4.0.5 L-mount USB autodetect; 4.1.3 Linux crash and GUI fixes); 4.1.5 moved to .NET Framework 4.8; 4.1.9 beta adds headless rotator/focuser control ([CHANGELOG](https://planewave.com/files/software/PWI4/CHANGELOG.txt)). The download page lists only "PWI4 4.1.6", with no platform named ([software](https://planewave.com/software/), [updates](https://planewave.com/software-updates/)). A community Rocky Linux package mirrors the **4.1.4** Linux tarball "to ensure future availability" ([warwick-one-metre/pwi4](https://github.com/warwick-one-metre/pwi4)). A Cloudy Nights user reports it works on Ubuntu 18/20 ([thread](https://www.cloudynights.com/forums/topic/818353-planewave-l-series-w-linux/)). Mono moved to WineHQ in 2024; the last Microsoft patch release was Feb 2024 ([WineHQ](https://winehq.org/news/2024091701), [LWN](https://lwn.net/Articles/987465)). The HTTP API is the same code, so `pwi4_client.py` should work unchanged. Not verified: Mono + 4.1.4.99/4.1.6 + L-500 on a current Ubuntu. |
| INDI PlaneWave mount | Exists, `indi_planewave_telescope` by Jasem Mutlaq, (c) 2023 | **Talks to PWI4 over HTTP :8220** (`/mount/goto_ra_dec_j2000`, `/mount/park`), so it still needs PWI4. Mount only, no focuser or rotator ([source](https://raw.githubusercontent.com/indilib/indi/master/drivers/telescope/planewave_mount.cpp)). Shipped in libindi packages ([Alpine](https://pkgs.alpinelinux.org/package/v3.22/community/x86_64/libindi)). |
| **QHY600M + QHY wheel** | **Good** | QHY ships a Linux SDK (libqhyccd). INDI `indi_qhy_ccd` supports QHY cameras including the camera-port CFW ([INDI QHY](https://casite-460905.cloudaccess.net/individuals/devices/cameras/qhy.html), [QHY wheel](https://stellarmate.com/devices/filter-wheels/104-qhy-filter-wheel.html)). A community Alpaca server, `qhyccd-alpaca` (Rust, SDK 24.12.26), reports **QHY600M "Passed"** and **QHYCFW3L-SR "Passed"** ASCOM validation on Kubuntu 24.04 amd64 and Pi 4 arm64. v0.1.2 is from 2025-01-06; no pulse guide or live mode ([docs.rs](https://docs.rs/qhyccd-alpaca)). Rust bindings note newer SDKs need OpenCV and may link poorly on ARM ([qhyccd-rs](https://www.lib.rs/crates/qhyccd-rs)). The CFW3 is fully controlled through the camera's 4-pin port ([QHY CFW3](https://qhyccd.com/qhycfw3)). Python bindings are community only (ctypes against libqhyccd is the usual route). Practical issue: 61 MP USB3 downloads on Linux usually need `usbfs_memory_mb` raised (common INDI/QHY advice, not checked here). |
| **Optec Gemini focuser + rotator** | **Good** | INDI Gemini driver (libindi ≥ 1.5) handles both focuser and rotator, over serial or Ethernet, with temperature compensation ([INDI/StellarMate page](https://casite-757701.cloudaccess.net/devices/rotators/148-optec-gemini-focusing-rotator.html)). Optec publishes a plain-text command set, so a direct pyserial or TCP driver is small. Not checked: whether ours is on USB-serial or Ethernet. |
| **Optec Alnitak panel** | **Good** | INDI `indi_flipflat` covers Flip-Flat (cover + light) and Flat-Man (light only), with per-filter brightness presets ([INDI](https://casite-460905.cloudaccess.net/individuals/devices/covers-light-sources/alnitak-flip-flat-flat-man.html)). The Alnitak serial protocol is a handful of commands. |
| **Pegasus UPBv3** | **Unity: Windows only. Box: supported** | Pegasus FAQ: Unity is supported only on Windows ([Unity Platform](https://pegasusastro.com/unity-platform), [downloads](https://pegasusastro.com/download/)). INDI has a dedicated **`indi_pegasus_upb3`** (6 DC ports, 3 dew, 2 variable, 8 USB, focuser, relay, power and environment monitoring; Linux/macOS/BSD; "no known bugs") ([INDI driver page](http://drivers.indilib.org/observatory-controllers/pegasus-astro/pegasus-upbv3/pegasus-upbv3/)). The older `indi_pegasus_upb` knows only V1/V2 ([source](https://raw.githubusercontent.com/indilib/indi/master/drivers/power/pegasus_upb.cpp)); a 2025 FirstLight review said INDI lacked v3, which is now outdated. INDIGO `aux_upb3` is "Stable" but **simulator-tested only** ([README](https://github.com/indigo-astronomy/indigo/blob/master/indigo_drivers/aux_upb3/README.md)). Easiest path: our `pegasus.py` already speaks the UPB serial command set through Unity's raw-command passthrough, so switching it to pyserial on the USB-serial port is a small change. |
| **ASTAP** | **Good** | Native Linux .deb/.rpm/.tar.gz and a CLI-only zip for amd64 and aarch64 (CLI v2026.09.19, GUI v2026.10.14). D50 is installed today; the site suggests D80 for fields of 0.5° or less. Our field is about 0.69° × 0.46°, so D50 or D80 both work ([hnsky.org/astap](https://www.hnsky.org/astap.htm)). Local astrometry.net is also packaged on Linux. |
| **Autofocus** | Do it ourselves | Ekos has a mature autofocus (INDI). In Python: Astra uses `astrafocus` ([arXiv 2607.12898](https://arxiv.org/abs/2607.12898)), pyobs has focusing modules ([pyobs paper](https://www.frontiersin.org/journals/astronomy-and-space-sciences/articles/10.3389/fspas.2022.891486/text)), pyscope has `run_autofocus` ([pyscope](https://github.com/macro-consortium/pyscope)). Our setup is simple: 9 points, 2000-step spacing, 30 s bin 2, hyperbolic fit on STARHFR, R² ≥ 0.8, 4 attempts. With the Moffat FWHM code and 716 reports to replay, a Python V-curve is low-risk. |
| **Guiding / PHD2** | Not needed | No guider in use. PHD2 does run on Linux if ever wanted. |
| **N.I.N.A itself** | No official Linux build | Upstream remains Windows ([isbeorn/nina](https://github.com/isbeorn/nina)). The community fork PI.N.S. is a Linux port with 0 stars and no releases ([MikeS11/pins](https://github.com/MikeS11/pins)); not a stopgap to rely on. |
| **KStars/Ekos (INDI)** | Mature, Linux-native | The Ekos scheduler is built for robotic use and can be scripted over DBus ([Automating KStars & Ekos](https://openastronomy.substack.com/p/automating-kstars-and-ekos-part-1), [Ekos scheduler](https://stellarmate.com/support/ekos/scheduler.html)). Truly headless use is awkward (it is a GUI app). Useful as a **reference and a manual stopgap**, not as our sequencer. |
| **ASCOM Alpaca + alpyca** | Cross-platform | Alpyca 3.1.2 is the official Python Alpaca client, needs no Windows ASCOM platform, and can reach classic Windows ASCOM drivers through **ASCOM Remote** ([alpyca docs](https://ascom-standards.org/alpyca/), [GitHub](https://github.com/ASCOMInitiative/alpyca)). Native Alpaca for our devices: QHY only through the community `qhyccd-alpaca`; none found for PWI4, Gemini, Alnitak or UPBv3. On Windows today, ASCOM Remote would expose the existing Gemini, Alnitak, UPB switch and PWI4 ASCOM drivers as Alpaca. |
| **Python observatory frameworks** | References | Astra (SPECULOOS, Alpaca-based, SPIE 2026, autofocus, plate-solve pointing, safety supervision; repo URL not found) ([arXiv](https://arxiv.org/abs/2607.12898)); pyobs ([ASCL](https://ascl.net/2203.012)); pyscope ([PyPI](https://pypi.python.org/pypi/pyscope)). |
| **N.I.N.A Advanced API** (installed) | Windows bridge | `GET /v2/api/equipment/camera/capture` on :1888, with optional solve and streaming ([docs](https://bump.sh/christian-photo/doc/advanced-api/operation/operation-get-equipment-camera-capture.md)). Python could drive N.I.N.A's camera, wheel and focuser *during* the transition without a sequence file. Not checked: whether it is enabled in the profile. |

---

## Part C. Recommended staged migration

**Principle:** keep N.I.N.A installed and launchable as the per-night fallback until a full season has run without it, as ARCHITECTURE_PLAN §2e already says. Because the camera and the UPB serial port are exclusive, each night is either a Python night or a N.I.N.A night. Within N.I.N.A nights, move out only what goes through shared HTTP services: PWI4 and, while it stays, Unity.

### Stage 0: capture seam (Windows, already planned as Phase 6). M, about 1 week

`CaptureSession` protocol with `capture/nina.py` wrapping today's generate, launch and teardown, so a `capture/native.py` can be selected per night by config. Includes a FakeCapture for CI.

### Stage 1: take the non-camera steps out of N.I.N.A (Windows, N.I.N.A still images). S, 3–5 days + 2 nights

- The conductor does these instead of N.I.N.A: StartPwi4/StopPwi4, heaters (`pwi4_client.heaters_set`), FindHome/park/unpark/tracking (already asked through `_ask_mount`), Pegasus ports (`pegasus.py`), Pushover, and the `.bat` ExternalScripts.
- The generated sequence shrinks to: connect, cool, rotator, focus, Center, autofocus, SmartExposure.
- Switch the dither to `dither_now.py`. This is the deferred A/B (memory: dither still not working; PWI4-offset fix built, switch deferred).
- Risk is low: these are all HTTP calls N.I.N.A is already sharing.

### Stage 2: native device layer, bench only (Windows, N.I.N.A not running, daytime, roof shut, mount parked). M, 2–3 weeks

- **Camera + wheel:** QHY SDK through ctypes (`libqhyccd`, the same SDK on Windows and Linux). Covers cooling, gain/offset, readout mode, 16-bit FITS with the headers our pipeline expects (FILTER, EXPTIME, FOCPOS, FOCTEMP, OFFSET, RA/DEC/OBJECT, DATE-OBS), and wheel moves through the camera. The alternative is the `qhyccd-alpaca` server plus alpyca: a validated QHY600M + CFW3 path, but it adds a Rust process and has one maintainer.
- **Gemini focuser and rotator:** direct serial or TCP driver, or alpyca over ASCOM Remote on Windows. Needs position, move, temperature, and rotator mechanical moves.
- **Alnitak panel:** serial (light on/off, brightness, cover).
- **UPBv3:** replace Unity HTTP with pyserial. Unity must not be holding the port, so this switches together with dropping Unity, and N.I.N.A's switch then goes away; Stage 1 already removed N.I.N.A's switch use.
- Gate: FITS headers and statistics from the Python path match N.I.N.A frames of the same exposure (bias level, OFFSET, gain, sky_pedestal calibration).

### Stage 3: Python capture nights (Windows; N.I.N.A as the per-night fallback). M–L, 2–3 weeks dev + 6–10 clear nights

Order, each step A/B'd against N.I.N.A nights with FITS headers as the judge:

1. **Darks, bias and flats** first. No sky, no mount; uses the trained per-filter times from the profile and N.I.N.A's flat histogram target of 0.5 ± 0.1.
2. **Exposure loop + dither + slot timing** (TimeCondition / above-horizon equivalents live in the conductor).
3. **Centering:** ASTAP CLI solve of a 30 s L frame, then PWI4 `mount_offset` or re-goto until within 1′, with a retry cap. The 2026-09 incident in `nina_sequence_gen.py:1083` (ASTAP failed 8 solves) argues for keeping a Gaia point-match fallback.
4. **Autofocus:** V-curve around the `focus_model` seed (9 × 2000 steps, 30 s bin 2, hyperbolic fit, R² ≥ 0.8). Validate offline first by replaying the 716 N.I.N.A reports. Then add triggers: filter change (seed offsets may be enough), FWHM rise from `frame_watcher`, and 2 °C.

**Gate:** a season (or the owner's chosen number of weeks) with no N.I.N.A fallback used.

### Stage 4: Linux. M, about 2 weeks + supervised nights

- **Bench PWI4 on Linux first**, before buying into the move. Get the current Linux tarball and L-500 support statement from PlaneWave, then run Mono + PWI4 on the target box with the mount connected but parked.
  - This needs the owner and touches hardware, so it was not done here.
  - If PWI4 fails on Mono: keep the existing PC (or a small Windows VM) as a PWI4-only appliance. PWI4's HTTP API accepts remote connections (config option, changelog 4.0.3), and `pwi4_client.py` already takes `host`.
- Install the QHY SDK, INDI optional, ASTAP CLI + D50, and udev rules (QHY, `dialout` for the UPB, Gemini and Alnitak serial ports and L-500 ttyACM).
- Port the Windows-isms: scheduled tasks → systemd units and timers; `.bat` → shell/Python; `tasklist`/`taskkill` → process checks; the audio input path; `C:/` paths in config.
- Repeat the Phase 2 safety rehearsals (roof/mount authority, blind park, end.py close) on the new box before any night.

### Effort summary (rough, one developer + Claude)

| Stage | Dev | Validation | Risk |
|---|---|---|---|
| 0 Capture seam | ~1 wk | CI + 1 normal night | low |
| 1 Non-camera steps out | 3–5 d | 2 nights | low |
| 2 Native drivers (bench) | 2–3 wk | daytime bench | medium (camera USB robustness) |
| 3 Python capture nights | 2–3 wk | 6–10 clear nights, then a season with N.I.N.A as fallback | medium-high (whole nights at stake) |
| 4 Linux | ~2 wk | bench + supervised nights | **depends on PWI4/Mono** |

### Hard blockers and open questions

1. **PWI4 on Linux (the only potential hard blocker).** It is documented and has run (4.0.x–4.1.4) under Mono, but there is no current Linux download on the page, it now targets .NET 4.8, and Mono is now maintained by WineHQ. Ask PlaneWave for the 4.1.6+ Linux tarball and L-500 support. Fallback: a Windows PWI4 appliance over HTTP. No INDI or Alpaca path avoids PWI4.
2. **Pegasus Unity** has no Linux build. This is not a blocker: there is a native INDI `indi_pegasus_upb3`, and the serial protocol is already used by `pegasus.py`.
3. **N.I.N.A's flat "trained exposure" and the Advanced API** are conveniences that need replacing, not blockers.
4. **Could not verify:**
   - the exact Alnitak model (cover or panel only)
   - whether the Gemini connects by serial or Ethernet
   - whether the Optec ASCOM drivers allow two clients at once
   - whether the Advanced API is enabled
   - QHY SDK USB3 robustness for 61 MP frames on our Linux box
   - Astra's repository URL and licence
5. **Correct ADR 0004's premise:** the autofocus in use is N.I.N.A's built-in, not Hocus Focus (716 of 716 reports).

