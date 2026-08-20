class PublicConfig():
    _config = dict({

        "globals": {
            "mastodon instance": None,
            "mqtt_client": None

        },
        "version": {
            "date": "2026.8.20.1"
        },

        "logger": {
            "logging": ""
        },
        "location": {
            "city": "Boston",
            "latitude": 41.8096,
            "longitude": -72.8305,
            "elevation": 100,
            "observatory_name": "Iris",
            "timezone": "America/New_York",
            "instructions":"local/my_instructions.json",
            "image_grid":"local/imaging_grid.png",
            # Written by the imaging grid, read by the status ticker so both
            # always name the same DSO. See _publish_tonight_target.
            "tonight_target":"local/tonight_target.json"
    },

        "scratch": {
            "directory": "iris_astronomy/scratch",
            "latest_jpg": "latest_annotated.jpg"
        },

        # Kasa KC420WS all-sky camera, read locally over the 19443 stream --
        # see scripts/probe_kasa_camera.py for why there is no RTSP or ONVIF.
        # That probe's verdict drives the choice of metric here: the camera has
        # no manual exposure, so auto-exposure renormalises every frame and
        # absolute brightness does NOT track the sky. A star count is
        # scale-invariant, so that is what sky_monitor records.
        "sky camera": {
            # DHCP-reserved to MAC BC-07-1D-29-97-0D. The reservation is what
            # makes this line safe to hardcode: on 2026-08-13 the camera
            # rebooted, DHCP moved it off 192.168.87.70, and the feed went dark
            # for half an hour reporting "camera unavailable" -- which reads as
            # a dead camera, not a changed address. If that happens again do
            # NOT hunt for it by grabbing frames off every camera on the LAN;
            # most of them are indoor. Run
            # hardware_control.kasa_utils.make_discovery_map() and match on
            # MODEL: the sky camera is the only KC420WS, the rest are KC410S.
            "host": "192.168.87.52",
            # Rolling archive of raw frames. Kept because the cloud/rain/snow
            # detector that comes next needs labelled examples, and they can
            # only be collected before the fact.
            "capture_dir": "local/sky_frames",
            # ~30 days at the 5-minute cadence (288 frames/day, ~590 KB each,
            # so about 5 GB against 628 GB free). Deliberately generous: the
            # auto-labelling below is a gridded weather model, and this site is
            # on a hilltop where the model demonstrably misses local rain. A
            # missed label means a genuine storm is never preserved and the
            # pruner deletes it, so retention has to be long enough that a human
            # can notice and rescue it. A month of slack is worth more than the
            # disk it costs.
            "keep_frames": 8640,
            # Frames that caught weather move to <capture_dir>/keep/ and are
            # exempt from the cap above -- see sentry/sky_archive.py for why a
            # flat cap is the wrong policy for a rare event. This is keep/'s own
            # ceiling, so a long wet spell cannot fill the disk unattended.
            # ~2.4 GB at 590 KB a frame.
            "keep_event_frames": 4000,

            # --- event-triggered burst ------------------------------------
            # Three stills five minutes apart cannot describe a fifteen-minute
            # shower. The camera delivers 15.1 fps, so when something is
            # happening we pull consecutive frames instead of waiting. Stored
            # as the raw H.264 elementary stream: ~1.3 MB for 8 seconds against
            # ~71 MB for the same frames as JPEGs.
            "burst_dir": "local/sky_bursts",
            "burst_seconds": 8.0,
            "keep_bursts": 2000,            # ~2 GB; storage is not the constraint
            # Night only, on the purity collapse. Proven: the 2026-08-07 rain
            # frame came in at 0.45 purity against 0.99 on a clear one, so this
            # fires on rain without needing the weather model to agree -- which
            # matters at a site whose weather departs from the grid.
            #
            # There is deliberately NO daylight trigger. Daytime rain is not
            # wanted: the roof is shut, so it carries no safety information, and
            # the camera's gain and exposure behave so differently in daylight
            # that daytime rain images would not transfer to a night-time
            # detector anyway. Dropping it also removes the only trigger that
            # had no evidence behind it.
            "burst_purity_below": 0.85,
            # Credentials do NOT go here. They belong in config_private.py
            # under the separate top-level "sky camera auth" key, because
            # config.data() merges the two configs with a shallow
            # dict.update(): a "sky camera" key in the private config would
            # replace this whole section, silently taking the host and every
            # threshold below with it. KASA_USERNAME / KASA_PASSWORD are
            # honoured as a fallback so an interactive run needs no edit.

            # --- star detector -------------------------------------------
            # ADU above the local background. 12 was not guessed: it was set by
            # running the detector on the negated residual, where every hit is
            # by construction a false positive. On the 2026-08-08 02:32 frame
            # that gave 185 real / 1 false (99.5% pure); 8 ADU gave 95.5% and
            # 15 ADU gave 100% at the cost of a third of the stars.
            "star_threshold_adu": 12.0,
            "star_fwhm_px": (2.0, 9.0),
            "star_max_elongation": 2.2,
            # Foliage is masked on LEVEL, not texture. At night exposure the
            # trees are too dim to be rough -- high-pass RMS is 2.6-3.4 over
            # foliage against 1.5-2.5 over sky, which does not separate -- but
            # they sit 10-22 ADU above a ~2 ADU sky. So the skyglow is fitted
            # with a robust 2-D cubic and anything this far above the fit is
            # masked. A texture mask tried first masked 57% of the frame,
            # including the three brightest stars.
            "foliage_resid_adu": 2.5,

            # --- rain prediction / detection (sentry/rain_detect.py) --------
            # Fraction of central-sky pixels changing between ADJACENT video
            # frames. Measured on 7.5 days of the camera's own SD card:
            # clear night 0.1-1.1%, rain 16-100%, daylight rain 3-7%.
            #
            # Two thresholds because onset is a ramp: 1% -> 10% took 26 and 38
            # minutes on the two storms measured, which is what makes a
            # prediction possible. Escalation past 10% is NOT gradual (10->30%
            # took 26 min on one storm and 2 min on the other), so "detect"
            # means already raining hard.
            #
            # 1% overlaps the clear-night range, so a single sample at 1% is
            # worthless -- rain_persist consecutive samples is what makes the
            # low threshold usable. At the 5-minute cadence 3 samples is 10
            # minutes of agreement.
            "rain_predict_pct": 1.0,
            "rain_detect_pct": 10.0,
            "rain_persist": 3,
            # One prediction and one detection per 6 h. Deliberately separate
            # limiters: a shared one would suppress the detection that confirms
            # the prediction, which is the pair worth seeing. The separation is
            # ASYMMETRIC -- a detection also stamps the prediction limiter,
            # because heavy rain satisfies the predict condition too and the
            # next sample would otherwise downgrade a storm in progress to
            # "rain likely soon" (observed 2026-08-13 00:45, five minutes after
            # a 99.3% detection).
            "rain_alert_gap_s": 21600,
            # How long the sky may go unmeasured without breaking a run. A
            # dropout is missing evidence, not evidence of a dry sky, but a long
            # outage means the samples either side of it are not consecutive and
            # must not be counted as a ramp.
            #
            # This governs cadence jitter as well as dropouts, and the jitter is
            # what sets the floor: across 281 night intervals the nominal 5 min
            # spacing is 5.3 min at p95 but reaches 19.4 min at the tail, and
            # one of those long intervals falls INSIDE the 2026-08-13 storm
            # (00:25 -> 00:40). A 900 s value was tried first and pushed that
            # storm's detection 10 minutes late. 1200 s clears every observed
            # interval and still tolerates three consecutive camera failures.
            "rain_gap_tolerance_s": 1200,
            "rain_diff_adu": 12.0,
            "rain_burst_seconds": 2.0,
            "rain_min_frames": 4,
        },

        # The ZWO ASI120MC-S fisheye on iris-pc -- the "all-sky camera". A
        # separate instrument from the "sky camera" above, which is the Kasa
        # KC420WS bolted outside, and it must not inherit any of that camera's
        # numbers: different sensor, different lens, different bit depth, and
        # every threshold up there was measured on 8-bit auto-exposed JPEGs.
        #
        # It lives UNDER THE ROOF and sees sky only while the roof is open,
        # which is why its published panel has to be gated rather than always
        # present.
        "allsky camera": {
            # Shipped with N.I.N.A rather than installed separately, so this is
            # the copy that is known to match the driver actually talking to
            # the camera.
            "sdk": r"C:\Program Files\N.I.N.A. - Nighttime Imaging 'N' Astronomy"
                   r"\External\x64\ASI\ASICamera2.dll",
            # Substring, not an index. The ASI432MM enumerates first, so index 0
            # is the wrong camera.
            "camera_match": "120",

            "capture_dir": "local/allsky_frames",
            "dark_dir": "local/allsky_darks",
            "settings_file": "local/allsky_settings.json",
            "solution_file": "local/allsky_solution.json",
            "keep_frames": 2000,

            # Nothing is burnt into these frames -- that is a Kasa habit. Left
            # unset this would inherit that camera's timestamp and watermark
            # bands and blank 130 rows of real sky.
            "overlay_boxes": [],

            # --- star detector -------------------------------------------
            # PLACEHOLDERS on a 16-bit scale, pending a night frame. Set them
            # with `python sentry/star_count.py <frame.fits> --profile
            # "allsky camera" --sweep`, which prints the negative-image control
            # beside each threshold so the choice is measured rather than
            # guessed -- the same way the Kasa's 12 ADU was arrived at.
            # This camera's GAIN is a free parameter the auto-exposure sweep
            # searches, so its cut cannot be a fixed number of ADU: gain scales
            # star, sky and read noise together, a fixed cut means something
            # different at every gain, and "most detections" then climbs the
            # gain axis whether or not anything was seen better. 5 sigma is
            # where the negative-image control put it when measured at gain 100
            # (5594 ADU against 1119 ADU of noise, 1896 stars at 99.6% purity),
            # so this is the same decision expressed in units that travel.
            "star_threshold_sigma": 5.0,
            # Superseded by the sigma cut above and kept only as the record of
            # that measurement. Ignored while star_threshold_sigma is set.
            "star_threshold_adu": 5594.0,
            # Zero on purpose. foliage_mask floors its cut at 3x the skyglow
            # fit's own sigma, so leaving this at 0 lets that floor govern and
            # the mask scales with gain like everything else. The 1400 that was
            # here is ~22 sigma at gain 0 and would have masked nothing at all.
            "foliage_resid_adu": 0.0,
            # PROVISIONAL. Completeness against radius, measured 2026-08-15, was
            # flat at 22-32% out to 600 px and softened past it, with no cliff --
            # unlike the Kasa, which read 23-40% to 800 px and then exactly zero.
            # 600 is where this one starts to degrade, not where it stops
            # delivering.
            #
            # Do not trust the number: it was measured through an 8s/gain-100
            # frame, i.e. the overexposed setting sweep_gains below was still
            # choosing, so the sky sat at 24% of full scale and buried the stars.
            # Cloud was blamed at the time and that was WRONG -- the Kasa was
            # reading limiting magnitude 5.28 on a black sky the same minute. The
            # centre bin (0-100 px) came in at 32%, which radius cannot explain
            # and overexposure can. Re-measure at gain 0, where the centre should
            # approach 100% and a real edge should appear.
            "measured_radius_px": 600.0,
            # Wider at the low end than the Kasa's 2.0. This sensor is small and
            # the lens is short, so a well-focused star lands in very few
            # pixels; the Kasa's floor would reject the sharpest stars in the
            # frame. See star_min_pixels for what that costs.
            "star_fwhm_px": (1.2, 8.0),
            # Below the default 6, which is what keeps single-pixel hot pixels
            # out of the Kasa's count. This sensor runs uncooled, so the master
            # dark (sentry/asi_allsky.py --dark) is what removes them here
            # instead. Without a dark on disk, capture() says so loudly.
            "star_min_pixels": 4,
            "star_max_elongation": 2.2,

            # --- auto-exposure sweep --------------------------------------
            # The objective is the star count itself: each rung is measured with
            # the detector that will publish the number, so the setting that
            # wins is the one that actually shows the most sky. Exposure is
            # capped at 30s because that is where trailing starts to matter --
            # at roughly 7 arcmin/px a star crosses a pixel in about 28s.
            "sweep_exposures_s": (1.0, 2.0, 4.0, 8.0, 15.0, 30.0),
            # 100 is this camera's MAXIMUM, and the old (50, 100) could not find
            # anything better because it never looked lower. Measured 2026-08-15
            # with a per-frame 5-sigma cut: gain 0/30s gave 2077 stars over a 976
            # ADU background against gain 100/30s giving 1308 over 19587 -- 20x the
            # sky for 37% fewer stars, worse purity (0.930 vs 0.959) and fatter
            # stars. Max gain also shrinks full well, which is why the published
            # frames looked washed out.
            "sweep_gains": (0, 25, 50),
            # Percent of the frame at full scale a rung may carry. Clipping does
            # not just lose the bright stars, it flattens them into the
            # background so they stop being detections at all.
            "clip_limit_pct": 0.5,

            # --- roof gate (scripts/allsky_monitor.py) ---------------------
            # Nothing from this camera is published as sky unless the roof is
            # known open, because with it shut the camera photographs the roof's
            # own underside and the detector returns hundreds of "stars" off the
            # texture. The primary evidence is the plate solution landing
            # catalogue stars on detections, which a closed roof cannot fake;
            # the safety camera is the fallback for daylight and cloud, when
            # there are no stars to appeal to.
            "roof_check_vision": True,
            # A vision read costs a Kasa discovery sweep plus an exposure ladder
            # on the indoor webcam, so verdicts are cached -- except "open",
            # which is always taken fresh. See _vision_roof for why.
            "roof_vision_cache_s": 900,

            # measured_radius_px is deliberately absent until it has been
            # measured on this lens. The Kasa's 800 px describes a different
            # optic entirely, and a borrowed radius would make the completeness
            # table a statement about the wrong camera.
        },

        "nina": {
            "image_dir": "C:/Users/iriso/Documents/N.I.N.A/Targets",
            "sequence_output": "C:/Users/iriso/Documents/N.I.N.A/Sequences/full_for_tonight.json",
            "sequence_input": "C:/Users/iriso/Documents/N.I.N.A/Sequences/cdk_full_sequence.json",
            "sequence_input1": "/home/taylor/Documents/srt/nina_gen/nina_sequence_gen.py",
            "arc_sec_per_pixel": 0.26,
            # Filter wheel: name -> wheel position. Taken from the flats
            # sequence, which is the one place these are known to be right --
            # it produces correctly labelled flats for every filter.
            #
            # BOTH are written into a generated sequence. Setting only the name
            # leaves the position pointing at whatever the template had, which
            # is how a sequence silently images the wrong filter: the template's
            # narrowband blocks sit at positions 1/2/3, which are R/G/B.
            "filter_wheel": {
                "L": 0, "R": 1, "G": 2, "B": 3,
                "S-II": 4, "O-III": 5, "Ha": 6,
            }
    },

        "camera safety": {
            "parked azimuth deg": 57.2,
            "parked altitude deg": 1.0,
            "closed template": "./base_images/closed_marker.jpg",
            "parked template": "./base_images/parked_marker.png",
            "open template": "./base_images/open_marker.jpg",
            "open pos": (172, 142),
            "closed pos": (829, 152),
            "parked pos": (590, 290),

            "accuracy": 150,
            # Minimum normalized template-match score (TM_CCOEFF_NORMED, 0..1)
            # required before a parked/closed/open state is trusted. Guards
            # against a spurious low-confidence match that happens to land near
            # the expected pixel position.
            "match_confidence": 0.6,
            # Daylight exposure-ladder capture. visual_status already sweeps ten
            # exposures every call and throws away nine; when the scene is bright
            # enough to be daylight, keep the whole ladder plus each frame's
            # per-template match confidence. This is the dataset needed to fix
            # daytime roof detection — the exposure scorer grades the WHOLE frame,
            # so a blown-out sky patch drags it to an exposure that leaves the
            # markers dark, and the roof cannot be confirmed open in daylight.
            # Costs no extra camera time: the frames are taken either way.
            # Score the exposure sweep by marker readability rather than
            # whole-frame brightness (vision_safety.marker_match_score). The
            # whole-frame scorer picked an exposure where the open marker was
            # 565 px out of place while a frame existed in the same ladder with
            # it 30 px out — see docs/daylight_roof_detection.md. Set False to
            # fall back to best_exposure_score.
            "marker_exposure_scorer": True,
            "exposure_capture": True,
            # Daylight test: the sun's altitude, not a brightness heuristic. A
            # luma threshold has to be guessed and the ladder spans ~500x in
            # shutter time, so any fixed cut is fragile; solar altitude is exact
            # and free. -6 deg is civil twilight, which covers the whole window
            # where opening earlier is worth doing.
            "exposure_capture_min_sun_alt": -6.0,
            "exposure_capture_dir": "./base_images/exposure_sets",
            "exposure_capture_keep": 30,      # rolling cap on saved ladders
            "scope_view": "./base_images/scope_view.jpg",
            "processed_view": "./base_images/processed.jpg",
            "no_image": "./base_images/no_image.jpg",
            # Output of the `live` command: a no-light, exposure-optimized view of
            # the sky from the scope-top webcam. Kept separate from scope_view so a
            # dark long-exposure sky frame never overwrites the lit safety snapshot.
            # `live` takes TWO passes: a low-gain, long-exposure pass that records
            # stars (sky_view_stars) and a high-gain pass that favours diffuse
            # skyglow/clouds (sky_view). At max gain the longest sub blows out, so
            # the exposure scorer falls to a ~15 ms frame with no stars — hence the
            # separate low-gain star pass. Gains are tunable here; validate against
            # a real night sky (watch the per-exposure clip% now logged in the sweep).
            "sky_view": "./base_images/sky_view.jpg",
            "sky_view_stars": "./base_images/sky_view_stars.jpg",
            "sky_stars_gain": 30,
            "sky_skyglow_gain": 100,
            "valid_data": False,
            "received_count": 0

        },
        "Calendar":
            {
                "image": "lightblue",
                "weather": "pink",
                "service": "orange"
            },
        "Globals":
            {
                "Observatory State": "In Development",
                "Imaging Tonight": "Unknown"
            },

        "web_chat": {
            "enabled": True,
            "port": 8095,
            "host": "0.0.0.0",
            "mastodon_mirror": False,
            "max_history": 500,
            "upload_dir": "saved_dso",
            # Where remote machines (the Spark) reach this chat's /api/post
            # over the Tailnet. Same-machine callers fall back to localhost.
            "remote_url": "http://100.95.7.19:8095",
        },

        "sync": {
            "rsync_path": "C:/Users/iriso/Documents/cwrsync_6.4.7_x64_free/bin/rsync",
            "source": "iriso@100.95.7.19:/cygdrive/c/Users/iriso/Documents/N.I.N.A/Targets",
            "destination": "~/Desktop"
        },

        "pegasus": {
            "unity_url": "http://localhost:32000",
            "driver_key": "",  # DriverUniqueKey from Unity; blank = auto-discover
        },

        "hardware": {
            # Shelly relay that toggles the roof motor (GETting this URL fires
            # the relay; the roof direction depends on its current position).
            "roof_relay_url": "http://192.168.87.41/relay/0?turn=on",
            # Shelly relay base URL for the dehumidifier; append ?turn=on / ?turn=off.
            "dehumidifier_relay_url": "http://192.168.87.28/relay/0",
            # Shelly 3EM Gen3 (S3EM, EM1 mode) used as a 120V current monitor.
            # Unlike the Gen1 relays above this speaks the Gen2/3 RPC API:
            # GET {base}/rpc/EM1.GetStatus?id=<channel> -> {voltage, current, ...}.
            "current_monitor_url": "http://192.168.87.46",
            "current_monitor_channel": 0,
            # ASTAP plate solver (also used by N.I.N.A) — used to give transit
            # candidates real sky coordinates and Gaia identifications.
            "astap_exe": "C:/Program Files/astap/astap.exe",
        },

        "calibration": {
            # Shot 2026-07-25 at the imaging settings: -10 C, gain 0, offset 10,
            # bin 1. The dark set holds 19x300s (plus one stray 60s that
            # load_calibration_set drops); darks are scaled by exposure ratio for
            # lights shot at other lengths.
            "bias_dir": "C:/Users/iriso/Documents/N.I.N.A/Targets/cdk17/2026-07-25/BIAS",
            "dark_dir": "C:/Users/iriso/Documents/N.I.N.A/Targets/cdk17/2026-07-25/DARK",
            # Flats live per session as cdk17/<date>/FLAT with every filter mixed
            # together, so this root is scanned recursively and grouped by the
            # FITS FILTER header (stacker.flats_by_filter) rather than by the
            # root/<FILTER>/ directory convention. Masters agreed to 0.2-0.3% RMS
            # between 07-14 and 07-31, so combining nights is safe.
            "flat_root": "C:/Users/iriso/Documents/N.I.N.A/Targets/cdk17",
        },

        # Percentages of SKY SIGNAL, and only meaningful with calibration
        # configured above. Uncalibrated they were percentages of the bias
        # pedestal, ~18x smaller for the same data, which is why these were once
        # 0.20 / 5.0. Entries in convergence.json without "calibrated": true are
        # on that old scale and is_dso_done ignores them.
        #
        # Derived 2026-08-01 on sh2-92, after subset stacks moved to the same
        # sigma-clip combine as the golden (which removed a 16-21%-of-sky floor):
        #   Ha    118 frames  slope -0.2645  RMSE  7.08%
        #   O-III 129 frames  slope -0.1415  RMSE 16.15%
        #   Ha     20 frames  slope -2.4129  RMSE 12.32%   (deliberately short)
        # 0.40 is not picked by eye: it is the value at which
        # frames_needed_estimate, run on the 20-frame subset, predicts 127 frames
        # to converge against the 137 actually shot (-8%). 0.30 predicts 169,
        # 0.50 predicts 101.
        #
        # rmse_done_threshold is NOT a convergence gate — look at the third row.
        # A 20-frame set scores 12.32% while fully-converged O-III scores 16.15%,
        # because O-III's residual is dominated by a correlated term (it falls
        # 2.4x slower than independent noise; see convergence.decay_ratio) that
        # no number of frames removes. Once the method floor was gone, RMSE
        # stopped measuring "enough frames" and started measuring "how clean is
        # this channel". The slope is the gate; this is a sanity backstop for a
        # channel far worse than anything seen so far.
        "convergence": {
            "tail_slope_threshold": 0.40,   # %/frame — abs(slope) below this = converged
            "rmse_done_threshold": 25.0,    # % — data-quality backstop, not a convergence test
            "min_frames_per_filter": 16,    # don't evaluate until this many frames
            "file": "local/convergence.json",
        },

        "transit": {
            "min_frames": 20,
            "aperture_fwhm_mult": 1.5,
            "sky_annulus_fwhm_mult": (3.0, 5.0),
            "comparison_quantile": 0.25,
            "min_baseline_days_for_bls": 1.0,
            # Report only the single best candidate (plot, field image, JSON).
            "top_n_plot": 1,
            # Skip the slow per-frame FWHM Gaussian fitting and just register
            # (astroalign). The search does its own clean-baseline weighting, so
            # FWHM weighting isn't needed — this is ~30 s vs ~15 min. Set False to
            # restore FWHM-weighted registration.
            "skip_fwhm_registration": True,
            # Permutation trials for the top candidate's false-alarm probability.
            "significance_permutations": 500,
            # Worker processes for the CPU-bound phases (per-star box+BLS search
            # and the permutation significance test). Those are pure-Python loops
            # and GIL-bound, so threads don't parallelise them — separate
            # processes use real cores. 0 = os.cpu_count(). NOTE: running several
            # searches at once multiplies this; lower it if you routinely run 2+
            # concurrently and want to avoid CPU oversubscription.
            "max_workers": 0,
            # Reject stars whose peak pixel reaches this many ADU in any frame
            # (7x7 window around the centroid). A star clipped at the full well
            # produces fake dips that preferentially top the score table —
            # validated on m92 where a G=9.8 star scored field_z=23 and pushed
            # a false alert. 0 disables the veto.
            "saturation_adu": 55000,
            # Variable-star search (Lomb-Scargle) over the same kept stars:
            # how many top variables to report, and the shortest period
            # searched (0.05 d ~ 1.2 h, below any RR Lyrae, catches SX Phe).
            # The longest period searched is baseline/2.
            "variables_top_n": 10,
            "ls_min_period_d": 0.05,
            # Match radius (arcsec) for labelling variables/candidates against
            # the AAVSO Variable Star Index (VSX). Within this of a catalogued
            # variable → known (name/type/period reported); otherwise flagged
            # as not-in-VSX (a candidate new variable). 5" ~ 19 px at 0.26"/px.
            "vsx_match_radius_arcsec": 5.0,
            # Robust field-outlier z-score above which the top candidate is
            # considered a strong single-night detection. When the top
            # candidate's field_z meets/exceeds this, a push notification is
            # sent. Still a candidate, not a confirmation (needs a 2nd transit).
            "field_z_alert": 6.0,
            # Reject stars whose light curve has a finite flux in fewer than this
            # fraction of frames. Edge stars drift into registration NaN borders
            # on some frames, leaving mostly-NaN curves that read as huge fake dips.
            "min_valid_fraction": 0.8,
            # Reject stars whose centre is within this multiple of the sky-annulus
            # outer radius of any frame edge, so the full aperture stays on-chip.
            "edge_margin_mult": 1.0,
            # Blank differential-flux samples more than this many robust sigma
            # ABOVE the baseline before searching. Stars never brighten during a
            # transit, so high spikes are photometry glitches; low excursions
            # (real dips) are kept.
            "outlier_high_sigma": 5.0,
            # Reject BLS detections whose fractional depth is outside (0, this).
            # Flux is normalised to ~1, so depth ≥ 1 means negative in-transit
            # flux — an outlier artifact, not a real transit/eclipse.
            "max_bls_depth": 0.8,
            # Divide the per-frame median (common mode) over kept stars out of
            # every light curve before searching. A flux change shared by many
            # stars (start-of-night ramp, between-session offset, transparency)
            # is never a transit — that affects one star.
            "common_mode_correction": True,
            # Down-rank any BLS period shared by at least this fraction of the
            # stars that produced a BLS detection — sparse/irregular sampling
            # aliases cluster unrelated stars at identical periods.
            "shared_period_frac": 0.05,
            # BLS significance floor (rejects sparse-sampling artifacts where a
            # near-flat periodogram inflates the power z-score):
            "min_bls_cycles": 2,        # baseline must span ≥ this many periods
            "depth_snr_min": 3.0,       # depth ≥ this × (scatter / sqrt(n_in_transit))
            "min_transit_points": 3,    # ≥ this many in-transit points required
            "min_bls_power": 0.01,      # absolute-power backstop
            # Reject when far more points pile into the transit window than
            # duration/period predicts — that's a sampling alias, not a transit.
            "max_in_transit_factor": 2.5,
            # Plate-solve the reference frame (ASTAP) and tag each candidate with
            # RA/Dec + the nearest Gaia source (id, G mag, separation).
            "identify_candidates": True,
            "gaia_match_radius_arcsec": 3.0,
            # --- Single-transit box score (primary ranking statistic) --------- #
            # Each star's light curve is searched for a flat-bottomed dip bracketed
            # by flat baseline on BOTH sides, sitting on a clean baseline. This is
            # what surfaces a shallow planet transit on a bright star over the
            # deeper noise/ramp artifacts a plain "lowest dip" statistic ranks
            # first. score = (depth·√n_in/σ_oot) · flat_bottom · clean_baseline.
            "single_transit_min_dur_h": 0.3,   # shortest trial transit width
            "single_transit_max_dur_h": 4.0,   # longest trial transit width
            "single_transit_n_widths": 16,     # width grid resolution
            "single_transit_min_in_points": 12,    # ≥ this many in-transit points
            "single_transit_min_side_points": 8,   # ≥ this many baseline points each side
            "single_transit_min_side_h": 0.4,      # ≥ this much baseline time each side
            # Clean-baseline weight = min(1, floor/σ_oot): a transit can't be
            # detected on a baseline noisier than its depth, so noisy stars
            # (faint, blended) are down-weighted toward zero.
            "single_transit_baseline_quality_floor": 0.01,
            # Reject dips deeper than this fraction as unphysical (blend / over-
            # subtracted sky); real planet transits are ≲ a few percent.
            "single_transit_max_depth": 0.5,
            # Weight on the BLS periodicity z-score added to the box score (the
            # box dominates; BLS only adds a bonus on multi-night periodic runs).
            "bls_score_weight": 0.5,
            "file": "local/transits.json",
        },

        # Per-machine overrides, keyed by socket.gethostname(). At config load
        # (configs/config.py:data()) the entry for the current host is overlaid
        # onto the shared config, so a key like "image_dir" replaces
        # cfg["nina"]["image_dir"] without any call site changing. Add a new
        # block here for each machine you run on. The top-level "nina"/"machine"
        # defaults below are the fallback for hosts with no entry.
        "machine": {
            "iris-pc": {
                "image_dir": "C:/Users/iriso/Documents/N.I.N.A/Targets",
            },
            "spark-3129": {
                "subs_dir": "/home/taylor/Desktop/Targets",
                "image_dir": "/home/taylor/Desktop/Targets",
                # The rsync from the observatory mirrors cdk17/<date>/ intact,
                # so the same BIAS/DARK/FLAT sets are here under a Linux path.
                # Without this the shared "calibration" block names C: paths,
                # calibration_paths_from_config finds nothing, and every stack
                # taken on the Spark is silently uncalibrated — which is what
                # happened to all the N2N stacks up to 2026-08-15.
                "bias_dir": "/home/taylor/Desktop/Targets/cdk17/2026-07-25/BIAS",
                "dark_dir": "/home/taylor/Desktop/Targets/cdk17/2026-07-25/DARK",
                "flat_root": "/home/taylor/Desktop/Targets/cdk17",
            },
        },

        "nn": {
            "models_dir": "local/models",
            "patch_size": 256,
            "pairs_per_epoch": 2000,
            "epochs": 60,
            "batch_size": 8,
            "min_frames_to_train": 20,
            "val_dsos": 2,
            "tile_size": 512,
            "tile_overlap": 64,
        },

    })

    def data(self):
        return self._config


if __name__ == "__main__":
    mt = PublicConfig()
    print(mt.data["mqtt"])
