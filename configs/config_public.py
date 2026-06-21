class PublicConfig():
    _config = dict({

        "globals": {
            "mastodon instance": None,
            "mqtt_client": None

        },
        "version": {
            "date": "2026.6.21.1"
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
            "image_grid":"local/imaging_grid.png"
    },

        "scratch": {
            "directory": "iris_astronomy/scratch",
            "latest_jpg": "latest_annotated.jpg"
        },

        "nina": {
            "image_dir": "C:/Users/iriso/Documents/N.I.N.A/Targets",
            "sequence_output": "C:/Users/iriso/Documents/N.I.N.A/Sequences/full_for_tonight.json",
            "sequence_input": "C:/Users/iriso/Documents/N.I.N.A/Sequences/cdk_full_sequence.json",
            "sequence_input1": "/home/taylor/Documents/srt/nina_gen/nina_sequence_gen.py",
            "arc_sec_per_pixel": 0.26
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
            "scope_view": "./base_images/scope_view.jpg",
            "processed_view": "./base_images/processed.jpg",
            "no_image": "./base_images/no_image.jpg",
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
            # ASTAP plate solver (also used by N.I.N.A) — used to give transit
            # candidates real sky coordinates and Gaia identifications.
            "astap_exe": "C:/Program Files/astap/astap.exe",
        },

        "calibration": {
            "bias_dir": None,
            "dark_dir": None,
            "flat_root": None,
        },

        "convergence": {
            "tail_slope_threshold": 0.20,   # %/frame — abs(slope) below this = converged
            "rmse_done_threshold": 5.0,     # % — final-point RMSE must also be below this
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
            },
        },

        "nn": {
            "models_dir": "local/models",
            "patch_size": 256,
            "pairs_per_epoch": 2000,
            "epochs": 200,
            "batch_size": 8,
            "min_frames_to_train": 20,
            "tile_size": 512,
            "tile_overlap": 64,
        },

    })

    def data(self):
        return self._config


if __name__ == "__main__":
    mt = PublicConfig()
    print(mt.data["mqtt"])
