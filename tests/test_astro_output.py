"""scripts/astro_output: the pure parts -- names, the gallery match, the fragment."""
from scripts import astro_output as ao

GALLERY = '''
        <article id="ngc2146" class="dso-section" data-added="2026-10-18" data-distance-ly="70000000">
            <div class="dso-image-container">
                <img src="images/ngc2146.jpg?v=2026-10-18" alt="NGC 2146" class="dso-image">
            </div>
            <div class="dso-content">
                <h2 class="dso-title">NGC2146 - The Dusty Hand</h2>
                <p class="science-text">NGC 2146 is a spiral galaxy in Camelopardalis.</p>
            </div>
        </article>

        <article id="stephans-quintet" class="dso-section" data-added="2026-10-18" data-distance-ly="300000000">
            <div class="dso-image-container">
                <img src="images/stephans-quintet.jpg?v=2026-10-18" alt="Quintet" class="dso-image">
            </div>
            <div class="dso-content">
                <h2 class="dso-title">HCG92 - Stephan&#39;s Quintet</h2>
                <p class="science-text">The fifth galaxy is NGC 7320, a foreground galaxy.</p>
            </div>
        </article>

    </div>

    <div class="container" id="notes-panel">
'''


def test_image_name_gives_target_and_recipe():
    assert ao.parse_image_name("ngc2146_LRGB_learned_sat3_crop.png") == ("ngc2146", "LRGB")
    assert ao.parse_image_name("ic1396_HSO_deblurred.jpg") == ("ic1396", "HSO")
    assert ao.parse_image_name("sh2-92_final.png") == ("sh2-92", None)
    assert ao.processing_from_name("ngc2146_LRGB_learned_sat3_crop.png") == [
        "denoised", "deconvolved (learned, luminance only)", "colour saturation x3", "cropped"]


def test_display_names():
    assert ao.display_name("ngc2146") == "NGC 2146"
    assert ao.display_name("m33") == "M 33"
    assert ao.display_name("sh2-92") == "Sh2-92"
    assert ao.display_name("trunk") == "trunk"


def test_gallery_match_by_id_title_or_text():
    arts = ao.gallery_articles(GALLERY)
    assert [a["id"] for a in arts] == ["ngc2146", "stephans-quintet"]
    a, how = ao.site_match("ngc2146", [], arts)
    assert (a["id"], how) == ("ngc2146", "id")
    # NGC 7320 has no entry of its own: found through the Quintet's title (HCG 92a) and text
    a, how = ao.site_match("ngc7320", ["HCG  92a", "UGC 12101"], arts)
    assert (a["id"], how) == ("stephans-quintet", "title")
    a, how = ao.site_match("ngc7320", [], arts)
    assert (a["id"], how) == ("stephans-quintet", "text")
    assert ao.site_match("ngc1961", ["LEDA 17625"], arts) is None


def test_windows_cache_paths_map_onto_the_local_mirror(tmp_path):
    dso_dir = tmp_path / "ngc2146"
    frame = dso_dir / "cdk17" / "2026-10-05" / "LIGHT" / "x.fits"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"")
    win = r"C:\Users\iriso\Documents\N.I.N.A\Targets\ngc2146\cdk17\2026-10-05\LIGHT\x.fits"
    assert ao.local_frame_path(win, dso_dir) == frame
    assert ao.local_frame_path(win.replace("x.fits", "gone.fits"), dso_dir) is None
    assert ao.local_frame_path(r"C:\elsewhere\m33\LIGHT\x.fits", dso_dir) is None


NOTES = '''
        <article id="learned-deblur" class="note">
            <h2 class="note-title">Learning to undo the blur: a <em>network</em> that moves light</h2>
            <p class="science-text">NGC 1961 and NGC 6946 were the test fields.</p>
        </article>
        <article id="psf" class="note">
            <h2 class="note-title">What shape is a star?</h2>
            <p class="science-text">M 13 only.</p>
        </article>
'''


def test_science_notes_that_name_the_target_are_listed_and_linkable():
    assert ao.site_notes(NOTES, "ngc1961") == [
        {"id": "learned-deblur", "title": "Learning to undo the blur: a network that moves light"}]
    assert ao.site_notes(NOTES, "ngc2146") == []
    assert ao.md_links("see [the note](#learned-deblur) &amp; more") == \
        'see <a href="#learned-deblur">the note</a> &amp; more'
    assert ao.processing_from_name("ngc1961_LRGB_deblurred_sat2.5_fullres_crop.jpg")[:2] == [
        "denoised", "deconvolved (starless Richardson-Lucy)"]


def test_short_aliases_never_match():
    arts = ao.gallery_articles(GALLERY)
    assert ao.site_match("m1", ["M 1"], arts) is None     # "m1" is inside many strings


def _subs():
    return {"L": [{"exptime": 300.0, "total": 60, "used": 56, "rejected": 4, "hours": 4.67,
                   "nights": ["2026-10-05", "2026-10-06"], "gate": "x", "rejected_files": ["a.fits"]}],
            "R": [{"exptime": 300.0, "total": 32, "used": 32, "rejected": 0, "hours": 2.67,
                   "nights": ["2026-10-07"], "gate": "x", "rejected_files": []}]}


def test_render_record_beats_the_replay():
    from datetime import datetime
    from pathlib import Path
    frames = [{"path": Path(f"/x/LIGHT/{n}.fits"), "filter": "L", "exptime": 300.0,
               "time": datetime(2026, 10, 5, 23), "night": "2026-10-05"} for n in ("a", "b", "c")]
    frames.append({"path": Path("/x/LIGHT/r.fits"), "filter": "R", "exptime": 300.0,
                   "time": datetime(2026, 10, 7, 1), "night": "2026-10-06"})
    record = {"L": {"frames": 3, "used": ["a.fits", "c.fits"], "rejected": ["b.fits"], "built": "2026-10-22T07:00"},
              "R": {"frames": 1, "used": ["r.fits"], "rejected": []}}
    subs = ao.subs_from_record(record, frames, ["L", "R"])
    assert subs["L"][0]["used"] == 2 and subs["L"][0]["total"] == 3
    assert subs["L"][0]["rejected_files"] == ["b.fits"]
    assert subs["R"][0]["nights"] == ["2026-10-06"]
    assert "render's own record" in ao.subs_text("m33", "LRGB", subs)


def test_subs_summary_and_text():
    s = ao.subs_summary(_subs())
    assert s["line"] == "56 L + 32 R &times; 300 s"
    assert s["nights"] == ["2026-10-05", "2026-10-06", "2026-10-07"]
    assert s["hours"] == 7.3
    txt = ao.subs_text("ngc2146", "LRGB", _subs())
    assert "L         300 s   used   56 of   60" in txt
    assert "over three nights" in txt
    assert "a.fits" in txt


def test_fragment_is_the_site_markup_with_measured_numbers():
    facts = {"dso": "ngc2146", "name": "NGC 2146", "processing": ["denoised", "deconvolved (learned, luminance only)"],
             "meta_frames": "56 L + 32 R &times; 300 s", "hours": 7.3,
             "nights": ["2026-10-05", "2026-10-06", "2026-10-07"]}
    art = {"title": "NGC2146 - The Dusty Hand", "distance_ly": 70000000,
           "distance_text": "~70 MILLION LIGHT YEARS", "type_text": "STARBURST GALAXY",
           "alt": "A tilted galaxy & dust", "paragraphs": ["First.", "Second."],
           "time_travel_title": "On Earth: ~70 Million Years Ago", "time_travel": "Dinosaurs."}
    frag = ao.build_fragment(art, facts, "ngc2146.jpg", "2026-10-22")
    assert '<article id="ngc2146" class="dso-section" data-added="2026-10-22" data-distance-ly="70000000">' in frag
    assert 'src="images/ngc2146.jpg?v=2026-10-22" alt="A tilted galaxy &amp; dust"' in frag
    assert "56 L + 32 R &times; 300 s | 7.3 HOURS OVER THREE NIGHTS | DENOISED + DECONVOLVED" in frag
    assert frag.count('<p class="science-text">') == 2
    html = ao.insert_fragment(GALLERY, frag)
    ids = [a["id"] for a in ao.gallery_articles(html)]
    assert ids == ["ngc2146", "stephans-quintet", "ngc2146"]
    assert html.index('id="notes-panel"') > html.index('data-added="2026-10-22"')
