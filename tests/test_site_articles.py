"""scripts/site_articles: the site's notes and images as data (2026-10-08)."""
from scripts.site_articles import articles

NOTE = '''
<div class="container" id="notes-panel">
  <article id="quintet-redshift" class="note">
    <h2 class="note-title">Five galaxies that only look like neighbors: measuring Stephan&#39;s Quintet with a filter</h2>
    <div class="note-meta"><span class="note-date">2026-10-06</span> &middot; REDSHIFT &middot; STATUS: DETECTED</div>
    <figure>
      <a href="images/quintet_lrgb_annotated.jpg" target="_blank"><img src="images/quintet_lrgb_annotated.jpg" alt="x"></a>
      <figcaption>Stephan's Quintet from our observatory.</figcaption>
    </figure>
    <h3>Abstract</h3>
    <p class="science-text note-abstract">
        Stephan's Quintet looks like five galaxies huddled together, but one of them,
        NGC 7320, is about eight times closer <em>than</em> the other four.
    </p>
    <p class="science-text">Body paragraph that must not be the summary.</p>
  </article>
</div>
'''

IMAGE = '''
<div class="container" id="gallery-panel">
  <!-- M33 -->
  <article id="m33" class="dso-section">
    <div class="dso-image-container">
      <img src="images/m33.jpg?v=2026-10-04" alt="The Triangulum Galaxy" class="dso-image">
    </div>
    <div class="dso-content">
      <h2 class="dso-title">M33 - Triangulum Galaxy</h2>
      <div class="dso-meta">DISTANCE: ~2.7 MILLION LIGHT YEARS | TYPE: SPIRAL GALAXY</div>
      <p class="science-text">The Triangulum Galaxy is the third largest member of the Local Group.</p>
      <p class="science-text">Second paragraph.</p>
    </div>
  </article>
  <article id="ic1318" class="dso-section">
    <div class="dso-image-container"><img src="https://raw.githubusercontent.com/x/y/main/images/ic1318.jpg" class="dso-image"></div>
    <div class="dso-content"><h2 class="dso-title">IC 1318</h2><p class="science-text">Gamma Cygni.</p></div>
  </article>
  <article id="not-an-entry"><p>no class, ignored</p></article>
</div>
'''


def test_note_fields():
    (a,) = articles(NOTE)
    assert a["id"] == "quintet-redshift" and a["kind"] == "note"
    assert a["title"] == "Five galaxies that only look like neighbors: measuring Stephan's Quintet with a filter"
    assert a["date"] == "2026-10-06"
    assert a["summary"].startswith("Stephan's Quintet looks like five galaxies")
    assert "than the other four." in a["summary"] and "Body paragraph" not in a["summary"]
    assert a["image_url"] == "https://irisscience.org/images/quintet_lrgb_annotated.jpg"
    assert a["url"] == "https://irisscience.org/#quintet-redshift"


def test_gallery_fields_and_both_image_url_forms():
    out = articles(IMAGE)
    assert [a["id"] for a in out] == ["m33", "ic1318"]
    m33, ic = out
    assert m33["kind"] == "image" and m33["date"] is None
    assert m33["title"] == "M33 - Triangulum Galaxy"
    assert m33["summary"] == "The Triangulum Galaxy is the third largest member of the Local Group."
    assert m33["image_url"] == "https://irisscience.org/images/m33.jpg?v=2026-10-04"
    assert ic["image_url"] == "https://raw.githubusercontent.com/x/y/main/images/ic1318.jpg"


def test_page_order_and_base_url():
    out = articles(IMAGE + NOTE, base_url="https://example.org/")
    assert [a["id"] for a in out] == ["m33", "ic1318", "quintet-redshift"]
    assert out[-1]["url"] == "https://example.org/#quintet-redshift"


def test_real_site_if_present():
    import os
    p = os.path.join(os.path.expanduser("~"), "Documents", "development", "taylorhogan.github.io", "index.html")
    if not os.path.exists(p):
        return
    out = articles(open(p, encoding="utf-8").read())
    ids = [a["id"] for a in out]
    assert len(ids) == len(set(ids)) and len(out) >= 40
    assert all(a["title"] and a["summary"] for a in out), \
        [a["id"] for a in out if not (a["title"] and a["summary"])]
    assert all(a["date"] for a in out if a["kind"] == "note")
    # a few early notes are text only; everything in the gallery has a picture
    assert all(a["image_url"] for a in out if a["kind"] == "image")
    assert sum(1 for a in out if a["image_url"]) >= len(out) - 5
