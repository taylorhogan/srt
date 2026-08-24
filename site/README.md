# Articles for irisscience.org

Content authored here, published elsewhere. The site's source is **not** in this
repo and is not on the Spark — irisscience.org serves "Iris Lab Notes" from a web
host reachable over the tailnet (`/srv/iris-live` is the live panel; see
`scripts/live_push.py`), and the page itself is maintained on another machine.
So these files are staged, not deployed.

Each `*_note.html` is a drop-in fragment for the site's Science section, matching
its existing convention exactly:

    <article id="..." class="note">
      <h2 class="note-title">                 title
      <div class="note-meta"><span class="note-date">  date, object, filters, method, status
      <h3>Abstract</h3><p class="science-text note-abstract">
      <figure><a href="images/..."><img src="images/..."></a><figcaption>
      <p class="science-text">                body
      <p class="science-text caveat">         optional closing caveat

No wrapper and no CSS changes are needed. Images referenced as `images/<name>`
relative to the page, the same way the existing notes do it.

A self-contained preview (site CSS snapshot + inlined image) can be regenerated
for checking a fragment before it is pasted; it is deliberately not committed,
because it duplicates the image as base64 and the CSS drifts with the live theme.
