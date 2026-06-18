# Prompt — M92 H–R diagram report

The prompt used to generate `M92_HR_Report.pdf` (the PDF itself is intentionally
not tracked in git; regenerate it from this spec).

## Request

Write a research report discussing the H–R / colour–magnitude diagrams this
observatory (Iris) produces, using the latest M92 figures.

Cover:
- What the H–R diagram is and how it shows the evolution of a star.
- Why the diagram shows structure (sharp evolutionary sequences) rather than a
  cloud of stars.
- How the shape of the diagram encodes the cluster's age.
- How star clusters form.

Level and style:
- Aimed at a reader with a solid physics background. Do **not** state the
  intended audience in the document.
- Introduce formulas where they sharpen the argument. In particular include:
  - the stellar initial mass function (mass distribution at formation), and
  - how stellar evolution and end-states depend on stellar mass.

Figures:
- Open with `Image13.jpg`, Iris's own colour photograph of M92 (frontispiece).
- Embed the latest four-panel M92 colour–magnitude diagram and describe all four
  panels.

Output:
- A PDF with embedded images, placed in `docs/`.
- Keep this prompt in `docs/`; do not git-track the PDF.
