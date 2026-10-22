# Gallery entry style

The site is "Iris Lab Notes": a small robotic observatory writing for a
college-educated reader who is not an astronomer. The gallery entries are the
pictures; the science notes are the measurements. An entry is two paragraphs
and an "On Earth" box, in the site's own markup, which the script renders from
`article.json`. The examples it shows are the NGC 2146 and Stephan's Quintet
entries in `index.html`; read them before writing.

## The keys

| key | what goes there |
|---|---|
| `title` | `NGC6946 - Fireworks Galaxy`: catalogue id without a space, a dash, the common name or (if none) the type |
| `distance_ly` | integer light years, used for the gallery's distance sort |
| `distance_text` | upper case, as the meta line prints it: `~25 MILLION LIGHT YEARS` |
| `type_text` | upper case: `FACE-ON INTERMEDIATE SPIRAL GALAXY` |
| `alt` | one sentence of what is visible, for a blind reader, from the actual picture |
| `paragraphs` | exactly two strings, plain prose; `[text](#note-id)` makes a link |
| `time_travel_title` | `On Earth: ~25 Million Years Ago` |
| `time_travel` | one paragraph: what Earth was like when the light left |

The script adds the frames line, hours, nights and "DENOISED + DECONVOLVED"
to the meta line from the measured facts, and escapes the text, so write
plain characters (an ampersand is fine; no HTML entities, no tags).

## The two paragraphs

**First, the object.** What it is and where, why it looks the way it does
in this picture (the colours, the lanes, the knots, the shape), with the
distance and anything well known about it. Point at things the reader can
find in the frame. Use the facts file's `object_type` (the imaging grid's
classifier plus the raw SIMBAD code) as a check on the type.

**Second, the picture.** The filters and the nights from the facts file, in
words ("luminance, red, green and blue from four nights between 15 and 19
October 2026"). What was done to it, from `processing`: denoised, then
deconvolved (starless Richardson-Lucy or the learned network, whichever the
name says), colour saturation raised for display. Link every entry in
`related_notes`. End with the site's standing caveat for sharpened pictures:
"The sharpened image is for looking at; nothing is measured from it."

## Voice

- Plain declarative prose. No exclamation marks, no em-dashes, no hedging
  filler, no "stunning".
- Facts about the object only when well established. Distances are
  approximate by nature; say "~" and round.
- The "On Earth" box names the geological epoch and three or four concrete
  things (animals, continents, climate), the way the existing entries do.

## A finished example (NGC 6946, 2026-10-22)

```json
{
 "title": "NGC6946 - Fireworks Galaxy",
 "distance_ly": 25000000,
 "distance_text": "~25 MILLION LIGHT YEARS",
 "type_text": "FACE-ON INTERMEDIATE SPIRAL GALAXY",
 "alt": "NGC 6946, the Fireworks Galaxy, deconvolved: a face-on spiral filling the frame, its golden core wrapped in loose blue arms that are dotted with pink and magenta knots of star formation and threaded with dark dust, a bright blue-white star cloud near the top of the disc, and a dense field of white and orange foreground stars across the whole picture.",
 "paragraphs": [
  "NGC 6946 sits on the border of Cepheus and Cygnus, low in the plane of our own galaxy, so we see it through a haze of Milky Way dust that dims and reddens it and fills the field with foreground stars. It is one of the closest large spirals outside our Local Group, and we see it almost exactly face on: a small golden core, and loose, open arms spattered with the pink of hydrogen glowing around newborn stars. The galaxy is forming stars at several times the Milky Way's rate, and the consequence gives it its name. Ten supernovae have been seen in it since 1917, more than in any other galaxy, which is why it is called the Fireworks Galaxy. The bright blue-white cloud near the top of the disc is one of the largest star-forming complexes known in any nearby galaxy, a cluster of young clusters a few million years old.",
  "The picture is luminance, red, green and blue from four nights between 15 and 19 October 2026. The stack was denoised and then deconvolved: the blur the atmosphere and the optics put on every star and every knot was measured from the field's own stars and taken out, with the stars lifted off first so that the deconvolution could not draw rings around them, then put back as imaged. Colour saturation is raised for display. NGC 6946 was held out of the training entirely and used as a test field for the network that now does this job, described in [Learning to undo the blur: a network that moves light](#learned-deblur). The sharpened image is for looking at; nothing is measured from it."
 ],
 "time_travel_title": "On Earth: ~25 Million Years Ago",
 "time_travel": "This light left NGC 6946 in the late Oligocene. The world was cooling from the greenhouse of earlier epochs, Antarctica had its ice sheet, and grasslands were spreading across the continents for the first time, remaking the animals that lived on them. The ancestors of horses and elephants were taking shape, the first apes had appeared in Africa, and the Alps and the Himalayas were still rising."
}
```
