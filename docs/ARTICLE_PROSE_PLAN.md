# Prose for the gallery entries, written on the Spark (2026-11-05)

Owner: "Where else can we use the GPU" -> item 5, "a local model to write
the publish articles" -> "Start the project for creating the prose for image
articles", and when the first draft of the work reached for a hosted model:
"I thought you would need to build a model. Number 5 in your potential GPU
tasks." So: an open-weight model served on the GB10, not an account.

## The gap

D2 of docs/AUTOMATION_PLAN.md makes every publish draft on the Spark. Its
prose came from `generate_writeup` through the Anthropic API when a key was
present, and the Spark has never had one (SUCCESSION.md, 2026-10-30), so
every draft since has been `fallback_writeup`: two sentences of measured
facts and "a fuller description will follow". The gate's card asks the owner
to approve that. Nobody will, so one-tap publishing has no prose to publish.
The five entries on the site since 2026-10-22 were written by a Claude Code
session by hand through the `/publish` skill, which is the standard.

## What it is

`scripts/article_prose.py`, called by `astro_output.handle_image` for every
new draft (`generate_writeup` is now a thin wrapper), with three backends in
`cfg["prose"]["writer"]` order ("auto": local, api, cli; or one of them; or
"off"):

- **local**: a vision model served by Ollama on this machine, the owner's
  user-level install under `~/.local/ollama` (no sudo, no system service;
  the writer starts `ollama serve` itself when the port is silent and the
  model unloads 15 minutes after the last call, so the GPU is the morning
  render's again). Nothing in SUCCESSION.md can take it away. Provisional
  model `gemma4:31b-it-q8_0` (34 GB, Google's Gemma terms of use, image input); the bench
  below decides between it and `qwen3-vl:32b-instruct-q8_0` (36 GB).
- **api**: the Anthropic SDK with `cfg["anthropic"]["api_key"]`, the
  original D2 design, kept for when the owner wants Opus.
- **cli**: `claude -p` with the owner's Claude Code login on the Spark; it
  works today (tested 2026-11-05: reads the picture through its Read tool),
  costs no setup, and dies with the account. A convenience, not the design.

The same prompt and the same loop for all three:

1. **The reference block** (`dossier`): the Wikipedia introduction for the
   object, by its name then its catalogue aliases (Melotte 15 has no page;
   IC 1805 does), capped at 2,500 characters; the object type; the
   distance on record in `local/dso_distances.json`; the related lab notes.
   It rides `article_request.json` as `reference`. A local model is told
   that everything it says about the object comes from there or is well
   established, and to list what is not in the reference as `claims`.
2. **Write**: the facts, the reference, the picture (1024 px for the local
   model), and the newest gallery entries as the voice (`site_examples`:
   NGC 2146 and the Quintet, plus the four newest by `data-added`). JSON
   out with the eight article fields plus `claims`; Ollama is asked for
   JSON (`format: json`), the others are parsed tolerantly; a non-answer is
   asked again once.
3. **Review**: a second call, text only, lists problems in four classes:
   a measured number contradicted, a statement the reference does not
   support, a style break, a missing note link. One rewrite with the list,
   one more review. What remains is `review_notes` on the article.
   **The reviewer is the other model** (owner, 2026-11-06: "Let's try the 2
   model adversarial approach"): `cfg["prose"]["review_backend"]` (local |
   api | cli | same) and `review_model` (qwen3-vl checks gemma4 by
   default); the reviewer is told the draft is another model's and to
   trust only the facts and the reference. A model checking its own draft
   tends to agree with itself; the M76 trial's "AD 500" was caught because
   arithmetic against a stated fact is easy, an invented detail it believed
   while writing it would believe again. A reviewer that cannot answer (not
   pulled, server down, no key) hands the review back to the writer, the
   log says so, and the article's `reviewer` names who did it. On the
   bench each writer is reviewed by the other model on the bench, so the
   two runs are mirror images and the review notes say how each does as a
   critic.
4. **The card** (`publish_gate.card_text`) shows who wrote it and the
   model, who reviewed it, the claims to check, and the reviewer's notes. The owner's
   `approve` is the last check; the meta line's numbers are written from
   the data, never from the article.

`scripts/article_prose.py write <dso>` rewrites a folder's draft (and with a
queued draft, re-hashes it: a new draft needs a new approve; `--announce`
re-cards it). `bench --models a,b` runs every model on the folders with a
hand-written entry and writes `local/article_prose/bench/bench.md`, the
machine's prose under the hand's for the same target.

## Why not train or fine-tune

ASSISTANT_PLAN.md (2026-09-03) made the same call for the operator's
assistant: a stock model plus retrieval knows the facts by construction and
stays current; a LoRA teaches style and the facts rot. Here the style is six
examples in the prompt and the facts are the reference block. The thing
worth measuring is which stock model writes the better entry from that.

## Acceptance

1. The bench on the six hand-written entries (IC 410, M1, M76, Melotte 15,
   NGC 1961, NGC 6946): the owner reads the machine's entry under the
   hand's. Pass = would approve with at most one edit; the reviewer's note
   count and the claims list are the numbers beside the judgment.
2. The next real draft (the first `converged` target with a QC pass) comes
   to the chat with prose, a claims list and reviewer notes, and the owner
   approves or rejects it on its merits.
3. The model that wins becomes `cfg["prose"]["local_model"]`.

## Owner's part

The download of an executable is the one thing this session may not do.
User-level, no sudo:

    mkdir -p ~/.local/ollama && cd ~/.local/ollama
    curl -fL https://github.com/ollama/ollama/releases/latest/download/ollama-linux-arm64.tar.zst | tar --zstd -x
    # (2026-11-06: the ollama.com/download/...tgz link 404s since the release
    #  moved to .tar.zst; `tar --zstd` needs zstd, which the Spark has)
    ~/.local/ollama/bin/ollama serve &            # or let the writer start it
    ~/.local/ollama/bin/ollama pull gemma4:31b-it-q8_0
    ~/.local/ollama/bin/ollama pull qwen3-vl:32b-instruct-q8_0
    cd ~/Documents/srt && .venv/bin/python scripts/article_prose.py bench \
        --models gemma4:31b-it-q8_0,qwen3-vl:32b-instruct-q8_0

Models live in `~/.ollama/models` (70 GB for the two; 2.2 TB free). Both
are needed for the two-model review: the writer and its critic sit loaded
together (~70 GB of the 121) and unload 15 minutes after the last call.

## Log

- 2026-11-05: built `scripts/article_prose.py` (dossier, prompts, the three
  backends, the write-review-rewrite loop, `write` / `bench` / `dossier`),
  `generate_writeup` delegated, `site_examples` takes the newest four, the
  card shows source, model, claims and reviewer notes, `cfg["prose"]`,
  `anthropic` SDK installed in the Spark's venv. Headless `claude -p`
  verified end to end on the M76 crop (reads the picture, answers JSON).
  The Ollama install and the model pulls are the owner's; the bench runs
  after them.
- 2026-11-05 (late), the loop end to end on a copy of the M76 folder with the
  cli backend (Opus through Claude Code), 95 s: the first draft put the
  light's departure at "AD 500"; the reviewer caught it (2,500 light years
  is 500 BC), the rewrite placed it right and the second review passed.
  Five claims listed (the white dwarf's temperature and the discovery came
  from the Wikipedia block, so not among them). The owner's copy:
  ~/Desktop/m76_article_trial_claude_cli.json. The local model's turn is
  the bench, after the pulls.
- 2026-11-06, **two-model review**: the owner asked what the problem with
  two models would be (memory, two things to keep pulled, a critic's taste
  leaking into the prose, against a self-review that agrees with itself)
  and chose the adversarial pair. `reviewer_for` / `reviewed` in
  article_prose: the review call goes to `review_backend` / `review_model`,
  falls back to the writer when that model cannot answer, and the article
  carries `reviewer`; the card prints "Reviewed by"; `write --reviewer`,
  `bench --reviewer`, and the bench pairs each writer with the other model
  by default. Still waiting on the pulls.
- 2026-11-06 08:40, **the first local drafts** (qwen3-vl q8 alone, reviewing
  itself, on the M76 copy; Ollama 0.40.15 user-level, the model 100 % on the
  GPU, 43 GB resident with the 32k context). Three lessons from the first
  run, each fixed before the second: (1) the writer was shown M76's OWN site
  entry among the voice examples (it was one of the newest four) and
  paraphrased it, cross-references to other entries included; a draft like
  that is no test of anything, so `site_examples(exclude=)` drops the
  target's own id and `_inputs` passes it. (2) The reviewer answered in
  checklist mode: 23 "problems", every one a sentence saying the draft was
  right, and the rewrite was asked to fix them; the review prompt now asks
  for defects only as {class, quote, problem} objects, `problem_lines`
  drops praise, and the reviewer runs at temperature 0.2. (3) The 'On
  Earth' box said AD 500 for 2,500 light years, the same slip Opus made on
  11-05; `light_left_words` puts the year in the prompt when the distance
  is on record and `YEAR_RULE` with the arithmetic is in both prompts.
  Second run: 166 s for write + review, 0 problems, 0 claims (every object
  fact in the Wikipedia block; the frames, nights and processing as
  measured), the year right, no leak. Paragraph 1 reads as an encyclopaedia
  digest (temperature, magnitude, size in one breath) and "Hubble palette"
  is the wrong name for HOO: the owner's call on the bench. Decode is ~4-6
  tokens/s at q8 on the GB10 (memory-bound), ~3 min a write with the
  picture, which is fine for a weekly job.
- 2026-11-06 09:35, **the bench's first run stopped after one call**: gemma4
  answered IC 410's first write in 797 s with 3,566 tokens, at qwen's
  tokens per second. Gemma 4 is a thinking model and Ollama has it on by
  default: a private reasoning pass before the answer, ~3,000 tokens of it.
  Probed on a one-line question: think off 9 tokens in 4 s, think on 383
  tokens in 89 s, the same answer. `local_think` (False) is sent as the
  chat API's `think` to a model whose `/api/show` capabilities include
  thinking (`local_capabilities`, `local_payload`); the log prints the
  thinking length when there is any. Also from that one call: with no
  distance on record and a 424-char Wikipedia block, gemma said "about
  10,000 light years" (the site says 12,000) and qwen's review flagged it
  five ways as unsupported, which is the adversarial pair doing its job;
  `known_distance(html=)` now takes the site's own `data-distance-ly` as
  the record for a target already in the gallery (every bench target, and
  a re-render), the file still first. Bench restarted with thinking off;
  the thinking-on log is bench/bench_run_thinking_on.log.
- 2026-11-06 11:30, **from the bench's first eight entries** (run 4, both
  models writing, each reviewed by the other; the bench itself untouched,
  these land for the next run): (1) Melotte 15 had no aliases on file and
  no page of its own, so the writer had no reference; gemma's first draft
  put it "in Aquila" with invented gas physics, qwen's review caught all
  nine statements, and the rewrite retreated to "Deep Space Object /
  UNKNOWN", honest and unpublishable. `dossier` now falls back to
  Wikipedia's search and takes a result whose intro names the object (the
  Heart Nebula's does), and `astro_output.object_type` reads the cropper's
  offline chain (queue obj_type, candidates kind) when the classifier has
  nothing: Melotte 15 now reads "nebula" with the Heart Nebula's page.
  (2) American spellings in both models' drafts (center, color, catalog):
  a British-spelling line in the writer's rules, American spelling as a
  class-3 defect for the reviewer, and `britishise` on the final text as
  the backstop (unambiguous words only, a capitalised mid-sentence word
  left as a name; the words changed ride the article as `spelling_fixed`
  and the log). (3) Gemma read 'ID - Name' as "IC 410 - ..." and qwen
  obeyed, so the reviewer gets the writer's example title. (4) Qwen's
  "retired when it reached the frame cap" (false; the two tests) was a
  statement about the observatory, which the claims rule does not cover:
  the writer may link a note but never describe what it found or why the
  target was chosen, and the reviewer treats such a statement as class 2.
  (5) Qwen flagged "seven nights between 25 October and 1 November" three
  times as implying the 30th, and one rewrite obeyed with "excluding 30
  October": a span of dates is not a defect, says the reviewer prompt.
  Pattern so far: qwen as critic passed every gemma draft in 2-6 s with no
  notes but Melotte 15's; gemma as critic found something in every qwen
  draft (title, em-dashes, unlinked notes), once wrongly.
- 2026-11-06 13:00, **the bench, run 4, complete** (bench/bench.md; the
  owner's review page ~/Desktop/prose_bench_review.html, every entry under
  the hand's with errors, doubts, spellings and the reviewer's catches
  marked). My reading, the owner's to overrule: gemma4 4 errors / 15
  doubts / 15 American spellings over six entries, approvable with one or
  two edits on five of six; qwen3-vl 15 / 19 / 17, approvable with three
  edits on four of six; Melotte 15 fails for both on the inputs (no
  reference, no type; fixed). Of qwen's 15 errors, four were titles its
  reviewer broke and five were invented purposes for M1's nine related
  notes; the rest and all of gemma's are in the 'On Earth' boxes (Uruk
  writing at 4500 BC, the Triassic at 180 Myr, no polar ice in the
  Oligocene), which no reference covers and neither reviewer checks. As
  critics: gemma finds something real in every qwen draft in ~20 s but
  misread the title rule every time; qwen passes gemma's drafts in 2-6 s
  with no notes, except Melotte 15 where it caught all nine inventions.
  Timing: gemma ~110 s a loop with no rewrite, qwen ~330 s with one.
  Recommendation: gemma4 writes, qwen3-vl reviews (the current default).
  Second-round fixes: `britishise` leaves a linked note's title alone (the
  hr-diagram note's own title is "color-magnitude"), more words; the note
  rule is one to three links whose titles concern the picture or the
  object, and the reviewer's class 4 is "none linked", not "one unlinked".
  Open: the 'On Earth' box needs a check of its own; the writer should be
  told to describe what it sees (qwen's picture paragraphs were the better).
- 2026-11-06 14:30, **the 'On Earth' box is a table, not the model's**
  (owner: "all the new objects and their basic times have been covered by
  other objects, it might be better to write a standard template for all
  the geological times, and have it point to the item in the template",
  and: cover the far ones too). `configs/on_earth.json`: 43 eras,
  contiguous from the last hundred years to the first galaxies (14
  billion), historical by AD/BC span, geological by the ICS chart, each
  with a name, a span, a one-line summary and a paragraph written to be
  true and dull. `scripts/on_earth.py`: `era_for(years_ago)`, the box
  ("This light left NGC 6946 about 25 million years ago, in [the late
  Oligocene](#era-oligocene): the world cooling, ..."), and the site's
  "When the light left" section, one anchored block per era, inserted at
  the end of the gallery panel between markers (`on_earth.py site`,
  `--push` to publish). `article_prose.compose` replaces the writer's box
  with the table's whenever a distance is known (the record, else the
  writer's own distance_ly); the writer is told one sentence, the
  reviewer that the box is not its concern; `build_fragment` renders the
  link. The harvested site boxes (scratch site_boxes.json) have slips of
  their own (NGC 6339 at 94 Myr with Tyrannosaurus, which is 68-66; NGC
  925 at 30 Myr with "terror birds"): they stay as the owner wrote them;
  new entries take the table. The timeline sits in the local site checkout
  uncommitted until the owner says push.
- 2026-11-06 evening, the timeline live on the site (site commit b52a4eb;
  `site --push` fixed to push a current file with an uncommitted diff).
  Eras whose name is not a period word take no early/middle/late
  (`phased: false`: "the late last ice age" read wrongly). **First entry
  through the whole pipeline**: IC 410 on a scratch copy, gemma4 writing,
  qwen reviewing, 125 s, one claim, no notes, no spelling to fix, the box
  from the table. One fault the critic cannot see: "the learned
  deconvolution network described in [Two decisions...]", the wrong
  content attributed to a linked note (the writer was told to link and not
  to describe; it did both). Next rule: a linked note is introduced by its
  title only, never "described in". The owner's pages:
  ~/Desktop/local_article_prose.html (the note, revised with the day's
  findings) and ~/Desktop/ic410_machine_article.html (the entry in the
  site's markup beside a link to the hand-written one).
- 2026-11-06 night, **published as a lab note**: "Who writes the caption"
  (irisscience.org/#machine-caption, site commit d8f0d1b): what the
  telescope decides already, the two sources of truth, the flow diagram,
  why two models, the bench table and what the errors were made of, the
  era table, IC 410 by hand and by machine, what it is not yet. The
  subscriber announcement is the owner's `scripts/site_push.py --yes` on
  the PC (the web-host key is not on the Spark). The timeline section
  moved OUT of the gallery panel (site d5c7742): the gallery sort
  re-appends the entry cards to the panel, so the section sat above the
  images ("the history text is before the images"); it is now a sibling
  between the panels, shown with the gallery tab by its own rule, and
  `with_timeline` migrates an old in-panel block.
