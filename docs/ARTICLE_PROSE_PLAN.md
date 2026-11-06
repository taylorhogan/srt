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
  model `gemma4:31b-it-q8_0` (34 GB, Apache 2.0, image input); the bench
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
4. **The card** (`publish_gate.card_text`) shows who wrote it and the
   model, the claims to check, and the reviewer's notes. The owner's
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
    curl -fsSL https://ollama.com/download/ollama-linux-arm64.tgz | tar -xz
    ~/.local/ollama/bin/ollama serve &            # or let the writer start it
    ~/.local/ollama/bin/ollama pull gemma4:31b-it-q8_0
    ~/.local/ollama/bin/ollama pull qwen3-vl:32b-instruct-q8_0
    cd ~/Documents/srt && .venv/bin/python scripts/article_prose.py bench \
        --models gemma4:31b-it-q8_0,qwen3-vl:32b-instruct-q8_0

Models live in `~/.ollama/models` (70 GB for the two; 2.2 TB free).

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
