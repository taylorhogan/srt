---
name: publish
description: Publish finished pictures dropped in ~/Desktop/astro_output: one folder per target with the frames used, the stats picture, and a gallery article on irisscience.org when the site has none, written here and pushed. Use this whenever the user says /publish, says they put or dropped an image in astro_output, asks to publish or post a picture to the site or gallery, or wants the sub counts / stats / write-up for a finished image, even if they do not name the script.
---

# /publish

`scripts/astro_output.py` does the mechanical part; this skill is the loop around it, including the write-up that the script cannot do here (no Claude API credential on the Spark). Read its module docstring once if anything below is unclear: it is the authority on what each file means.

## 1. Run the script

```bash
cd /home/taylor/Documents/srt && source .venv/bin/activate && python -u scripts/astro_output.py 2>&1 | grep -v "fit may be unsuccessful\|RuntimeWarning\|m_fwhm\|logging at"
```

Options the user may ask for: `--no-push` (commit the article, do not push), `--no-publish` (data only, no site), `--folder DIR`.

For every image sitting directly in the folder it creates `<dso>/`, moves the image in, writes `<dso>_subs.txt` / `.json` (frames per filter the stack used, hours, nights) and `<dso>_stats.jpg`, then checks the site. Read the log: it says whether the target is already on the site ("matched by id/title/image/text", then nothing more to do) or that an article is pending. It is also where the "frames taken after the picture left out" and "replayed gate" caveats appear; carry those into the report.

The first run on a target may take a while if its frames are not yet in the stats cache; that is the per-frame measurement and it happens once.

## 2. Write each pending article

A pending article is a `<dso>/article_request.json` with no `article.json` beside it. For each one:

1. Read `article_request.json`: the measured facts (filters, frames, hours, nights, processing from the file name, related science notes). These numbers go into the entry's meta line by the script; never change them.
2. Look at the picture (Read the image) for the alt text and the description of what is visible.
3. Read the two live entries the script uses as style examples and any related note listed in `related_notes` (in `~/Documents/development/taylorhogan.github.io/index.html`). Match their voice. Read `references/article-style.md` for the rules and a finished example.
4. Write `<dso>/article.json` with exactly these keys: `title`, `distance_ly`, `distance_text`, `type_text`, `alt`, `paragraphs` (two), `time_travel_title`, `time_travel`. Link every related note from the second paragraph as `[its title](#its-id)`.

Only state facts about the object that are well established (distance, type, named features, supernova history). If unsure of something, leave it out rather than guess: the site is read by subscribers.

## 3. Run the script again

The same command. A folder with an `article.json` and no `article_published.json` is published: picture copied to `images/<id>.jpg`, entry inserted in the site's gallery markup, committed and pushed. Live within five minutes. Then verify, from the site repo:

```bash
git -C ~/Documents/development/taylorhogan.github.io log --oneline -3
```

and check the inserted entry renders the link(s) and the meta line as expected (`grep -n '<!-- NGC 6946 -->' -A 12 index.html` style).

## 4. Report

Say for each image: published or already on the site (and what matched), the frames line and hours, the related notes linked, and any caveat from the log. Subscriber announcements are not sent by this flow; `scripts/site_push.py` does that and needs the web-host key, which the Spark does not have. Nothing in the srt repo is committed by this skill.
