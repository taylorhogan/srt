"""The one-tap publishing gate (D3 of docs/AUTOMATION_PLAN.md, 2026-10-30).

A draft the Spark prepared (scripts/publish_prepare) is one card in the chat
and one decision by the owner: `approve <dso>` or `reject <dso> [why]`. Silence
publishes after publish.auto_approve_days -- once the gate is live. Every
part that decides is here and pure:

  * card_text / fail_card_text   what the chat shows
  * decide(record, decision, today, gate_live)   publish | wait | rejected | stale | none
  * latest_drafts(entries)        the drafts the journal knows (DRAFT_READY notes)
  * record_decision / load_decisions   local/publish_decisions.json on the PC

The hash is the contract. A draft is the article as written plus the
picture (astro_output.draft_hash); an approve names the hash it saw, and a
draft re-rendered since (new frames, new words) carries a new hash, so the
old approve is stale and the card comes again. The Spark learns the hash of
the draft it posted; the PC learns it from the DRAFT_READY note the Spark
journals through scripts/post_event.py (facts flow one way; the PC keeps
the decisions, the Spark reads them: D4).

cfg["publish"]["gate"]: "off" (no cards, no timer), "shadow" (cards and
commands work, the timer is not armed, nothing publishes), "live".
"""
import json
import os
from datetime import date, datetime
from typing import Optional

GATE_MODES = ("off", "shadow", "live")
DEFAULT_GATE = "shadow"
DECISIONS_FILE = os.path.join("local", "publish_decisions.json")
DRAFT_EVENT = "DRAFT_READY"
APPROVED_EVENT, REJECTED_EVENT, DONE_EVENT = "PUBLISH_APPROVED", "PUBLISH_REJECTED", "PUBLISH_DONE"


def gate_mode(cfg) -> str:
    m = str(((cfg or {}).get("publish") or {}).get("gate", DEFAULT_GATE)).lower()
    return m if m in GATE_MODES else DEFAULT_GATE


# ------------------------------------------------------------------ the cards

def card_text(rec: dict, article: dict, facts: dict, gate: str = DEFAULT_GATE) -> str:
    """The chat card for a draft: title, the site's meta line, the paragraphs,
    the light-travel box when there is one, who wrote it, and the reply."""
    dso = rec.get("dso", "")
    qc = (rec.get("qc") or {})
    verdict = "QC PASS" if qc.get("pass") else "QC FAIL"
    head = (f"PUBLISH DRAFT: {article.get('title', dso)}   "
            f"({rec.get('recipe', '?')}, {rec.get('product', '?')}; {verdict})")
    meta_bits = []
    if article.get("distance_text"):
        meta_bits.append("DISTANCE: " + str(article["distance_text"]))
    if article.get("type_text"):
        meta_bits.append("TYPE: " + str(article["type_text"]))
    if facts.get("meta_frames"):
        meta_bits.append(str(facts["meta_frames"]).replace("&times;", "x"))
    if facts.get("hours") is not None and facts.get("nights") is not None:
        n = len(facts.get("nights") or [])
        meta_bits.append(f"{facts['hours']} HOURS OVER {n} NIGHT{'S' if n != 1 else ''}")
    lines = [head, " | ".join(meta_bits)]
    lines += [str(p).strip() for p in article.get("paragraphs") or []]
    if (article.get("time_travel") or "").strip():
        lines.append(f"{article.get('time_travel_title', 'On Earth')}: {str(article['time_travel']).strip()}")
    src = rec.get("article_source") or article.get("source") or "?"
    lines.append("Written by: " + ("the model" if src == "api" else "facts only (no model; the key will replace it)"))
    tail = f"Reply `approve {dso}` or `reject {dso} [why]`."
    if rec.get("auto_publish_on"):
        if gate == "live":
            tail += f" Auto-publishes {rec['auto_publish_on']}."
        else:
            tail += f" Would auto-publish {rec['auto_publish_on']} (gate: {gate}; nothing publishes yet)."
    lines.append(tail)
    return "\n".join(line for line in lines if line)


def fail_card_text(dso: str, recipe: str, qc: dict) -> str:
    """The card for a final that did not pass: what failed and what to do."""
    reasons = "; ".join((qc or {}).get("reasons") or ["no verdict"])
    return (f"FINAL RENDER NEEDS A HAND: {dso} {recipe} -- QC FAIL: {reasons}\n"
            f"Try `process {dso} {recipe} auto` and `keep` the variant you like; "
            "no timer is armed for this one.")


# ------------------------------------------------------------------ the decision

def decide(record: Optional[dict], decision: Optional[dict], today: date, gate_live: bool) -> dict:
    """What to do with a queue record given the owner's decision (or none).

    -> {"action": "publish" | "wait" | "rejected" | "stale" | "none", "why": str}
      none     no pending draft (no record, on the site already, QC failed)
      publish  approved for THIS hash, or the timer ran out with the gate live
      rejected rejected for this hash
      stale    a decision for another hash: the draft changed since; a fresh card is due
      wait     nothing decided and the timer has not run out (or the gate is not live)
    """
    if not record or record.get("status") != "pending" or not record.get("draft_hash"):
        return {"action": "none", "why": "no pending draft (%s)" % ((record or {}).get("status") or "no record")}
    h = record["draft_hash"]
    if decision:
        if decision.get("draft_hash") and decision["draft_hash"] != h:
            return {"action": "stale", "why": "decided on draft %s, this is %s" % (
                str(decision["draft_hash"])[:12], h[:12])}
        d = str(decision.get("decision") or "").lower()
        if d == "approve":
            return {"action": "publish", "why": "approved by %s %s" % (decision.get("by", "?"), decision.get("at", ""))}
        if d == "reject":
            return {"action": "rejected", "why": "rejected by %s: %s" % (decision.get("by", "?"), decision.get("why") or "no reason given")}
    on = record.get("auto_publish_on")
    if on and gate_live and today >= date.fromisoformat(str(on)[:10]):
        return {"action": "publish", "why": "no answer by %s: auto-publish" % on}
    if on and not gate_live:
        return {"action": "wait", "why": "awaiting approve/reject (gate not live; the timer is not armed)"}
    return {"action": "wait", "why": "awaiting approve/reject; auto-publishes %s" % (on or "never")}


# ------------------------------------------------------------------ the journal's drafts

def latest_drafts(entries) -> dict:
    """{dso: DRAFT_READY data} for drafts not since PUBLISH_DONE, from journal
    entries (dicts with event/kind/data, or iris.core.journal Entry). Pure."""
    out = {}
    for e in entries:
        ev = getattr(e, "event", None) or (e.get("event") if isinstance(e, dict) else None)
        data = getattr(e, "data", None) if not isinstance(e, dict) else e.get("data")
        data = data or {}
        dso = str(data.get("dso") or "").lower().replace(" ", "")
        if not dso:
            continue
        if ev == DRAFT_EVENT:
            out[dso] = dict(data, dso=dso)
        elif ev == DONE_EVENT:
            out.pop(dso, None)
    return out


def read_journal_drafts(root: str, days: int = 21) -> dict:
    """latest_drafts over the PC's journal files (local/journal); {} if unreadable."""
    try:
        from pathlib import Path

        from iris.core.journal import Journal
        return latest_drafts(Journal(Path(root) / "local" / "journal").replay(days=days))
    except Exception:  # noqa: BLE001
        return {}


# ------------------------------------------------------------------ the decisions file

def load_decisions(path) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def record_decision(path, dso: str, decision: str, draft_hash: Optional[str], by: str,
                    why: str = "", now: Optional[datetime] = None) -> dict:
    """Write {dso: {decision, draft_hash, by, at, why}}; -> the entry."""
    now = now or datetime.now()
    d = load_decisions(path)
    key = str(dso).lower().replace(" ", "")
    entry = {"decision": decision, "draft_hash": draft_hash, "by": by,
             "at": now.isoformat(timespec="minutes"), "why": why or ""}
    d[key] = entry
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=1)
    return entry


def drafts_lines(drafts: dict, decisions: dict, today: date, gate: str) -> list:
    """One line per known draft for the `drafts` command."""
    out = []
    for dso, d in sorted(drafts.items()):
        rec = {"dso": dso, "status": "pending", "draft_hash": d.get("draft_hash"),
               "auto_publish_on": d.get("auto_publish_on")}
        v = decide(rec, decisions.get(dso), today, gate == "live")
        out.append("%s  %s %s  draft %s  %s -- %s: %s" % (
            dso, d.get("recipe", "?"), d.get("product", "?"), str(d.get("draft_hash") or "")[:12],
            "model" if d.get("article_source") == "api" else "facts only", v["action"], v["why"]))
    return out or ["no drafts awaiting a decision"]
