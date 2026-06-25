import functools
import json
import logging
import os.path
from datetime import datetime

from astropy.utils.iers import conf
conf.auto_max_age = None
from astropy.utils import iers
iers.conf.auto_download = False
from astropy.time import Time

from iris_astronomy import astro_dso_visibility
from cmd_processing import social_server
from configs import config

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_INSTRUCTIONS_PATH = os.path.join(_PROJECT_ROOT, config.data()["location"]["instructions"])

status_dict = {"in process": 3, "waiting": 2, "completed": 1}


def delete_instruction_db(hash_value):
    with open(_INSTRUCTIONS_PATH, 'r') as f:
        instructions = json.load(f)
    for instruction in instructions:
        if instruction["hash"] == hash_value:
            instructions.remove(instruction)
    with open(_INSTRUCTIONS_PATH, 'w') as f:
        f.writelines(json.dumps(instructions, indent=4))


def set_completed_instruction_db(hash_value):
    logger = logging.getLogger(__name__)
    logger.info("completing", hash_value)

    with open(_INSTRUCTIONS_PATH, 'r') as f:
        instructions = json.load(f)
    for instruction in instructions:
        if instruction["hash"] == hash_value:
            instruction["status"] = "completed"
    with open(_INSTRUCTIONS_PATH, 'w') as f:
        f.writelines(json.dumps(instructions, indent=4))


def remove_hash():
    with open(_INSTRUCTIONS_PATH, 'r') as f:
        instructions = json.load(f)
    for instruction in instructions:
        if 'hash' in instruction.keys():
            del instruction["hash"]

    with open(_INSTRUCTIONS_PATH, 'w') as f:
        f.writelines(json.dumps(instructions, indent=4))


def _normalize_dso(name: str) -> str:
    return name.lower().replace(" ", "")


def normalize_and_deduplicate_db() -> int:
    """Normalize all DSO names (lowercase, no spaces) and remove duplicates.

    For each set of entries that share a normalized name, the highest-priority
    entry (first in sort order) is kept and the rest are removed.
    Returns the number of entries removed.
    """
    with open(_INSTRUCTIONS_PATH, 'r') as f:
        instructions = json.load(f)

    for instruction in instructions:
        instruction["dso"] = _normalize_dso(instruction["dso"])

    sorted_instructions = sorted(instructions, key=functools.cmp_to_key(compare))
    seen: set[str] = set()
    deduped = []
    for instruction in sorted_instructions:
        key = instruction["dso"]
        if key not in seen:
            seen.add(key)
            deduped.append(instruction)

    removed = len(instructions) - len(deduped)
    with open(_INSTRUCTIONS_PATH, 'w') as f:
        f.writelines(json.dumps(deduped, indent=4))
    return removed


def rehash_db():
    remove_hash()
    next_hash = 0
    hash_set = {-1}
    # Persist on-disk status verbatim: the convergence-driven "done" flip is
    # display-only and must not rewrite a user-set "waiting" to "completed".
    instructions = get_sorted_instructions(apply_convergence=False)

    for instruction in instructions:
        instruction["hash"] = str(next_hash)
        next_hash = next_hash + 1

    with open(_INSTRUCTIONS_PATH, 'w') as f:
        f.writelines(json.dumps(instructions, indent=4))


def calc_and_store_hours_above_horizon(force=False):
    with open(_INSTRUCTIONS_PATH, 'r') as f:
        instructions = json.load(f)
    for instruction in instructions:
        dso = text = instruction["dso"]
        # Stored RA/Dec wins over a name lookup (positional targets have no
        # resolvable name); falls back to Simbad for named DSOs.
        obj = astro_dso_visibility.resolve_target(instruction)
        if obj is not None:
            above, max_altitude  = astro_dso_visibility.get_above_horizon_time(obj, Time.now())
            instruction["above_horizon"] = str(above)
            instruction["air_mass"] = "{:.2f}".format(astro_dso_visibility.air_mass (max_altitude))
        else:
            instruction["above_horizon"] = '0'
            instruction["air_mass"] = '0'

        value = instruction.get("best",None)

        if obj is None:
            instruction['best'] = 'None'
        elif value is None or force is True:
            best_date, best_time, max_altitude = astro_dso_visibility.best_day_for_dso(obj)
            if best_date is None:
                instruction['best'] = 'None'
            else:
                formatted_date = best_date.strftime("%Y-%m-%d")
                instruction['best']=formatted_date + "\n"+str(best_time)


    with open(_INSTRUCTIONS_PATH, 'w') as f:
        f.writelines(json.dumps(instructions, indent=4))


def time_to_seconds(time_str):
    """Converts a time string (HH:MM:SS or MM:SS or SS) to seconds."""
    if time_str is None:
        return 0
    if time_str == "None":
        return 0

    parts = time_str.split(":")
    parts.reverse()
    seconds = 0
    multiplier = 1
    for part in parts:
        seconds += int(part) * multiplier
        multiplier *= 60
    return seconds


def compare(r1, r2):
    s1 = r1["status"]
    s2 = r2["status"]
    s1p = status_dict.get(s1, 0)
    s2p = status_dict.get(s2, 0)

    if s1p < s2p:
        return 1
    if s1p > s2p:
        return -1

    p1 = r1.get("priority", 5)
    p2 = r2.get("priority", 5)

    if p1 < p2:
        return 1
    if p1 > p2:
        return -1

    oh1 = time_to_seconds(r1.get("above_horizon","0"))
    oh2 = time_to_seconds(r2.get("above_horizon","0"))

    if oh1 < oh2:
        return 1
    if oh1 > oh2:
        return -1

    n1 = r1["dso"]
    n2 = r2["dso"]

    if n1 < n2:
        return 1
    if n1 > n2:
        return -1

    return 0


def create_instructions_table(force = False):
    rehash_db()
    calc_and_store_hours_above_horizon(force)
    sorted_l = get_sorted_instructions()
    logger = logging.getLogger(__name__)
    logger.info("in create")


    social_server.post_html_message(_instructions_table_html(sorted_l))


# GitHub-dark palette, matching the "Active targets" table in super_user_commands.
_TBL_BORDER, _TBL_ROW, _TBL_DIM, _TBL_TEXT, _TBL_ACCENT, _TBL_BRIGHT = (
    "#30363d", "#21262d", "#8b949e", "#c9d1d9", "#3fb950", "#e6edf3")
_TBL_BG, _TBL_TILE_BORDER = "#0d1117", "#30363d"

# (text colour, pill background) per status — preserves the old
# green/blue/pink colour coding in the dark theme.
_STATUS_STYLE = {
    "in process": ("#3fb950", "rgba(63,185,80,0.15)"),
    "waiting":    ("#58a6ff", "rgba(88,166,255,0.15)"),
    "completed":  ("#f778ba", "rgba(247,120,182,0.14)"),
}


def _instructions_table_html(sorted_l, per_page=None):
    """Render the imaging queue as a styled HTML table for the web chat.

    Matches the look of the other chat tables (dark card, dim header row,
    tabular-nums, status pills) instead of the old matplotlib PNG.

    `per_page` caps the number of rows shown; None (default) shows all.
    """
    from html import escape

    th = (f'padding:4px 9px;border-bottom:1px solid {_TBL_BORDER};color:{_TBL_DIM};'
          f'font-size:10.5px;font-weight:600;white-space:nowrap;')
    td = (f'padding:5px 9px;border-bottom:1px solid {_TBL_ROW};color:{_TBL_TEXT};'
          f'font-variant-numeric:tabular-nums;vertical-align:top;')

    rows = sorted_l if per_page is None else sorted_l[:per_page]
    headers = [('DSO', 'left'), ('Requestor', 'left'), ('State', 'left'),
               ('Best Date', 'left'), ('Tonight', 'right'),
               ('Air Mass', 'right'), ('ID', 'right')]

    def _two_line(main, sub):
        h = f'<span style="color:{_TBL_TEXT};">{escape(str(main))}</span>'
        if sub:
            h += (f'<br><span style="color:{_TBL_DIM};font-size:10px;">'
                  f'{escape(str(sub))}</span>')
        return h

    out = [
        f'<div style="font-size:13px;font-weight:600;color:{_TBL_ACCENT};'
        f'margin-bottom:10px;">Imaging queue — {len(rows)} object{"s" if len(rows) != 1 else ""}</div>',
        f'<div style="background:{_TBL_BG};border:1px solid {_TBL_TILE_BORDER};'
        f'border-radius:8px;padding:10px 12px;display:inline-block;'
        f'max-width:100%;overflow-x:auto;">',
        '<table style="border-collapse:collapse;font-size:11px;"><thead><tr>',
    ]
    for label, align in headers:
        out.append(f'<th style="{th}text-align:{align};">{escape(label)}</th>')
    out.append('</tr></thead><tbody>')

    for instr in rows:
        status = instr.get("status", "")
        s_fg, s_bg = _STATUS_STYLE.get(status, (_TBL_DIM, "rgba(139,148,158,0.12)"))
        pill = (f'<span style="display:inline-block;padding:1px 8px;border-radius:9px;'
                f'background:{s_bg};color:{s_fg};font-size:10px;font-weight:600;'
                f'white-space:nowrap;">{escape(status)}</span>')

        # best = "YYYY-MM-DD\nHH:MM:SS(.ffffff)" — show date, time dimmed (no microsec)
        best = (instr.get("best") or "").split("\n")
        best_date = best[0].strip()
        best_time = best[1].strip().split(".")[0] if len(best) > 1 else ""

        cells = [
            f'<td style="{td}text-align:left;font-weight:600;color:{_TBL_BRIGHT};'
            f'white-space:nowrap;">{escape(str(instr.get("dso", "")))}</td>',
            f'<td style="{td}text-align:left;white-space:nowrap;">'
            f'{_two_line(instr.get("requestor", ""), instr.get("request_time", ""))}</td>',
            f'<td style="{td}text-align:left;">{pill}</td>',
            f'<td style="{td}text-align:left;white-space:nowrap;">'
            f'{_two_line(best_date, best_time)}</td>',
            f'<td style="{td}text-align:right;white-space:nowrap;">'
            f'{escape(str(instr.get("above_horizon", "")))}</td>',
            f'<td style="{td}text-align:right;">{escape(str(instr.get("air_mass", "")))}</td>',
            f'<td style="{td}text-align:right;color:{_TBL_DIM};">'
            f'{escape(str(instr.get("hash", "")))}</td>',
        ]
        out.append('<tr>' + "".join(cells) + '</tr>')

    out.append('</tbody></table></div>')
    return "".join(out)


def reset_all_priorities_db() -> int:
    """Reset every waiting instruction's priority to 5 (the natural default).
    Returns the number of instructions updated."""
    with open(_INSTRUCTIONS_PATH, 'r') as f:
        instructions = json.load(f)
    count = 0
    for instruction in instructions:
        if instruction.get("status") == "waiting":
            instruction["priority"] = 5
            count += 1
    with open(_INSTRUCTIONS_PATH, 'w') as f:
        f.writelines(json.dumps(instructions, indent=4))
    return count


def set_priority_instruction_db(dso_name: str, priority: int) -> bool:
    """Set the priority of the first waiting instruction matching dso_name.
    Returns True if a match was found and updated, False otherwise."""
    with open(_INSTRUCTIONS_PATH, 'r') as f:
        instructions = json.load(f)
    matched = False
    for instruction in instructions:
        if instruction["dso"].lower().replace(" ", "") == dso_name.lower().replace(" ", "") and instruction["status"] == "waiting":
            instruction["priority"] = priority
            matched = True
            break
    if matched:
        with open(_INSTRUCTIONS_PATH, 'w') as f:
            f.writelines(json.dumps(instructions, indent=4))
    return matched


def add_dso_object_instruction(dso_name, recipe, requestor, priority=5,
                               ra_deg=None, dec_deg=None):
    normalized = dso_name.lower().replace(" ", "")
    now = datetime.now()
    formatted_date = now.strftime("%Y-%m-%d")
    with open(_INSTRUCTIONS_PATH, 'r') as f:
        instructions = json.load(f)
    for instruction in instructions:
        if (instruction["dso"].lower().replace(" ", "") == normalized
                and instruction["status"] != "completed"):
            return False
    new_instruction = {
        "dso": normalized,
        "uuid": "1",
        "recipe": recipe,
        "requestor": requestor,
        "request_time": formatted_date,
        "status": "waiting",
        "priority": priority
    }
    # Persist an explicit position so this target resolves without a name lookup.
    if ra_deg is not None and dec_deg is not None:
        new_instruction["ra_deg"] = ra_deg
        new_instruction["dec_deg"] = dec_deg
    instructions.append(new_instruction)
    with open(_INSTRUCTIONS_PATH, 'w') as f:
        f.writelines(json.dumps(instructions, indent=4))
    return True


def get_instruction_by_dso(dso_name):
    """Return the instruction record matching dso_name (normalized), or None."""
    normalized = _normalize_dso(dso_name)
    with open(_INSTRUCTIONS_PATH, 'r') as f:
        instructions = json.load(f)
    for instruction in instructions:
        if _normalize_dso(instruction["dso"]) == normalized:
            return instruction
    return None


def resolve_target_by_name(dso_name):
    """Resolve a name to a FixedTarget, preferring a queued record's stored
    RA/Dec so positional targets resolve without a Simbad lookup. Falls back to
    a plain name lookup when the name isn't queued."""
    instruction = get_instruction_by_dso(dso_name)
    if instruction is not None:
        return astro_dso_visibility.resolve_target(instruction)
    return astro_dso_visibility.is_a_dso_object(dso_name)


def get_sorted_instructions(apply_convergence=True):
    """Return instructions sorted for display/selection.

    When *apply_convergence* is True, a ``waiting`` target the convergence
    model considers done is shown as ``completed`` (so it drops out of nightly
    selection). This flip is advisory and must never be written back to disk —
    only ``set_completed_instruction_db`` (the explicit ``dbc`` command) may
    persist a completion. Callers that write the queue back (e.g.
    ``rehash_db``) must pass ``apply_convergence=False`` so a user-set
    ``waiting`` status is preserved on disk.
    """
    from fits_processing.convergence import is_dso_done

    with open(_INSTRUCTIONS_PATH, 'r') as f:
        instructions = json.load(f)

    scored = []
    for instr in instructions:
        d = dict(instr)
        if apply_convergence and d.get("status") == "waiting":
            try:
                if is_dso_done(d["dso"]):
                    d["status"] = "completed"
            except Exception:
                pass
        scored.append(d)

    return sorted(scored, key=functools.cmp_to_key(compare))


def get_dso_object_tonight():
    sorted_l = get_sorted_instructions()

    best = sorted_l[0]

    return best


if __name__ == "__main__":
    rehash_db()
    create_instructions_table()
    #set_completed_instruction_db(0)
    # delete_instruction_db(4)
