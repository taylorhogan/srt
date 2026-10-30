"""docs/tasks and the heir docs (F4 of docs/AUTOMATION_PLAN.md, 2026-10-30): every
scheduled job names a script that exists, and the handbook names every hold and
its release, so a stranger can act on an alert from the handbook alone."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_spark_crontab_names_existing_scripts():
    text = (ROOT / "docs" / "tasks" / "spark_crontab.txt").read_text()
    jobs = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    assert len(jobs) >= 5
    for ln in jobs:
        m = re.search(r"/home/taylor/Documents/srt/(scripts/\S+)", ln)
        assert m, ln
        assert (ROOT / m.group(1)).exists(), m.group(1)


def test_pc_task_table_names_existing_scripts():
    text = (ROOT / "docs" / "tasks" / "README.md").read_text()
    rows = re.findall(r"^\| (Iris\w+) \| (scripts/\S+) \|", text, re.M)
    assert len(rows) >= 10
    for task, script in rows:
        assert (ROOT / script).exists(), f"{task}: {script}"


def test_any_exported_task_xml_names_an_existing_script():
    for xml in (ROOT / "docs" / "tasks").glob("Iris*.xml"):
        text = xml.read_text(encoding="utf-16", errors="ignore") or xml.read_text(errors="ignore")
        cmd = re.search(r"<Command>([^<]*)</Command>", text)
        assert cmd, xml.name
        # IrisMachineSnapshot runs the iris-machine repo's script, not ours
        m = re.search(r"[\\/]srt[\\/]((?:scripts|apps)[\\/][\w.]+)", cmd.group(1))
        if not m:
            continue
        assert (ROOT / m.group(1).replace("\\", "/")).exists(), xml.name


def test_handbook_covers_every_hold_and_its_release():
    hb = (ROOT / "docs" / "HANDBOOK.md").read_text()
    src = (ROOT / "iris" / "core" / "machine.py").read_text()
    holds = re.search(r"HOLD_STATES = frozenset\(\{([^}]*)\}\)", src).group(1)
    for state in re.findall(r'"(\w+)"', holds):
        assert state in hb, state
    for cmd in ("resolve!", "safe!", "stop!", "reopen", "approve", "reject", "drafts", "filters", "hold",
                "moon", "maxhours", "type", "suggest", "update", "replan"):
        assert "`" + cmd in hb or "`" + cmd + " " in hb, cmd
    for flag in ("planner.auto_stop", "nina.filter_advisor", "publish.final_mode", "publish.gate"):
        assert flag in hb, flag


def test_succession_names_every_private_credential():
    sc = (ROOT / "docs" / "SUCCESSION.md").read_text()
    blank = (ROOT / "configs" / "config_blank_private.py").read_text()
    for key in re.findall(r'^\s{8}"(\w+)": \{', blank, re.M):
        assert key in sc.lower() or key in sc, key
    for word in ("Tailscale", "Pushover", "GitHub", "OneDrive", "Anthropic", "domain"):
        assert word in sc, word
