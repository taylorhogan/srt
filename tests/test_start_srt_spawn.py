"""Every Process target in start_srt must be a MODULE-LEVEL function.

Windows multiprocessing spawns a fresh interpreter that re-imports start_srt
as __mp_main__ with the `if __name__ == "__main__"` block skipped, then
unpickles the target by name. A target defined inside that block pickles fine
in the parent and does not exist in the child, which dies with exit 1 before
the service runs a line. From 2026-08-28 to 2026-09-06 the shadow conductor
was launched exactly that way and never once came up under start_srt; the
only symptom was port 8096 not listening, and nothing was looking.

AST rather than import, for the same reason as test_compile: start_srt drags
in the whole observatory at import time and CI has none of it.
"""
import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parent.parent / "end_points" / "start_srt.py"


def _process_targets(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Process":
            for kw in node.keywords:
                if kw.arg == "target":
                    yield kw.value


def _root_name(expr):
    while isinstance(expr, ast.Attribute):
        expr = expr.value
    return expr.id if isinstance(expr, ast.Name) else None


def test_process_targets_are_module_level():
    tree = ast.parse(SRC.read_text(encoding="utf-8"), filename=str(SRC))
    top = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    top |= {a.asname or a.name.split(".")[0]
            for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
            for a in n.names}
    targets = list(_process_targets(tree))
    assert targets, "start_srt no longer launches anything with Process?"
    nested = [ast.unparse(t) for t in targets if _root_name(t) not in top]
    assert not nested, (
        "Process target(s) not defined at module level -- the spawned child "
        "cannot unpickle them and dies silently: %s" % nested)
