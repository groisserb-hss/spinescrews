"""Drift guard: the Slicer plugin's POSSIBLE_LEVELS must equal the pipeline's possible_levels.

The Hybrid Screw Planner (dicom_tools/HybridScrewPlanner) runs inside 3D Slicer's bundled Python
and cannot import the spinescrews package, so it keeps a verbatim copy of the accepted vertebral
levels. This test enforces that the copy stays in sync with the source of truth in
src/spinescrews/tools/__init__.py — if they drift, plan export would validate names against a
different level set than the pipeline accepts.

Runnable two ways:
    python dicom_tools/tests/test_possible_levels_sync.py     # prints PASS/SKIP/FAIL
    pytest dicom_tools/tests/                                  # if pytest installed

Runs in the screws310 env (where spinescrews is installed); needs no 3D Slicer and no data.
"""

import ast
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_PLUGIN = os.path.normpath(
    os.path.join(_HERE, "..", "HybridScrewPlanner", "HybridScrewPlanner.py"))

# ---- skip mechanism compatible with pytest and the plain runner ----
try:
    import pytest
    _SKIP_EXC = pytest.skip.Exception

    def skip(msg):
        pytest.skip(msg)
except Exception:                       # pytest not installed
    class _SkipExc(Exception):
        pass
    _SKIP_EXC = _SkipExc

    def skip(msg):
        raise _SkipExc(msg)


def _plugin_possible_levels():
    """Extract the plugin's module-level POSSIBLE_LEVELS literal (no Slicer import needed)."""
    with open(_PLUGIN) as f:
        tree = ast.parse(f.read(), filename=_PLUGIN)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "POSSIBLE_LEVELS":
                    return tuple(ast.literal_eval(node.value))
    raise AssertionError("POSSIBLE_LEVELS not found in %s" % _PLUGIN)


def test_possible_levels_in_sync():
    try:
        from spinescrews.tools import possible_levels
    except Exception as e:                  # spinescrews not installed in this interpreter
        skip("spinescrews not importable (run in the screws310 env): %s" % e)
    plugin_levels = _plugin_possible_levels()
    assert tuple(possible_levels) == plugin_levels, (
        "POSSIBLE_LEVELS in HybridScrewPlanner.py is out of sync with possible_levels in "
        "src/spinescrews/tools/__init__.py:\n"
        "  plugin:  %r\n  package: %r" % (plugin_levels, tuple(possible_levels)))


if __name__ == "__main__":
    try:
        test_possible_levels_in_sync()
    except _SKIP_EXC as e:
        print("SKIP  test_possible_levels_in_sync:", e)
    except AssertionError as e:
        print("FAIL  test_possible_levels_in_sync:", e)
        raise SystemExit(1)
    else:
        print("PASS  test_possible_levels_in_sync")
