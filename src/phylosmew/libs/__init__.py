from pathlib import Path
import sys

# Path to the git submodule on disk inside your installed package
_RGS_ROOT = Path(__file__).resolve().parent / "RAxMLGroveScripts"

# Make the submodule’s repo root importable so that `import tools` in org_script.py works.
p = str(_RGS_ROOT)
if _RGS_ROOT.exists() and p not in sys.path:
    sys.path.insert(0, p)