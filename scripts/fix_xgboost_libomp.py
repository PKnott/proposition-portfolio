"""Make xgboost's bundled libxgboost.dylib find an OpenMP runtime on macOS.

Why this exists
---------------
The xgboost macOS wheel is linked against ``@rpath/libomp.dylib`` and ships with a
single rpath entry: ``/opt/homebrew/opt/libomp/lib``. On a machine without
Homebrew that path does not exist, so ``import xgboost`` fails with::

    XGBoostError: XGBoost Library (libxgboost.dylib) could not be loaded.
    Library not loaded: @rpath/libomp.dylib

The documented fix is ``brew install libomp``, which requires installing
Homebrew (and admin rights). We avoid that: the scikit-learn wheel already
bundles ``libomp.dylib``, so we copy it next to ``libxgboost.dylib`` and add
``@loader_path`` as an rpath.

Loading *one* shared OpenMP runtime for both scikit-learn and xgboost is the
correct configuration anyway -- the classic macOS crash comes from loading two
different OpenMP runtimes into one process, which this specifically avoids.

Re-run this after any ``pip install``/upgrade of xgboost, which restores the
pristine wheel.

Usage:  python scripts/fix_xgboost_libomp.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True)


def _rpaths(dylib: Path) -> list[str]:
    out = _run("otool", "-l", str(dylib)).stdout
    paths, lines = [], out.splitlines()
    for i, line in enumerate(lines):
        if "LC_RPATH" in line:
            for follow in lines[i : i + 4]:
                if "path " in follow:
                    paths.append(follow.split("path ", 1)[1].split(" (offset")[0].strip())
    return paths


def main() -> int:
    if sys.platform != "darwin":
        print("Not macOS - nothing to do.")
        return 0

    try:
        import sklearn  # noqa: F401  (import for its file location only)
        import xgboost.libpath  # noqa: F401
    except Exception:
        pass

    import sklearn

    src = Path(sklearn.__file__).parent / ".dylibs" / "libomp.dylib"
    if not src.exists():
        print(f"ERROR: no bundled libomp in scikit-learn at {src}")
        print("Install libomp another way (e.g. `brew install libomp`).")
        return 1

    # Locate the xgboost wheel's lib directory without importing xgboost
    # (importing is exactly what fails before this fix is applied).
    import importlib.util

    spec = importlib.util.find_spec("xgboost")
    if spec is None or not spec.submodule_search_locations:
        print("ERROR: xgboost is not installed.")
        return 1
    xgb_lib = Path(list(spec.submodule_search_locations)[0]) / "lib"
    dylib = xgb_lib / "libxgboost.dylib"
    if not dylib.exists():
        print(f"ERROR: {dylib} not found.")
        return 1

    dest = xgb_lib / "libomp.dylib"
    if not dest.exists():
        shutil.copy2(src, dest)
        print(f"Copied {src}  ->  {dest}")
    else:
        print(f"Already present: {dest}")

    if "@loader_path" in _rpaths(dylib):
        print("rpath @loader_path already set.")
    else:
        r = _run("install_name_tool", "-add_rpath", "@loader_path", str(dylib))
        if r.returncode != 0:
            print("ERROR: install_name_tool failed:\n" + r.stderr)
            return 1
        print("Added rpath @loader_path to libxgboost.dylib")
        # Editing a Mach-O invalidates its signature on Apple silicon; re-sign ad-hoc.
        _run("codesign", "--force", "--sign", "-", str(dylib))

    # Prove it worked, in a clean interpreter.
    check = _run(
        sys.executable,
        "-c",
        "import xgboost, numpy as np;"
        "m=xgboost.XGBRegressor(objective='count:poisson',n_estimators=5,verbosity=0);"
        "m.fit(np.random.rand(50,3), np.random.poisson(1.4,50));"
        "print('xgboost', xgboost.__version__, 'OK')",
    )
    print(check.stdout.strip() or check.stderr.strip()[-500:])
    return check.returncode


if __name__ == "__main__":
    raise SystemExit(main())
