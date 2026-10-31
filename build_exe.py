"""Build the Windows package: dist\\SCAPSync\\SCAPSync.exe (+ its _internal folder) and a zip of it.
Run on Windows with Python 3.9+:  py -3 build_exe.py
The package contains Python and every library SCAPSync needs; users only need SCAPS.
"""
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fermipilot import __version__  # noqa: E402

NEEDED = ["pyinstaller", "streamlit", "pandas", "numpy", "altair", "scikit-learn", "matplotlib", "openpyxl"]


def sh(cmd):
    print(">", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=HERE)


def main():
    t0 = time.time()
    print("Python", sys.version, flush=True)
    sh([sys.executable, "-m", "pip", "install", "--disable-pip-version-check"] + NEEDED)
    sep = ";" if os.name == "nt" else ":"
    for d in ("build", os.path.join("dist", "SCAPSync")):
        shutil.rmtree(os.path.join(HERE, d), ignore_errors=True)
    sh([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir", "--console",
        "--name", "SCAPSync",
        "--add-data", f"fermipilot_app.py{sep}.",
        "--add-data", f"fermipilot{sep}fermipilot",
        "--collect-all", "streamlit", "--collect-all", "altair",
        "--copy-metadata", "streamlit", "--copy-metadata", "pandas", "--copy-metadata", "numpy",
        "--copy-metadata", "altair", "--copy-metadata", "scikit-learn", "--copy-metadata", "matplotlib",
        "--collect-submodules", "fermipilot",
        "launcher.py"])
    dist = os.path.join(HERE, "dist", "SCAPSync")
    shutil.copy(os.path.join(HERE, "README.md"), os.path.join(dist, "README.md"))
    with open(os.path.join(dist, "HOW TO START.txt"), "w", encoding="utf-8") as f:
        f.write("Double-click SCAPSync.exe. SCAPSync opens in your web browser.\n"
                "Keep the black window open while you use it; close it to quit.\n\n"
                "You need SCAPS-1D installed (free from Ghent University). SCAPSync finds it by itself.\n")
    zip_base = os.path.join(HERE, "dist", f"SCAPSync-{__version__}-windows")
    shutil.make_archive(zip_base, "zip", os.path.join(HERE, "dist"), "SCAPSync")
    size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(dist) for f in fs)
    print(f"BUILD OK in {time.time() - t0:.0f} s: {dist} ({size / 1e6:.0f} MB), "
          f"zip {os.path.getsize(zip_base + '.zip') / 1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
