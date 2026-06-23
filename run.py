"""
Auto-installer & Launcher for AI Resume Screening System
Run: python run.py
"""
import subprocess, sys, os

PACKAGES = [
    "streamlit", "scikit-learn", "pandas", "numpy",
    "matplotlib", "seaborn", "pymupdf", "pdfplumber"
]

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

print("=" * 55)
print("  AI Resume Screening System - Auto Launcher")
print("=" * 55)

print("\n[1/2] Checking dependencies...")
for pkg in PACKAGES:
    try:
        __import__(pkg.replace("-","_").split("[")[0])
        print(f"  ✓ {pkg}")
    except ImportError:
        print(f"  ↓ Installing {pkg}...")
        install(pkg)
        print(f"  ✓ {pkg} installed")

print("\n[2/2] Launching app...")
print("  → Open browser at: http://localhost:8501\n")

os.system(f"{sys.executable} -m streamlit run app.py --server.port 8501")
