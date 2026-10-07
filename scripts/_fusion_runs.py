
import subprocess, sys
PY = r"C:\Dev\AIEngineeringProjects\Grey-Resolve\.venv\Scripts\python.exe"
BASE = [PY, "benchmarks/evaluate_fusion.py",
        "--data-root", r"C:\Users\pchri\JHUAIEngineering\705.603-Creating_AI-Enabled_Systems\Modules\Mod8-VectorSearchSystems-3\705-603-fall-2026-ironclad-pcschmidt\ironclad\storage\multi_image_gallery",
        "--max-identities", "100", "--limit", "2",
        "--n-identities", "30", "--media-per-identity", "2", "--seed", "42"]
RUNS = [
    ["--query-degradation", "downsample", "0.25"],
    ["--query-degradation", "gaussian_blur", "3.0"],
    ["--query-degradation", "downsample", "0.1"],
]
for run in RUNS:
    print("=" * 30, "RUN:", run, flush=True)
    r = subprocess.run(BASE + run)
    print("exit:", r.returncode, flush=True)
    if r.returncode != 0:
        sys.exit(r.returncode)
print("ALL DONE")
