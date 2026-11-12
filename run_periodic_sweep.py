"""Reproduce the periodic sine-game comparison at three resolutions."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--K", type=int, default=20000)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--nsub", type=int, default=800)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    script = here / "periodic_diffusion_grid.py"
    out = (args.outdir or here / "periodic_grid_histories").resolve()
    out.mkdir(parents=True, exist_ok=True)
    logs = out / "logs"
    logs.mkdir(exist_ok=True)
    env = os.environ.copy()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    runs = []
    for nx, nt in ((64, 25), (128, 50), (256, 100)):
        for kappa in (1, 0):
            for metric in ("S", "E"):
                name = "S3" if metric == "S" else "E"
                tag = f"periodic_k{kappa}_{name}_Nx{nx}_Nt{nt}"
                if any(out.glob(f"{tag}_*")):
                    raise SystemExit(f"Results already exist for {tag}; choose a fresh --outdir.")
                command = [sys.executable, str(script), "--Nx", str(nx), "--Nt", str(nt),
                           "--K", str(args.K), "--metric", metric, "--step",
                           "0.25" if metric == "S" else "auto", "--kappa", str(kappa),
                           "--nsub", str(args.nsub), "--ncheck", "18", "--outdir",
                           str(out), "--tag", tag]
                runs.append({"tag": tag, "command": command})
    manifest = {"source_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
                "K": args.K, "workers": args.jobs, "runs": runs}
    manifest_path = out / "rerun_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    def execute(item):
        started = time.perf_counter()
        with (logs / f'{item["tag"]}.log').open("w") as handle:
            result = subprocess.run(item["command"], cwd=here, env=env,
                                    stdout=handle, stderr=subprocess.STDOUT)
        record = {"tag": item["tag"], "exit_code": result.returncode,
                  "seconds": round(time.perf_counter() - started, 2)}
        print(json.dumps(record), flush=True)
        return record

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        manifest["results"] = list(executor.map(execute, runs))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return int(any(result["exit_code"] for result in manifest["results"]))


if __name__ == "__main__":
    raise SystemExit(main())
