# src/phylosmew/runner.py (core bits)
import subprocess, yaml
from pathlib import Path
from importlib.resources import files

def run_workflow(config, outdir, cores, snakemake_extra):
    cfg_path = Path(config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text()) or {}

    snakefile = files("phylosmew.workflow").joinpath("Snakefile")
    cmd = [
        "snakemake",
        "--snakefile", str(snakefile),
        "--configfile", str(cfg_path),
        #"--directory", str(Path(outdir).resolve()),
        "--cores", str(cores),
        "--rerun-incomplete",
    ] + snakemake_extra
    if outdir:
        cmd += ["--config", f"out_dir={outdir}"]

    subprocess.run(cmd, check=True)
