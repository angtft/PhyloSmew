# src/phylosmew/runner.py (core bits)
import os
import subprocess
import yaml
from pathlib import Path
from importlib.resources import files

def run_workflow(config, outdir, cores, snakemake_extra, tools_path=None):
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

    # Allow users to override/extend inference tools:
    # Priority: explicit path in config.plugins.inference_tools -> ./inference_tools.py -> none
    env = os.environ.copy()

    if not tools_path:
        try:
            tools_path = cfg.get("plugins", {}).get("inference_tools", None)
        except Exception:
            tools_path = None

    if tools_path:
        p = Path(tools_path)
    else:
        p = Path(config).parent / "inference_tools.py"
    if p.is_file():
        env["PHYLOSMEW_TOOLS_PATH"] = str(p.resolve())
    subprocess.run(cmd, check=True, env=env)
