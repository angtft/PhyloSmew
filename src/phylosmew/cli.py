# src/phylosmew/cli.py
from __future__ import annotations

import os.path
from pathlib import Path
from importlib.resources import files
import typer
from .runner import run_workflow
from .aggregate import aggregate_run
from .visualize.dash_app import run_server

app = typer.Typer(add_help_option=True)

@app.command()
def init(
    path: str = typer.Option("config.yaml", "--config", "-c", help="Where to write the config"),
    force: bool = typer.Option(False, "--force", help="Overwrite if exists")
):
    target = Path(path)
    if target.exists() and not force:
        raise typer.Exit(f"Refusing to overwrite existing {target}. Use --force to replace.")
    template = files("phylosmew.templates").joinpath("config.default.yaml").read_text()
    target.write_text(template)
    typer.echo(f"Wrote default config to {target}")

@app.command()
def run(
    config: str = typer.Option("config.yaml", "--config", "-c"),
    outdir: str = typer.Option(None, "--out", "-o"),
    cores: int = typer.Option(1, "--cores", "-j"),
    extra: list[str] = typer.Argument(None, help="Extra args forwarded to Snakemake")
):
    from .runner import run_workflow
    run_workflow(config, outdir, cores, extra or [])

@app.command()
def aggregate(
    run_dir: str = typer.Option(..., "--run-dir", help="Directory for a finished run (e.g., out/smew_test)"),
    out: str = typer.Option(None, "--out", help="Where to write the aggregated CSV")
):
    if not out:
        dsc_name = os.path.basename(run_dir) if os.path.basename(run_dir) else os.path.basename(os.path.dirname(run_dir))
        out = f"{dsc_name}.csv"
    aggregate_run(run_dir, out)

@app.command()
def visualize(
    csv: str = typer.Option(..., "--csv", help="Aggregated results CSV"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8050, "--port")
):
    run_server(csv, host, port)

def main():
    app()

if __name__ == "__main__":
    main()
