from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Optional, List

import click
import typer
from rich.logging import RichHandler
from typer.main import get_command

from .aggregate import aggregate_run
from .runner import run_workflow
from .visualize.dash_app import run_server
from phylosmew.legacy import scripts

# -----------------------------------------------------------------------------
# Global flags (parsed from argv before Typer runs)
# -----------------------------------------------------------------------------
@dataclass
class _GlobalFlags:
    plain: bool = False


def _consume_global_flags_from_argv(argv: List[str]) -> _GlobalFlags:
    """
    Find and remove --plain/--no-rich and --log-file (with or without '=') anywhere
    in argv. Return the parsed values. argv is modified in-place.
    """
    gf = _GlobalFlags()
    i = 1  # skip program name
    while i < len(argv):
        tok = argv[i]
        if tok in ("--plain", "--no-rich"):
            gf.plain = True
            del argv[i]
            continue
        i += 1
    return gf


# -----------------------------------------------------------------------------
# Typer app
# -----------------------------------------------------------------------------
app = typer.Typer(add_help_option=True, no_args_is_help=True)

def _print_full_help_and_exit() -> None:
    click_cmd = get_command(app)
    with click.Context(click_cmd) as c:
        typer.echo(click_cmd.get_help(c))
        for name, cmd in click_cmd.commands.items():
            typer.echo(f"\n# {name}\n")
            with click.Context(cmd) as sc:
                typer.echo(cmd.get_help(sc))
    raise typer.Exit()

@app.callback(invoke_without_command=True)
def _main_callback(
    help_all: bool = typer.Option(
        False, "--help-all", is_eager=True, help="Show full help for all subcommands and exit."
    ),
    plain: bool = typer.Option(
        False, "--plain", help="Removes the bling-bling from the terminal output."
    ),
):
    if help_all:
        _print_full_help_and_exit()

@app.command("help")
def help_command(command: Optional[str] = typer.Argument(None, help="Command to show help for")):
    click_cmd = get_command(app)
    if not command:
        with click.Context(click_cmd) as c:
            typer.echo(click_cmd.get_help(c))
        raise typer.Exit()
    sub = click_cmd.commands.get(command)
    if not sub:
        raise typer.BadParameter(f"Unknown command: {command}")
    with click.Context(sub) as sc:
        typer.echo(sub.get_help(sc))

# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------
@app.command(help="Creates the helper files 'config.yaml' and 'inference_tools.py' in the current directory.")
def init(
    path: str = typer.Option("config.yaml", "--config", "-c", help="Where to write the config"),
    force: bool = typer.Option(False, "--force", help="Overwrite if exists"),
):
    target = Path(path)
    if target.exists() and not force:
        raise typer.Exit(f"Refusing to overwrite existing {target}. Use --force to replace.")
    template = files("phylosmew.templates").joinpath("config.default.yaml").read_text()
    target.write_text(template)

    # Also scaffold a local inference_tools.py (user-editable)
    tools_dst = target.parent / "inference_tools.py"
    if not tools_dst.exists():
        tools_src = files("phylosmew.templates").joinpath("inference_tools.user.py").read_text()
        tools_dst.write_text(tools_src)
        typer.echo(f"Wrote user tools template to {tools_dst}")
    typer.echo(f"Wrote default config to {target}")


@app.command(help="Runs the snakemake pipeline.")
def run(
    config: str = typer.Option("config.yaml", "--config", "-c"),
    outdir: Optional[str] = typer.Option(None, "--out", "-o"),
    cores: int = typer.Option(1, "--cores", "-j"),
    tools: str = typer.Option(None, "--tools", help="Code file containing the implementation to execute the phylogenetic inference tools."),
    extra: Optional[List[str]] = typer.Argument(None, help="Extra args forwarded to Snakemake"),
):
    run_workflow(config, outdir, cores, extra or [], tools_path=tools)

@app.command(help="Collects the results into a CSV file.")
def aggregate(
    run_dir: str = typer.Option(..., "--root-dir", help="Directory for a finished run (e.g., out/smew_test)"),
    out: Optional[str] = typer.Option(None, "--out", help="Where to write the aggregated CSV"),
    no_time: bool = typer.Option(False, "--no-time", help="Skip runtime parsing to speed up aggregation"),
):
    if not out:
        base = os.path.basename(run_dir) or os.path.basename(os.path.dirname(run_dir))
        out = f"{base}.csv"
    aggregate_run(run_dir, out)

@app.command(help="Runs a Dash app to visualize the results stored in the CSV file (generate it using 'phylosmew aggregate')")
def visualize(
    csv: str = typer.Option(..., "--csv", help="Aggregated results CSV"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8050, "--port"),
):
    run_server(csv, host, port)

# README helpers (wired to legacy.scripts)
@app.command("copy-msas", help="Tries to copy and rename (and clean) the MSAs from source to destination according to our expected format.")
def copy_msas_cmd(
    source_dir: str = typer.Option(..., "--source-dir", "-s", help="Directory with your MSAs"),
    clean: bool = typer.Option(True, "--clean/--no-clean", help="Clean MSAs (remove empties/dups/etc.)"),
    dest_dir: Optional[str] = typer.Option(None, "--dest-dir", "-d", help="Destination (defaults to ./out/<source_basename>)"),
    #suffix: Optional[str] = typer.Option(None, "--suffix", help="Optional suffix added to destination directory name"),
):
    bn = os.path.basename(source_dir) if os.path.basename(source_dir) else os.path.basename(os.path.dirname(source_dir))
    if dest_dir:
        dest_dir = os.path.join(dest_dir, bn)
    else:
        dest_dir = os.path.join("out", bn)

    scripts.copy_msas(source_dir, "1" if clean else "0", dest_dir or "")
    typer.echo("MSAs copied. Created selection files (representatives.json / selected_datasets.json).")


@app.command("make-plots", help="Creates some simple plots.")
def clean_cmd(
    root_dir: str = typer.Option(..., "--root-dir", help="Root containing dataset subdirs with assembled_sequences.fasta"),
):
    scripts.create_plots(root_dir)
    typer.echo("Making plots.")


@app.command("reset-evaluation", help="Resets the evaluation. This allows adding new inference tools to the pipeline and compare their results with the existing ones.")
def reset_evaluation_cmd(
    root_dir: str = typer.Option(..., "--root-dir", help="Root containing dataset subdirs"),
):
    scripts.reset_evaluation(root_dir)
    typer.echo("Removed evaluation artifacts (rf/ntd/llh/true/… files).")


@app.command("create-repr-files", help="Creates creates a list of present datasets to pass to snakemake (usually not needed).")
def create_repr_files_cmd(
    root_dir: str = typer.Option(..., "--root-dir", help="out/<dsc> directory with dataset subfolders"),
):
    scripts.create_repr_files(root_dir)
    typer.echo("Wrote representatives.json and selected_datasets.json.")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main():
    # Parse & remove global flags anywhere in argv
    gf = _consume_global_flags_from_argv(sys.argv)
    app()

if __name__ == "__main__":
    main()
