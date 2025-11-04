# src/phylosmew/aggregate.py
from pathlib import Path
import pandas as pd

from phylosmew.legacy.scripts import make_csv

def aggregate_run(run_dir: str, out_csv: str):
    make_csv(run_dir, out_csv)

