#!/usr/bin/env python3
import collections
from io import StringIO
import json
from math import e, log
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys


def create_dir_if_needed(path: str):
    """
    Creates a directory at path if that directory does not exist yet
    @param path: path to directory
    @return:
    """
    try:
        os.makedirs(path)
    except OSError as e:
        pass


def read_json(path):
    with open(path) as file:
        return json.load(file)


def rename_old_tb_true_tree(root_dir):
    for exp_id in os.listdir(root_dir):
        exp_dir = os.path.join(root_dir, exp_id)
        if not os.path.isdir(exp_dir):
            continue
        for msa_id in os.listdir(exp_dir):
            msa_dir = os.path.join(exp_dir, msa_id, "default")

            if not os.path.isdir(msa_dir):
                continue

            for fn in os.listdir(msa_dir):
                if fn.startswith("true.raxml"):
                    old_path = os.path.join(msa_dir, fn)
                    new_path = os.path.join(msa_dir, fn.replace("true.", "bigraxml_eval."))
                    os.rename(old_path, new_path)


def reset_all_rule(root_dir):
    all_rule_fn = [
        "rf.raxml.rfDistances",
        "top_test.iqtree",
        "quartet_dists.txt",
        "concat.bestTrees",
        "llh_diffs",
        "bionj_eval.raxml.bestTree",
        # "pars_eval.raxml.bestTree"
    ]

    for exp_id in os.listdir(root_dir):
        exp_dir = os.path.join(root_dir, exp_id)
        if not os.path.isdir(exp_dir):
            continue
        for msa_id in os.listdir(exp_dir):
            msa_dir = os.path.join(exp_dir, msa_id, "default")

            for fn in all_rule_fn:
                del_path = os.path.join(msa_dir, fn)
                if os.path.isfile(del_path):
                    os.remove(del_path)


def collect_run_times(root_dir):
    file_names = [
        "raxml_eval.raxml.log", "iqt2.log", "ft2.log", "pars.raxml.log", "bionj.log", "bigraxml_eval.raxml.log"
    ]
    value_dct = collections.defaultdict(lambda: [])
    for exp_id in os.listdir(root_dir):
        exp_dir = os.path.join(root_dir, exp_id)
        if not os.path.isdir(exp_dir):
            continue
        for msa_id in os.listdir(exp_dir):
            msa_dir = os.path.join(exp_dir, msa_id, "default")
            if not os.path.isdir(msa_dir):
                continue

            for fn in file_names:
                log_path = os.path.join(msa_dir, fn)
                time = -1

                if os.path.isfile(log_path):
                    with open(log_path) as file:
                        for line in file:
                            line = line.strip()
                            if line.startswith("Elapsed time: "):
                                time = float(line.split()[2])
                            if line.startswith("Total wall-clock time used: "):
                                time = float(line.split()[4])
                            if line.startswith("Total time: "):
                                time = float(line.split()[2])
                value_dct[fn].append(time)

    ref_idx = 0
    relative_times = collections.defaultdict(lambda: [])
    for i in range(len(value_dct[file_names[0]])):
        ref_val = value_dct[file_names[ref_idx]][i]
        for key in file_names:
            val = value_dct[key][i]
            if val != -1:
                relative_times[key].append(val / ref_val)

    for key in file_names:
        l = relative_times[key]
        if len(l) > 1:
            print(f"{key}: {statistics.mean(l)}")



if __name__ == "__main__":
    globals()[sys.argv[1]](*sys.argv[2:])
