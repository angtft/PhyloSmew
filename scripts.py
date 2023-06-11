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

sys.path.insert(1, os.path.join("libs", "RAxMLGroveScripts"))
sys.path.insert(1, os.path.join("libs", "PyPythia"))

import inference_tools
import libs.RAxMLGroveScripts.org_script as rgs
from libs.PyPythia.prediction_no_install import predict_difficulty as pythia_predict_difficulty
from util import *


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


def delete_unfinished(root_dir):
    for exp_id in os.listdir(root_dir):
        exp_dir = os.path.join(root_dir, exp_id)
        if not os.path.isdir(exp_dir):
            continue

        req_path = os.path.join(exp_dir, "msa_0", "default", "rf.raxml.rfDistances")
        if not os.path.isfile(req_path):
            print(f"{req_path} not found! removing exp_dir")
            shutil.rmtree(exp_dir)


def rename_experiments(root_dir):
    exp_ids = list(os.listdir(root_dir))
    for exp_id in exp_ids:
        exp_dir = os.path.join(root_dir, exp_id)
        if not os.path.isdir(exp_dir):
            continue

        tree_dct_path = os.path.join(exp_dir, "msa_0", "default", "tree_dict.json")
        if not os.path.isfile(tree_dct_path):
            print(f"{tree_dct_path} not found")
            continue

        with open(tree_dct_path) as file:
            tree_dct = json.load(file)[0]

        tree_id = tree_dct["TREE_ID"]
        new_dir = os.path.join(root_dir, tree_id)

        os.rename(exp_dir, new_dir)


def select_representatives():
    # TODO: this is temporary!
    source = "TB"
    query = "DATA_TYPE = 'AA'"
    sort_by = "PATTERNS_BY_TAXA"


    command = [
        "find",
        "-q", query,
        "--list"
    ]
    if source == "RGS":
        db_path = "latest.db"
        if os.path.isfile(rgs_db_path):
            db_path = rgs_db_path
        print(db_path)
        command.extend([
            "-n", db_path
        ])
    elif source == "TB" or source == "RGS_TB":
        db_path = "tb_all.db"
        if os.path.isfile(rgs_db_path):
            db_path = rgs_db_path
        command.extend([
            "-n", db_path,
        ])
    else:
        raise ValueError(f"unknown data source: {source}")

    grouped_results, _ = rgs.main(command)
    sorted_results = [grouped_results[tree_id][0] for tree_id in grouped_results]
    if sort_by in sorted_results[0]:
        sorted_results.sort(key=lambda element: element[sort_by])
    else:
        if sort_by == "PATTERNS_BY_TAXA":
            sorted_results.sort(key=lambda x: x["OVERALL_NUM_PATTERNS"] / x["NUM_TAXA"])
        else:
            raise ValueError(f"unknown sorting criterion: {sort_by}")

    ignored_msas = {}
    #if filter_file:
    #    ignored_msas = read_filter_file(filter_file)
    sorted_results = list(filter(lambda x: x["TREE_ID"] not in ignored_msas, sorted_results))

    result_dict = {}
    l = len(sorted_results)
    for i in range(num_splits):
        temp_range = sorted_results[int(l * i / num_splits): int(l * (i + 1) / num_splits)]
        result_dict[i] = temp_range

    # in order to avoid any floating point stuff in bucket creation (leading to duplicated or empty buckets)
    print(num_splits, len(sorted_results))
    if num_splits == len(sorted_results):
        for i in range(num_splits):
            result_dict[i] = [sorted_results[i]]

    if any([not result_dict[i] for i in result_dict]):
        print(f"there are empty representatives buckets! aborting")
        return

    for i in range(1, num_splits-1):
        buckets = (result_dict[i-1], result_dict[i], result_dict[i+1])
        for j in range(len(buckets)-1):
            b1 = buckets[j]
            b2 = buckets[j+1]
            for t1 in b1:
                for t2 in b2:
                    if t1["TREE_ID"] == t2["TREE_ID"]:
                        print("found same tree in two representative buckets! aborting")
                        return

    for key in result_dict:
        repr_set = result_dict[key]
        temp_repr = random.choice(repr_set)
        tree_id = temp_repr["TREE_ID"]

        exp_dir = os.path.join(os.path.dirname(out_file),f"{key}")
        create_dir_if_needed(exp_dir)
        tree_id_file = os.path.join(exp_dir,"tree_id")
        with open(tree_id_file, "w+") as file:
            file.write(f"{tree_id}\n")

    with open(out_file,"w+") as file:
        json.dump(result_dict,file,indent=4)

def download_datasets(root_dir, query, db_name, ref_dir=""):
    if db_name != "tb_all.db":
        raise ValueError(f"only TB database supported right now!")

    ref_ids = {}
    if os.path.isdir(ref_dir):
        for ref_id in os.listdir(ref_dir):
            ref_ids[ref_id] = os.path.join(ref_dir, ref_id)

    command_find = [
        "find",
        "-q", query,
        "-n", db_name,
        "-o", root_dir,
        "--list"
    ]

    grouped_results, _ = rgs.main(command_find)

    for tree_id in list(grouped_results.keys()):
        if tree_id in ref_ids:
            print(f"{tree_id} already available!")
            continue

        temp_query = f"TREE_ID = '{tree_id}'"
        temp_command = [
            "generate",
            "-q", temp_query,
            "-n", db_name,
            "--generator", "alisim",
            "--no-simulation",
            "-o", root_dir
        ]
        rgs.main(temp_command)



if __name__ == "__main__":
    globals()[sys.argv[1]](*sys.argv[2:])
