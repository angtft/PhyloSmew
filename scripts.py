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
import yaml

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


def reset_all_rule(root_dir, exp_ids):
    all_rule_fn = [
        "rf.raxml.rfDistances",
        "top_test.iqtree",
        "quartet_dists.txt",
        "concat.bestTrees",
        "llh_diffs",
        # "bionj_eval.raxml.bestTree",
        # "pars_eval.raxml.bestTree"
    ]

    # for exp_id in os.listdir(root_dir): # TODO: this one is kinda dangerous...
    for exp_id in exp_ids.split(","):
        exp_dir = os.path.join(root_dir, exp_id)
        if not os.path.isdir(exp_dir):
            continue

        for msa_id in os.listdir(exp_dir):
            msa_dir = os.path.join(exp_dir, msa_id, "default")

            for fn in all_rule_fn:
                del_path = os.path.join(msa_dir, fn)
                if os.path.isfile(del_path):
                    os.remove(del_path)

        # cover new directory structure
        for fn in all_rule_fn:
            del_path = os.path.join(exp_dir, fn)
            if os.path.isfile(del_path):
                os.remove(del_path)


def collect_run_times(root_dir):
    file_names = [
        "raxml.raxml.log", "iqt2.log", "ft2.log", "pars.raxml.log", "bionj.log", "bigraxml.raxml.log"
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


def download_datasets_from_config(root_dir, ref_dir=""):
    with open("config.yaml") as file:
        config_dct = yaml.safe_load(file)

    dsc_name = config_dct["data_sets"]["used_dsc"]
    query = config_dct["data_sets"][dsc_name]["query"]
    db_name = config_dct["data_sets"][dsc_name]["db"]

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

        new_tree_dir = os.path.join(root_dir, tree_id)
        if os.path.isdir(new_tree_dir):
            print(f"{new_tree_dir} already exists!")
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

    repr_path = os.path.join(root_dir, "representatives.json")
    if os.path.isfile(repr_path):
        raise ValueError(f"{repr_path} already exists!")

    repr_dct = {}
    for i, key in enumerate(grouped_results):
        repr_dct[i] = [grouped_results[key][0]]
    with open(repr_path, "w+") as file:
        json.dump(repr_dct, file, indent=4)

    sel_path = os.path.join(root_dir, "selected_datasets.json")
    sel_dct = {}
    if not os.path.isfile(sel_path):
        with open(repr_path) as repr_file:
            representatives = json.load(repr_file)
        for data_set_num in representatives:
            temp_repr = random.choice(representatives[data_set_num])
            tree_id = temp_repr["TREE_ID"]
            sel_dct[data_set_num] = tree_id

        with open(sel_path, "w+") as sel_file:
            json.dump(sel_dct, sel_file, indent=4)
    else:
        raise ValueError(f"{sel_path} already exists!")


def test_func():
    with open("config.yaml") as file:
        dct = yaml.safe_load(file)
    print(dct)


def copy_old_version_datasets(src_dir, dst_dir):
    sel_dct = {}
    repr_dct = {}

    for exp_id in os.listdir(src_dir):
        src_exp_dir = os.path.join(src_dir, exp_id, "msa_0", "default")
        tree_dct_path = os.path.join(src_exp_dir, "tree_dict.json")
        if not os.path.isfile(tree_dct_path):
            continue

        with open(tree_dct_path) as file:
            tree_dct = json.load(file)
        tree_id = tree_dct[0]["TREE_ID"]

        sel_dct[exp_id] = tree_id
        repr_dct[exp_id] = [tree_dct[0]]
        dst_exp_dir = os.path.join(dst_dir, tree_id)

        if os.path.isdir(dst_exp_dir):
            print(f"{dst_exp_dir} already exists!")
            continue

        print(f"copying {src_exp_dir}")
        shutil.copytree(src_exp_dir, dst_exp_dir)

    with open(os.path.join(dst_dir, "representatives.json"), "w+") as file:
        json.dump(repr_dct, file, indent=4)

    with open(os.path.join(dst_dir, "selected_datasets.json"), "w+") as file:
        json.dump(sel_dct, file, indent=4)


def copy_old_version_msas(src_dir, dst_dir):
    to_copy = ["assembled_sequences.fasta", "sim_partitions.txt", "tree_dict.json", "log_0.txt", "model_0.txt",
               "iqt.pr_ab_matrix", "tree_best.newick", "difficulty"]
    sel_dct = {}

    for exp_id in os.listdir(src_dir):
        src_exp_dir = os.path.join(src_dir, exp_id, "msa_0", "default")
        tree_dct_path = os.path.join(src_exp_dir, "tree_dict.json")
        if not os.path.isfile(tree_dct_path):
            continue

        with open(tree_dct_path) as file:
            tree_dct = json.load(file)
        tree_id = tree_dct[0]["TREE_ID"]

        sel_dct[exp_id] = tree_id

        dst_exp_dir = os.path.join(dst_dir, tree_id)
        if os.path.isdir(dst_exp_dir):
            print(f"{dst_exp_dir} already exists!")
            continue

        print(f"copying {src_exp_dir}")
        create_dir_if_needed(dst_exp_dir)
        for fn in to_copy:
            src_fp = os.path.join(src_exp_dir, fn)
            dst_fp = os.path.join(dst_exp_dir, fn)

            if not os.path.isfile(src_fp):
                print(f"{src_fp} not found")
                continue
            shutil.copy2(src_fp, dst_fp)

    src_repr_path = os.path.join(src_dir, "representatives_all.txt")
    dst_repr_path = os.path.join(dst_dir, "representatives.json")

    shutil.copy2(src_repr_path, dst_repr_path)

    with open(os.path.join(dst_dir, "selected_datasets.json"), "w+") as file:
        json.dump(sel_dct, file, indent=4)


def copy_old_version_finished(src_dir, dst_dir):
    to_include = ["tree_dict.json", "orig_msa.fasta", "sim_msa_differences", "assembled_sequences.fasta", "difficulty",]
    to_include_prefix = ["iqt2.", "iqt2_eval.raxml.",  "pars_eval.", "bigraxml."]
    to_rename = {
        "raxml_eval": "raxml",
        "ft2_bin_eval": "ft2_eval",
        "mf.ft2.bestTree": "ft2.mf.tree",
        "bin.ft2.bestTree": "ft2.bin.tree",
    }

    sel_dct = {}
    repr_dct = {}
    counter = 0

    for exp_id in os.listdir(src_dir):
        src_exp_dir = os.path.join(src_dir, exp_id, "msa_0", "default")
        tree_dct_path = os.path.join(src_exp_dir, "tree_dict.json")
        if not os.path.isfile(tree_dct_path):
            continue

        if counter % 100 == 0:
            print(counter)
        counter += 1

        with open(tree_dct_path) as file:
            tree_dct = json.load(file)
        tree_id = tree_dct[0]["TREE_ID"]

        sel_dct[exp_id] = tree_id
        repr_dct[exp_id] = [tree_dct[0]]
        dst_exp_dir = os.path.join(dst_dir, tree_id)

        create_dir_if_needed(dst_exp_dir)

        for fn in os.listdir(src_exp_dir):
            src_path = os.path.join(src_exp_dir, fn)
            if fn in to_include or any(fn.startswith(prefix) for prefix in to_include_prefix):
                shutil.copy2(src_path, dst_exp_dir)
            else:
                for key in to_rename:
                    if fn.startswith(key):
                        dst_path = os.path.join(dst_exp_dir, fn.replace(key, f"{to_rename[key]}"))
                        shutil.copy2(src_path, dst_path)
                        break

    with open(os.path.join(dst_dir, "representatives.json"), "w+") as file:
        json.dump(repr_dct, file, indent=4)

    with open(os.path.join(dst_dir, "selected_datasets.json"), "w+") as file:
        json.dump(sel_dct, file, indent=4)


def create_repr_files(src_dir):
    sel_dct = {}
    repr_dct = {}
    counter = 0

    for exp_id in os.listdir(src_dir):
        src_exp_dir = os.path.join(src_dir, exp_id)
        tree_dct_path = os.path.join(src_exp_dir, "tree_dict.json")
        if not os.path.isfile(tree_dct_path):
            continue

        if counter % 100 == 0:
            print(counter)
        counter += 1

        with open(tree_dct_path) as file:
            tree_dct = json.load(file)
        tree_id = tree_dct[0]["TREE_ID"]

        sel_dct[counter] = tree_id
        repr_dct[counter] = [tree_dct[0]]

    dst_dir = src_dir
    with open(os.path.join(dst_dir, "representatives.json"), "w+") as file:
        json.dump(repr_dct, file, indent=4)

    with open(os.path.join(dst_dir, "selected_datasets.json"), "w+") as file:
        json.dump(sel_dct, file, indent=4)


def remove_files_from_exps(root_dir, fn_list, prefix_list):
    # "1" rf.raxml.,top_test,quartet_,llh_,tree_best,bigraxml,true

    for counter, exp_id in enumerate(os.listdir(root_dir)):
        exp_dir = os.path.join(root_dir, exp_id, "msa_0", "default")
        if not os.path.isdir(exp_dir):
            exp_dir = os.path.join(root_dir, exp_id)
            if not os.path.isdir(exp_dir):
                continue

        for fn in fn_list.split(","):
            file_path = os.path.join(exp_dir, fn)
            if os.path.isfile(file_path):
                os.remove(file_path)
            else:
                print(f"{file_path} not found!")

        for fn in os.listdir(exp_dir):
            for prefix in prefix_list.split(","):
                if fn.startswith(prefix):
                    file_path = os.path.join(exp_dir, fn)
                    if os.path.isfile(file_path):
                        os.remove(file_path)

        if counter % 100 == 0:
            print(counter)


def find_broken_iqt2_trees(root_dir):
    s = "ERROR: Tree taxa and alignment sequence do not match"

    exp_ids = list(os.listdir(root_dir))
    for exp_id in exp_ids:
        exp_dir = os.path.join(root_dir, exp_id)
        if not os.path.isdir(exp_dir):
            continue

        log_path = os.path.join(exp_dir, "iqt2.log")
        if not os.path.isfile(log_path):
            log_path = os.path.join(exp_dir, "msa_0", "default", "iqt2.log")
            if not os.path.isfile(log_path):
                print(f"{log_path} not found!")
                continue

        with open(log_path) as file:
            for line in file:
                line = line.strip()
                if s in line:
                    print(f"{exp_dir}: {line}")


if __name__ == "__main__":
    globals()[sys.argv[1]](*sys.argv[2:])
