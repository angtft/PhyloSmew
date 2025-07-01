#!/usr/bin/env python3
import argparse
import collections
from distutils.dir_util import copy_tree
from io import StringIO
import itertools
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

from Bio import Phylo, SeqIO, Seq, SeqRecord
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import msa_parser

sys.path.insert(1, os.path.join("libs", "RAxMLGroveScripts"))
sys.path.insert(1, os.path.join("libs", "PyPythia"))

import inference_tools
import libs.RAxMLGroveScripts.org_script as rgs
from libs.PyPythia.prediction_no_install import predict_difficulty as pythia_predict_difficulty
from util import *

pythia_predictor_path = os.path.join("libs", "PyPythia", "pypythia", "predictors", "predictor_sklearn_rf_v0.0.1.pckl")

CAT_NAME_DICT = {
    "rf": "RF distance",
    "llh": "LnL difference",
    "ntd": "Normalized Tree Distance",
    "llh_percent": "LnL difference (percent)",
    "quartet": "Quartet distance",
    "top": "passed statistical tests",
    "additional_rf": "RF distance",
    "additional_llh_percent": "LnL difference (percent)",
    "_additional_llh_percent_difficulty_single": "LnL difference (percent)",
    "consel": "Proportion of datasets passing the AU test"
}
TOOL_NAME_DICT = {
    "true": "true",
    "pars": "parsimony",
    "rand": "random",
    "raxml": "RAxML-NG",
    "iqt2": "IQ-TREE2",
    "ft2": "FastTree2",
    "bigraxml": "RAxML-NG-100",
    "bionj": "BIONJ"
}


display_names = {"raxml": "RAxML-NG", "iqt2": "IQ-TREE2", "ft2": "FastTree2", "pars": "pars", "bionj": "BIONJ"}
display_names_true = {"true": "true", "raxml": "RAxML-NG", "iqt2": "IQ-TREE2", "ft2": "FastTree2", "pars": "pars",
                      "bionj": "BIONJ"}


def adjust_names_true(df):
    present_names = []
    for key in df:
        if key.startswith("llh_"):
            present_names.append(key.replace("llh_", ""))
    for name in present_names:
        if name not in display_names_true:
            display_names_true[name] = name
    return present_names


def adjust_names(df):
    present_names = []
    for key in df:
        if key.startswith("llh_") and key != "llh_true":
            present_names.append(key.replace("llh_", ""))
    for name in present_names:
        if name not in display_names:
            display_names[name] = name
        if name not in display_names_true:
            display_names_true[name] = name
    return present_names


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


def collect_run_times(root_dir, file_names):
    #file_names = [
    #    "raxml.raxml.log", "iqt2.log", "ft2.log", "pars.raxml.log", "bionj.log", "bigraxml.raxml.log", "parsconst1.raxml.log", "bpc1.raxml.log", "raxad.raxml.log", "bpc1f.raxml.log", "raxml1.raxml.log"
    #]
    value_dct = {}
    for fn in file_names:
        tool_name = fn.split(".")[0]
        log_path = os.path.join(root_dir, fn)
        if not os.path.isfile(log_path) and fn == "raxml.raxml.log":
            log_path = os.path.join(root_dir, "raxml_eval.raxml.log")

        time = -1

        if os.path.isfile(log_path):
            with open(log_path) as file:
                for line in file:
                    if "raxml-ng --evaluate" in line and fn != "bigraxml.raxml.log":
                        print("EVAL in TIME MEASURE!!!")

                    line = line.strip()
                    if line.startswith("Elapsed time: "):
                        time = float(line.split()[2])
                    if line.startswith("Total wall-clock time used: "):
                        time = float(line.split()[4])
                    if line.startswith("Total time: "):
                        time = float(line.split()[2])
        else:
            if fn != "bigraxml.raxml.log":
                print(f"{log_path} not found for time collection!")
        value_dct[f"abs_time_{tool_name}"] = time

    ref_key = "abs_time_raxml"
    ref_val = 1     #value_dct[ref_key] # TODO: fix!
    relative_times = {}

    for fn in file_names:
        tool_name = fn.split(".")[0]
        relative_times[f"rel_time_{tool_name}"] = value_dct[f"abs_time_{tool_name}"] / ref_val

    return value_dct, relative_times


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


def guess_data_type_from_model(model):
    """
    Guesses the datatype based on the model string. TODO: the list might be incomplete!
    Args:
        model: model string

    Returns:
        guessed data type
    """

    # stolen from RGS stuff
    sub_models = {
        "DNA": ['JC', 'K80', 'F81', 'HKY', 'TN93ef', 'TN93', 'K81', 'K81uf', 'TPM2', 'TPM2uf', 'TPM3', 'TPM3uf', 'TIM1',
                'TIM1uf', 'TIM2', 'TIM2uf', 'TIM3', 'TIM3uf', 'TVMef', 'TVM', 'SYM', 'GTR'],
        "AA": ['Blosum62', 'cpREV', 'Dayhoff', 'DCMut', 'DEN', 'FLU', 'HIVb', 'HIVw', 'JTT', 'JTT-DCMut', 'LG', 'mtART',
               'mtMAM', 'mtREV', 'mtZOA', 'PMB', 'rtREV', 'stmtREV', 'VT', 'WAG', 'LG4M', 'LG4X', 'PROTGTR'],
        "BIN": ["BIN"],
        "UNPHASED_DIPLOID_GENOTYPES": ['GTJC', 'GTHKY4', 'GTGTR4', 'GTGTR'],
        "MULTISTATE": ["MULTI"]
        # "USERDEFINED": ""     # TODO
    }
    model_str = model.split("+")[0].lower()
    for key in sub_models:
        sub_models[key] = [s.lower() for s in sub_models[key]]

    for dt in sub_models:
        if model_str in sub_models[dt]:
            return dt
    raise ValueError(f"MSA data type for model '{model_str}' could not be guessed!")


def read_sim_part_file(part_path:str, format_type="standard"):
    """
    Expects partition file in format
        data_type, sub_model, part_name = start_index1-end_index1, ...
    e.g.,
        AA, JTT+G, partition_0 = 1-121

    Args:
        part_path: path to sim_partitions.txt as generated by RGS
        format_type: standard   : data type, model, part name = x1 - x2
                     old        : data type, part name = x1 - x2

    Returns:
        list of tuples (data_type, sub_model, part_name, rest)
    """

    lst = []
    with open(part_path) as file:
        for line in file:
            split_line = line.split("=")
            tmp = [s.strip() for s in split_line[0].split(",")]
            if format_type == "standard":
                if len(tmp) == 3:                   # assuming that first entry is specified data type
                    lst.append((tmp[0], tmp[1], tmp[2], split_line[1].strip()))
                elif len(tmp) == 2:                 # assuming that data type was not specified
                    model_str = tmp[0]
                    part_name = tmp[1]
                    dt = guess_data_type_from_model(model_str)
                    lst.append((dt, model_str, part_name, split_line[1].strip()))
                else:
                    raise ValueError(f"unexpected number of partition arguments in '{tmp}'")
            elif format_type == "old":
                lst.append((tmp[0], tmp[1], split_line[1].strip()))
            else:
                raise ValueError(f"unknown part file format type: '{format_type}'")
    return lst


def write_raxml_part_file(part_path: str, substitution_model: str, set_subst: int):
    msa_dir = os.path.dirname(part_path)
    part_out_path = os.path.join(msa_dir, "raxml_partitions.txt")
    part_info = read_sim_part_file(part_path)

    with open(part_out_path, "w+") as file:
        for _, part_model, part_name, rest in part_info:
            tmp_model = substitution_model if set_subst else part_model
            file.write(f"{tmp_model}, {part_name} = {rest}\n")


def force_write_raxml_part_file(part_path, model):
    msa_dir = os.path.dirname(part_path)
    part_out_path = os.path.join(msa_dir, "raxml_partitions.txt")
    part_info = read_sim_part_file(part_path, "old")

    with open(part_out_path, "w+") as file:
        for dt, part_name, rest in part_info:
            file.write(f"{model}, {part_name} = {rest}\n")


def write_iqt_part_file(part_path: str, substitution_model: str, set_subst: int):
    msa_dir = os.path.dirname(part_path)
    part_out_path = os.path.join(msa_dir, "iqt_partitions.txt")
    part_info = read_sim_part_file(part_path)

    with open(part_out_path, "w+") as file:
        for data_type, part_model, part_name, rest in part_info:
            #tmp_model = substitution_model if set_subst else part_model
            file.write(f"{data_type}, {part_name} = {rest}\n")


def write_iqt_part_file_for_aa(part_path: str, substitution_model: str, set_subst: int):
    msa_dir = os.path.dirname(part_path)
    part_out_path = os.path.join(msa_dir, "iqt_partitions.txt")
    part_info = read_sim_part_file(part_path)

    with open(part_out_path, "w+") as file:
        for data_type, part_model, part_name, rest in part_info:
            tmp_model = substitution_model if set_subst else part_model
            file.write(f"{tmp_model}, {part_name} = {rest}\n")


def force_write_iqt_part_file(part_path, model):
    msa_dir = os.path.dirname(part_path)
    part_out_path = os.path.join(msa_dir, "iqt_partitions.txt")
    part_info = read_sim_part_file(part_path, "old")

    with open(part_out_path, "w+") as file:
        for dt, part_name, rest in part_info:
            file.write(f"{dt}, {part_name} = {rest}\n")


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


def download_datasets_from_config(root_dir, ref_dir="", max_num=0):
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
    if not max_num:
        max_num = len(grouped_results)
    else:
        max_num = int(max_num)

    key_list = list(grouped_results.keys())
    random.shuffle(key_list)
    for counter, tree_id in enumerate(key_list[:max_num]):
        print(f"{counter}: {tree_id}")
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


def create_repr_from_config(root_dir):
    with open("config.yaml") as file:
        config_dct = yaml.safe_load(file)

    dsc_name = config_dct["data_sets"]["used_dsc"]
    query = config_dct["data_sets"][dsc_name]["query"]
    db_name = config_dct["data_sets"][dsc_name]["db"]

    print(query)

    command_find = [
        "find",
        "-q", query,
        "-n", db_name,
        "-o", root_dir,
        "--list"
    ]
    grouped_results, _ = rgs.main(command_find)

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

        try:
            with open(tree_dct_path) as file:
                tree_dct = json.load(file)
        except Exception as e:
            print(f"error in {exp_id}:\n"
                  f"{e}")
            continue

        tree_id = tree_dct[0]["TREE_ID"]

        sel_dct[counter] = tree_id
        repr_dct[counter] = [tree_dct[0]]

    dst_dir = src_dir
    with open(os.path.join(dst_dir, "representatives.json"), "w+") as file:
        json.dump(repr_dct, file, indent=4)

    with open(os.path.join(dst_dir, "selected_datasets.json"), "w+") as file:
        json.dump(sel_dct, file, indent=4)


def remove_files_from_exps(root_dir, fn_list, prefix_list):
    # "1" rf.raxml.,top_test,quartet_,llh_,ntd_,true,concat_slh,cleaned_dir

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
                if not prefix:
                    print("no prefix!!!")
                    continue

                if fn.startswith(prefix):
                    file_path = os.path.join(exp_dir, fn)
                    if os.path.isfile(file_path):
                        os.remove(file_path)

        if counter % 100 == 0:
            print(counter)


def get_num_partitions(root_dir):
    tree_dct_path = os.path.join(root_dir, "tree_dict.json")
    with open(tree_dct_path) as file:
        tree_dct = json.load(file)
    num_part = tree_dct[0]["OVERALL_NUM_PARTITIONS"]
    return num_part


def read_pr_ab_matrix(path):
    """
    Reads the presence/absence matrix and returns the number of 0s and 1s
    as well as the matrix itself (currently as list of lists...)
    @param path: path to matrix file
    @return: number of 0s, number of 1s in the matrix, and the matrix itself
    """
    with open(path) as file:
        first_line = True
        num_bits = 0
        num_0 = 0
        num_1 = 0
        matrix = []

        for line in file:
            line_spl = line.rstrip().split()
            if first_line:
                first_line = False
                num_bits = int(line_spl[1])
                continue

            temp_list = []

            for i in range(1, num_bits + 1):
                val = int(line_spl[-i])
                if val == 0:
                    num_0 += 1
                elif val == 1:
                    num_1 += 1

                temp_list.append(val)
            temp_list.reverse()
            matrix.append(temp_list)

    return num_0, num_1, matrix


def write_part_files(root_dir, set_model=""):
    for counter, exp_id in enumerate(os.listdir(root_dir)):
        if counter % 100 == 0:
            print(counter)

        exp_dir = os.path.join(root_dir, exp_id, "msa_0", "default")
        if not os.path.isdir(exp_dir):
            exp_dir = os.path.join(root_dir, exp_id)
            if not os.path.isdir(exp_dir):
                continue

        num_part = get_num_partitions(exp_dir)
        if num_part == 1:
            continue

        part_path = os.path.join(exp_dir, "partitions.txt")

        if not set_model:
            write_raxml_part_file(part_path, substitution_model=set_model, set_subst=set_model)
            write_iqt_part_file(part_path, substitution_model=set_model, set_subst=set_model)
        else:
            force_write_raxml_part_file(part_path, set_model)
            force_write_iqt_part_file(part_path, set_model)


def write_new_part_files(root_dir, set_model=""):
    for counter, exp_id in enumerate(os.listdir(root_dir)):
        if counter % 100 == 0:
            print(counter)

        exp_dir = os.path.join(root_dir, exp_id, "msa_0", "default")
        if not os.path.isdir(exp_dir):
            exp_dir = os.path.join(root_dir, exp_id)
            if not os.path.isdir(exp_dir):
                continue

        try:
            num_part = get_num_partitions(exp_dir)
            if num_part == 1:
                continue
        except Exception as e:
            print(e)
            continue


        part_path = os.path.join(exp_dir, "sim_partitions.txt")

        if set_model:
            write_raxml_part_file(part_path, substitution_model=set_model, set_subst=set_model)
            write_iqt_part_file(part_path, substitution_model=set_model, set_subst=set_model)
        else:
            raise ValueError(f"no model specified!")


def write_new_part_files_for_aa(root_dir, set_model=""):
    for counter, exp_id in enumerate(os.listdir(root_dir)):
        if counter % 100 == 0:
            print(counter)

        exp_dir = os.path.join(root_dir, exp_id, "msa_0", "default")
        if not os.path.isdir(exp_dir):
            exp_dir = os.path.join(root_dir, exp_id)
            if not os.path.isdir(exp_dir):
                continue

        num_part = get_num_partitions(exp_dir)
        if num_part == 1:
            continue

        part_path = os.path.join(exp_dir, "sim_partitions.txt")

        if set_model:
            write_raxml_part_file(part_path, substitution_model=set_model, set_subst=set_model)
            write_iqt_part_file_for_aa(part_path, substitution_model=set_model, set_subst=set_model)
        else:
            raise ValueError(f"no model specified!")


def check_for_empty_sequences(sequences):
    counter = 0
    for sequence in sequences:
        if sequence.count("-") == len(sequence):
            counter += 1
    return counter


def assemble_sequences(path_list, matrix_path="", in_format="fasta"):
    records_list = []
    if os.path.isfile(matrix_path):
        _, _, matrix = read_pr_ab_matrix(matrix_path)
    else:
        matrix = []

    for path in path_list:
        records = SeqIO.parse(path, in_format)
        records_list.append(list(records))

    sequences = []
    part_lengths = []
    for i in range(len(records_list[0])):
        sequence = ""
        for j in range(len(path_list)):
            bit = matrix[i][j] if matrix else 1

            if bit:
                sequence += str(records_list[j][i].seq)
            else:
                sequence += "-" * len(str(records_list[j][i].seq))
            if i == 0:
                part_lengths.append(len(str(records_list[j][i].seq)))
        sequences = []

    return sequences, part_lengths


def check_empty_seqs_in_part_msa(part_msa_lst, pr_ab_matrix_path):
    part_msa_lst.sort(key=lambda x: x[1])
    sorted_seq_paths = [x[0] for x in part_msa_lst]
    assembled_sequences, part_lengths = assemble_sequences(sorted_seq_paths, matrix_path=pr_ab_matrix_path)

    contains_empty = check_for_empty_sequences(assembled_sequences)
    if contains_empty > 0:
        return True, part_lengths
    return False, part_lengths


def reset_exp(exp_dir):
    "rf.raxml.* top_test* quartet_* llh_* ntd_* true*"
    prefixes = ["rf.raxml.rfDistances", "llh_diffs", "ntd_dists.txt", "concat_slh.catpv", "cleaned_dir", "true"]
    for fn in os.listdir(exp_dir):
        for prefix in prefixes:
            if fn.startswith(prefix):
                rm_path = os.path.join(exp_dir, fn)
                print(f"deleting {rm_path}")
                try:
                    os.remove(rm_path)
                except:
                    pass


def get_msa_params_from_raxml_as_dct(raxml_log_path):
    dct = {}
    with open(raxml_log_path) as file:
        for line in file:
            if line.startswith("Alignment sites / patterns:"):
                line = line.replace(" ", "").split(":")
                pn = int(line[1].split("/")[1])
                dct["num_patterns"] = pn
            elif line.startswith("Gaps:"):
                line = line.replace(" ", "").split(":")
                gp = float(line[1].split("%")[0])
                dct["gaps"] = gp
            elif "Loaded alignment with" in line:
                tres = re.findall("taxa and (.*?) sites", line)
                sl = int(tres[0])
                dct["num_sites"] = sl

                tres = re.findall("with (.*?) taxa", line)
                nt = int(tres[0])
                dct["num_taxa"] = nt
    return dct


def make_csv_new_dirs(root_dir, out_path="", no_time=False):
    ds_name = os.path.basename(root_dir)
    if not ds_name:
        ds_name = os.path.basename(os.path.dirname(root_dir))

    df = pd.DataFrame()
    counter = 0

    not_in_counter = 0
    tool_counters = collections.defaultdict(int)
    sl_ratios = []
    sl_ratios_easy = []
    sl_ratios_easy_fu = []

    exclude_dirs = []

    for exp_id in os.listdir(root_dir):
        if exp_id in exclude_dirs:
            print(f"excluding {exp_id}")
            continue

        tree_dir = os.path.join(root_dir, exp_id)
        if not os.path.isdir(tree_dir):
            continue
        order_path = os.path.join(tree_dir, "top_test.names")
        rf_order_path = os.path.join(tree_dir, "rf.raxml.log")
        diff_path = os.path.join(tree_dir, "difficulty")
        llh_path = os.path.join(tree_dir, "llh_diffs")
        rfs_path = os.path.join(tree_dir, "rf.raxml.rfDistances")
        log_path = os.path.join(tree_dir, "true.raxml.log")
        ntd_path = os.path.join(tree_dir, "ntd_dists.txt")
        consel_path = os.path.join(tree_dir, "concat_slh.catpv")
        consel_tree_path = os.path.join(tree_dir, "concat_slh.siteLH")

        tree_dict_path = os.path.join(tree_dir, "tree_dict.json")
        if not os.path.isfile(tree_dict_path):
            print(f"no tree dict! {exp_id}")
            continue

        num_part = 1
        try:
            with open(tree_dict_path) as file:
                tree_dct = json.load(file)
            num_part = tree_dct[0]["OVERALL_NUM_PARTITIONS"]
        except Exception as e:
            print(f"error with {tree_dir}")
            print(e)

        if num_part > 1 and not os.path.isfile(consel_path):
            print(f"part > 1 and no consel!")

        exp_incomp = False
        for p in [order_path, diff_path, llh_path, rfs_path, log_path, ntd_path, consel_path]:
            if not os.path.isfile(p):
                print(f"{exp_id} is incomplete! {p} missing")
                exp_incomp = True
        if exp_incomp:
            continue

        if any(not os.path.isfile(p) for p in [order_path, diff_path, llh_path, rfs_path, log_path, ntd_path, consel_path, consel_tree_path]):
            print(f"{exp_id} is incomplete!")
            continue

        names = read_order_file(order_path)
        rf_names = read_rf_order_file(rf_order_path)
        consel_names = read_consel_name_file(consel_tree_path)

        difficulty = read_diff_file(diff_path)
        llhs, txt_llhs = parse_llh_file(llh_path)
        rfs = read_rf_file(rfs_path, rf_names)

        #sl, pn, gp = get_msa_params_from_raxml(log_path)
        msa_param_dct = get_msa_params_from_raxml_as_dct(log_path)
        sl = msa_param_dct["num_sites"]
        pn = msa_param_dct["num_patterns"]
        gp = msa_param_dct["gaps"]

        cleaned_names = [os.path.basename(n) for n in names]
        if not no_time:
            abs_times, rel_times = collect_run_times(tree_dir, cleaned_names)
        else:
            abs_times, rel_times = ({"no_abs_time": "true"}, {"no_rel_time": "true"})
        ntds = read_ntd_file(ntd_path, cleaned_names)
        consels = read_consel_test_results(consel_path, consel_names)

        sl_ratios.append(pn/sl)
        if difficulty < 0.2:
            sl_ratios_easy.append(pn / sl)

        if not consels["consel_true"] and difficulty < 0.2:
            already_added = 0
            not_in_counter += 1
            sl_ratios_easy_fu.append(pn/sl)
            for key in txt_llhs:
                if key == "llh_true":
                    continue
                if txt_llhs["llh_true"] == txt_llhs[key]:
                    print(tree_dir, key, sl, pn, pn/sl)
                    tool_counters[key] += 1
                    already_added += 1
            if already_added == 0:
                print(f"match not found")
            if already_added > 1:
                print("multiple highest llhs!")

        temp_dct = {
            "exp_id": exp_id,
            "difficulty": difficulty,
            "seq_len": sl,
            "num_patterns": pn,
            "num_taxa": msa_param_dct["num_taxa"],
            "gaps": gp,
            **llhs,
            **rfs,
            **ntds,
            **consels,
            **abs_times,
            **rel_times
        }
        temp_dct = {key: [value] if not isinstance(value, (list, tuple)) else value for key, value in temp_dct.items()}
        df = pd.concat([df, pd.DataFrame(temp_dct)])

        if counter % 100 == 0:
            print(f"{counter}")
        counter += 1

    if not out_path:
        df.to_csv(f"{ds_name}_stats.csv", index=False)
    else:
        df.to_csv(out_path, index=False)


def msas_to_datasets(*dirs, ext=""):
    accepted_exts = ["p", "fasta", "phylip", "phylips"]
    if ext:
        accepted_exts.append(ext)

    with open("config.yaml") as file:
        config_dct = yaml.safe_load(file)
    raxml_ng_path = os.path.abspath(config_dct["software"]["raxml_ng"]["command"])

    sel_dct = {}
    counter = 0
    out_dir = dirs[-1]
    for root_dir in dirs[:-1]:
        print(f"{counter}: {root_dir}")
        for fn in os.listdir(root_dir):
            ext = fn.split(".")[-1]
            if ext not in accepted_exts:
                continue
            source_path = os.path.join(root_dir, fn)
            dest_dir = os.path.join(out_dir, fn)
            create_dir_if_needed(dest_dir)

            dest_path = os.path.join(dest_dir, "assembled_sequences.fasta")

            if os.path.isfile(dest_path):
                sel_dct[counter] = fn
                counter += 1
                print(f"{dest_path} already exists! skipping")
                continue

            shutil.copy2(source_path, dest_path)

            difficulty = pythia_predict_difficulty(dest_path, pythia_predictor_path, raxml_ng_path)
            with open(os.path.join(dest_dir, "difficulty"), "w+") as file:
                file.write(f"{difficulty}\n")

            sel_dct[counter] = fn
            counter += 1

    with open(os.path.join(out_dir, "selected_datasets.json"), "w+") as file:
        json.dump(sel_dct, file, indent=4)


def simulate_msas_with_sizes(out_dir, taxa_nums, seq_lens, repeats=10, db_name="latest_all_0_7.db"):
    taxa_nums = [int(x) for x in taxa_nums.split(",")]
    seq_lens = [int(x) for x in seq_lens.split(",")]

    for num_taxa, num_sequences in itertools.product(taxa_nums, seq_lens):
        query = f"NUM_TAXA>={num_taxa} " \
                f"and NUM_TAXA<={max(num_taxa + 50, num_taxa*1.05)} " \
                f"and DATA_TYPE='DNA' " \
                f"and MODEL like 'GTR%' " \
                f"and ALPHA " \
                f"and OVERALL_NUM_ALIGNMENT_SITES > 0 " \
                f"and FREQ_A " \
                f"and RATE_AC"
        command_find = [
            "find",
            "-q", query,
            "-n", db_name,
            "--list"
        ]
        grouped_results, _ = rgs.main(command_find)
        print(grouped_results.keys())

        #return

        if len(grouped_results) == 0:
            raise ValueError(f"No results found for {query}")

        for i in range(int(repeats)):
            rand_id = np.random.choice(list(grouped_results.keys()))

            command_gen = [
                "generate",
                "-q", f"TREE_ID='{rand_id}'",
                "-o", out_dir,
                "--seq-len", f"{num_sequences}",
                "-n", db_name
            ]
            gen_results, out_paths = rgs.main(command_gen)

            current_ntaxa = gen_results[rand_id][0]["NUM_TAXA"]
            if current_ntaxa != num_taxa:
                print(f"num taxa is higher than requested: {current_ntaxa}, sampling {num_taxa} sequences")
                msa_path = os.path.join(out_paths[0], "assembled_sequences.fasta")
                sequences = msa_parser.parse_msa_somehow(msa_path)
                random.shuffle(sequences)

                sequences = sequences[:num_taxa]
                msa_parser.save_msa(sequences, msa_path, msa_format="fasta")


def plot_pd_llh_combined(file_path, out_dir, names=["pars", "raxml", "iqt2", "ft2"]):
    num_buckets = 50
    df = pd.read_csv(file_path)

    fig, axs = plt.subplots(int(len(names) / 2), int(len(names) / 2), figsize=(6 * len(names) / 2, 4 * len(names) / 2),
                            sharex=True, sharey=True,
                            layout="constrained")
    fig.tight_layout(rect=[2, 0.06, 1, 0.95])

    shared_ax = None
    for j in range(len(names)):
        name = names[j]
        cat_name = f"llh_{name}"
        # ax = axs[int(j / (len(names)/2)), j % int(len(names)/2)]
        # ax = plt.subplot(2, 2, j+1, sharex=shared_ax, sharey=shared_ax)
        ax = plt.subplot(2, 2, j + 1)

        df_pars = copy.deepcopy(df[[f"llh_{name}", "difficulty"]])
        df_pars[cat_name] = df["llh_true"] - df[f"llh_{name}"]

        # print(df_pars)
        for i in range(len(df_pars["difficulty"])):
            df_pars.iat[i, 1] = math.floor(df_pars.iat[i, 1] * num_buckets) / num_buckets

        group_pars = df_pars.groupby("difficulty")

        mean_val = group_pars.mean().rename(columns={name: "mean_llh"})
        median_val = group_pars.median().rename(columns={name: "median_llh"})
        val_25perc = group_pars.quantile(q=.25).rename(columns={name: "llh_25perc"})
        val_75perc = group_pars.quantile(q=.75).rename(columns={name: "llh_75perc"})
        val_5perc = group_pars.quantile(q=.05).rename(columns={name: "llh_5perc"})
        val_95perc = group_pars.quantile(q=.95).rename(columns={name: "llh_95perc"})
        max_llh = group_pars.max().rename(columns={name: "max_llh"})
        min_llh = group_pars.min().rename(columns={name: "min_llh"})
        count = group_pars.count().rename(columns={name: "count"})

        mean_val["mean_llh"] = group_pars.mean()
        median_val["median_llh"] = group_pars.median()
        val_25perc["llh_25perc"] = group_pars.quantile(q=.25)
        val_75perc["llh_75perc"] = group_pars.quantile(q=.75)
        val_5perc["llh_5perc"] = group_pars.quantile(q=.05)
        val_95perc["llh_95perc"] = group_pars.quantile(q=.95)
        max_llh["max_llh"] = group_pars.max()
        min_llh["min_llh"] = group_pars.min()

        data_frames = [mean_val, median_val, val_25perc, val_75perc, val_5perc, val_95perc, max_llh, min_llh, count]
        merged_df = pd.concat(data_frames, join="outer", axis=1)

        x = merged_df.index
        y = merged_df.mean_llh
        y_median = merged_df.median_llh
        y_25 = merged_df.llh_25perc
        y_75 = merged_df.llh_75perc
        y_5 = merged_df.llh_5perc
        y_95 = merged_df.llh_95perc

        ax.plot(x, y_median, color="darkorchid", marker=".", label="median")
        ax.plot(x, y_25, color="darkorchid", linestyle="-.", label="25th %")
        ax.plot(x, y_75, color="darkorchid", linestyle="-.", label="75th %")
        ax.plot(x, y_5, color="goldenrod", linestyle="-.", label="5th %")
        ax.plot(x, y_95, color="goldenrod", linestyle="-.", label="95th %")

        if name == "raxml":
            ax.legend()

        ax.set_yscale("log")
        ax.set_title(f"{TOOL_NAME_DICT[name]}", fontsize=12)

        if j == 0:
            shared_ax = ax

        ax.fill_between(x, y_25, y_75, alpha=0.1, color="darkorchid")
        ax.fill_between(x, y_5, y_95, alpha=0.1, color="goldenrod")
        ax.grid(alpha=0.2, which="both")

        plt.subplots_adjust(wspace=0.1, hspace=0.15)

    fig.supxlabel("difficulty", fontsize=16)
    fig.supylabel(f"absolute LnL difference", fontsize=16)

    fig.savefig(os.path.join(out_dir, f'pd_llh.svg'), format='svg')
    fig.savefig(os.path.join(out_dir, f'pd_llh.png'), format='png')
    fig.show()


def violin_plots(df, buckets, present_names, prefix, x_key, out_dir=""):
    fig, axs = plt.subplots(len(buckets), figsize=(2 * len(present_names), 4 * len(buckets)))
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    # fig.tight_layout()
    intervals = np.arange(0.0, 1 + 1 / len(buckets), 1 / len(buckets))

    print(f"num datasets: {len(df['difficulty'])}")

    for i, bucket in enumerate(buckets):
        ax = axs[i]
        bucket_df = df[df['Bucket'] == bucket]

        ax.violinplot([bucket_df[f"{prefix}_{name}"] for name in present_names], showmeans=True, showextrema=True,
                      showmedians=True)
        x = np.arange(len(present_names))
        ax.set_xticks(x + 1)
        temp_names = [TOOL_NAME_DICT[tn] if tn in TOOL_NAME_DICT else tn for tn in present_names]
        if i != len(axs) - 1:
            temp_names = []

        ax.set_xticklabels(temp_names, fontsize=16)
        ax.tick_params(axis="y", labelsize=16)

        if x_key in CAT_NAME_DICT:
            fig.suptitle(CAT_NAME_DICT[x_key], fontsize=20)
        else:
            fig.suptitle(x_key, fontsize=20)

        axs.flat[i].set_title(
            f"Difficulty bucket {i}: Range [{intervals[i]:.2f},{intervals[i + 1]:.2f}) - num of data sets {len(bucket_df['difficulty'])}",
            fontsize=16)
    if not out_dir:
        fig.savefig(f'out_img/{global_csv_path}_{x_key}.svg', format='svg', bbox_inches='tight')
        fig.savefig(f'out_img/{global_csv_path}_{x_key}.png', format='png', bbox_inches='tight')
    else:
        fig.savefig(os.path.join(out_dir, f'{x_key}.svg'), format='svg', bbox_inches='tight')
        fig.savefig(os.path.join(out_dir, f'{x_key}.png'), format='png', bbox_inches='tight')
    fig.show()


def violin_plots2(df, buckets, present_names, prefix, x_key, out_dir=""):
    if type(prefix) != list:
        prefix = [prefix, prefix]

    fig, axs = plt.subplots(len(buckets), figsize=(2 * len(present_names), 4 * len(buckets)))
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    # fig.tight_layout()
    intervals = np.arange(0.0, 1 + 1 / len(buckets), 1 / len(buckets))

    print(f"num datasets: {len(df['difficulty'])}")

    for i, bucket in enumerate(buckets):
        ax = axs[i]
        bucket_df = df[df['Bucket'] == bucket]

        width = 0.3
        offset = width / 2
        positions = [i + 1 - offset for i in range(len(present_names))]
        positions_twin = [i + 1 + offset for i in range(len(present_names))]

        ax.violinplot([bucket_df[f"{prefix[0]}_{name}"] for name in present_names], positions=positions, showmeans=True,
                      showextrema=True, showmedians=True, widths=width)
        x = np.arange(len(present_names))
        ax.set_xticks(x + 1)
        temp_names = [TOOL_NAME_DICT[tn] if tn in TOOL_NAME_DICT else tn for tn in present_names]
        if i != len(axs) - 1:
            temp_names = []

        ax_twin = ax.twinx()
        violin_twin = ax_twin.violinplot([bucket_df[f"{prefix[1]}_{name}"] for name in present_names],
                                         positions=positions_twin, showmeans=True, showextrema=True, showmedians=True,
                                         widths=width)
        ax_twin.get_xaxis().set_visible(False)

        for pc in violin_twin["bodies"]:
            pc.set_facecolor("tab:orange")

        ax.set_xticklabels(temp_names, fontsize=16)
        ax.tick_params(axis="y", labelsize=16)
        ax_twin.tick_params(axis="y", labelsize=16)

        ax.set_ylabel(CAT_NAME_DICT[x_key[0]], fontsize=16)
        ax_twin.set_ylabel(CAT_NAME_DICT[x_key[1]], fontsize=16)

        p1 = x_key[0]
        p2 = x_key[1]
        if p1 in CAT_NAME_DICT and p2 in CAT_NAME_DICT:
            fig.suptitle(f"{CAT_NAME_DICT[p1]} and {CAT_NAME_DICT[p2]}", fontsize=20)
        else:
            fig.suptitle(x_key, fontsize=20)

        axs.flat[i].set_title(
            f"Difficulty bucket {i}: Range [{intervals[i]:.2f},{intervals[i + 1]:.2f}) - num of data sets {len(bucket_df['difficulty'])}",
            fontsize=16)
    if not out_dir:
        fig.savefig(f'out_img/{x_key[0]}_{x_key[1]}.svg', format='svg', bbox_inches='tight')
        fig.savefig(f'out_img/{x_key[0]}_{x_key[1]}.png', format='png', bbox_inches='tight')
    else:
        fig.savefig(os.path.join(out_dir, f'{x_key[0]}_{x_key[1]}.svg'), format='svg', bbox_inches='tight')
        fig.savefig(os.path.join(out_dir, f'{x_key[0]}_{x_key[1]}.png'), format='png', bbox_inches='tight')
    fig.show()


def plot_rf(csv_path):
    x_key = "rf"
    df = pd.read_csv(csv_path)
    intervals = np.arange(0.0, 1.2, 0.2)
    filter_names = [""] # ["bionj"]

    present_names = adjust_names(df)
    present_names = [name for name in present_names if name not in filter_names]

    # Create a DataFrame to store the results
    table_data = pd.DataFrame(columns=['Bucket'] + list(display_names.values()))  # Include 'Bucket' column

    df['Bucket'] = pd.cut(df['difficulty'], bins=intervals, labels=False)
    df['Bucket'] = df['Bucket'].astype(int)

    buckets = sorted(df['Bucket'].unique())

    violin_plots(df, buckets, present_names, prefix="rf_true", x_key=x_key, out_dir=os.path.dirname(csv_path))


def plot_rf_and_llh(csv_path):
    x_key = "rf"
    df = pd.read_csv(csv_path)
    intervals = np.arange(0.0, 1.2, 0.2)
    filter_names = ["bionj"]

    present_names = adjust_names(df)
    present_names = [name for name in present_names if name not in filter_names]

    # Create a DataFrame to store the results
    table_data = pd.DataFrame(columns=['Bucket'] + list(display_names.values()))  # Include 'Bucket' column

    df['Bucket'] = pd.cut(df['difficulty'], bins=intervals, labels=False)
    df['Bucket'] = df['Bucket'].astype(int)

    buckets = sorted(df['Bucket'].unique())

    for name in present_names:
        df[f"llh_diff_{name}"] = (-(np.array(df[f"llh_{name}"]) - np.array(df['llh_true'])) / np.array(
            df['llh_true'])) * 100

    violin_plots2(df, buckets, present_names, prefix=["rf_true", "llh_diff"], x_key=["rf", "llh_percent"], out_dir=os.path.dirname(csv_path))


def plot_ntd(csv_path):
    filter_names = ["bionj"]

    df = pd.read_csv(csv_path)
    intervals = np.arange(0.0, 1.2, 0.2)

    present_names = adjust_names(df)
    present_names = [name for name in present_names if name not in filter_names]

    # Create a DataFrame to store the results
    table_data = pd.DataFrame(columns=['Bucket'] + list(display_names.values()))  # Include 'Bucket' column

    df['Bucket'] = pd.cut(df['difficulty'], bins=intervals, labels=False)
    df['Bucket'] = df['Bucket'].astype(int)

    buckets = sorted(df['Bucket'].unique())

    violin_plots(df, buckets, present_names, prefix="ntd_true", x_key="ntd", out_dir=os.path.dirname(csv_path))


def plot_llh(csv_path):
    filter_names = ["bionj"]

    df = pd.read_csv(csv_path)
    intervals = np.arange(0.0, 1.2, 0.2)
    present_names = adjust_names(df)
    present_names = [name for name in present_names if name not in filter_names]

    # Create a DataFrame to store the results
    table_data = pd.DataFrame(columns=['Bucket'] + list(display_names.values()))  # Include 'Bucket' column

    df['Bucket'] = pd.cut(df['difficulty'], bins=intervals, labels=False)
    df['Bucket'] = df['Bucket'].astype(int)

    buckets = sorted(df['Bucket'].unique())

    draw_df = copy.deepcopy(df)
    for name in present_names:
        draw_df[f"llh_diff_{name}"] = (-(np.array(df[f"llh_{name}"]) - np.array(df['llh_true'])) / np.array(
            df['llh_true'])) * 100

    for bucket in buckets:
        bucket_df = df[df['Bucket'] == bucket]

        row_data = {'Bucket': bucket}

        for name in present_names:
            col_name = f"llh_{name}"
            row_data[display_names[name]] = np.mean((bucket_df[col_name] - bucket_df['llh_true']))
            # draw_df[f"llh_diff_{name}"] = (np.array(bucket_df[col_name]) - np.array(bucket_df['llh_true']))

    violin_plots(draw_df, buckets, present_names, prefix="llh_diff", x_key="llh_percent", out_dir=os.path.dirname(csv_path))


def plot_consel(csv_path):
    df = pd.read_csv(csv_path)
    intervals = np.arange(0.0, 1.2, 0.2)
    filter_names = ["bionj"]

    present_names_true = adjust_names_true(df)
    present_names_true = [name for name in present_names_true if name not in filter_names]

    # Create a DataFrame to store the results
    table_data = pd.DataFrame(columns=['Bucket'] + list(display_names_true.values()))  # Include 'Bucket' column

    df['Bucket'] = pd.cut(df['difficulty'], bins=intervals, labels=False)
    df['Bucket'] = df['Bucket'].astype(int)

    buckets = sorted(df['Bucket'].unique())

    for bucket in buckets:
        bucket_df = df[df['Bucket'] == bucket]

        row_data = {'Bucket': bucket}

        for name in present_names_true:
            col_name = f"consel_{name}"
            top_vec = np.where(bucket_df[col_name] == 1, 1, 0)
            row_data[display_names_true[name]] = np.mean(top_vec)

    violin_plots(df, buckets, present_names_true, prefix="consel", x_key="consel", out_dir=os.path.dirname(csv_path))


def print_abs_runtimes(csv_path):
    df = pd.read_csv(csv_path)
    intervals = np.arange(0.0, 1.2, 0.2)

    present_names = adjust_names(df)

    # Create a DataFrame to store the results
    table_data = pd.DataFrame(columns=['Bucket'] + list(display_names.values()))  # Include 'Bucket' column

    df['Bucket'] = pd.cut(df['difficulty'], bins=intervals, labels=False)
    df['Bucket'] = df['Bucket'].astype(int)

    buckets = sorted(df['Bucket'].unique())

    for bucket in buckets:
        bucket_df = df[df['Bucket'] == bucket]
        print(f"{bucket}: {len(bucket_df['rf_true_raxml'])}")

    for bucket in buckets:
        bucket_df = df[df['Bucket'] == bucket]
        row_data = {'Bucket': bucket}

        for name in present_names:
            if name == "true" or name == "bigraxml":
                continue

            col_name = f"rel_time_{name}"
            row_data[display_names[name]] = np.mean(bucket_df[col_name])

        # Add row_data to table_data
        table_data = pd.concat([table_data, pd.DataFrame([row_data])], ignore_index=True)

    # Find minimum values in each row and bold them
    table_data_styled = table_data.style.apply(
        lambda row: ['textbf:--rwrap;' if val == row.iloc[1:].min() else '' for val in row], axis=1)
    table_data_styled = table_data_styled.hide(axis="index")

    # Convert table_data to LaTeX table format
    latex_table = table_data_styled.to_latex(hrules=True)

    # Print the LaTeX table
    print(latex_table)


def create_plots(root_dir):
    csv_path = os.path.join(root_dir, "stats.csv")
    make_csv_new_dirs(root_dir, out_path=csv_path)
    plot_pd_llh_combined(csv_path, root_dir)

    plot_rf(csv_path)
    plot_rf_and_llh(csv_path)
    plot_llh(csv_path)
    plot_consel(csv_path)
    plot_ntd(csv_path)


if __name__ == "__main__":
    globals()[sys.argv[1]](*sys.argv[2:])
