#!/usr/bin/env python3
import collections
import copy
import itertools
import json
import math
import os
import random
import re
import statistics
import sys
from ast import literal_eval

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from Bio import Phylo, SeqIO, Seq, SeqRecord


CAT_NAME_DICT = {
    "rf": "RF distance",
    "llh": "LnL difference",
    "llh_percent": "LnL difference (percent)",
    "quartet": "Quartet distance",
    "top": "passed statistical tests",
    "additional_rf": "RF distance",
    "additional_llh_percent": "LnL difference (percent)",
    "_additional_llh_percent_difficulty_single": "LnL difference (percent)"
}
TOOL_NAME_DICT = {
    "true": "true",
    "pars": "parsimony",
    "rand": "random",
    "raxml": "RAxML-NG",
    "iqt2": "IQ-TREE2",
    "ft2": "FastTree2",
    "bigraxml": "RAxML-NG-100"
}



def draw_game_plots(out_dir, info_dict, point_lists, eps, tag, sign=1, colors=["dodgerblue", "orange", "green"]):
    for key in info_dict:
        x_list = info_dict[key]
        y_list = []

        out_x_list1 = []
        out_y_list1 = []
        out_x_list2 = []
        out_y_list2 = []
        out_x_list3 = []
        out_y_list3 = []
        for i in range(0, len(x_list)):
            val_fasttree = point_lists[1][i]
            val_raxml = point_lists[0][i]
            thresh = val_fasttree * eps

            if sign * ((val_fasttree - thresh) - val_raxml) < 0:
                out_x_list2.append(x_list[i])
                out_y_list2.append(point_lists[1][i])
            elif sign * ((val_fasttree - thresh) - val_raxml) == 0:
                out_x_list3.append(x_list[i])
                out_y_list3.append(point_lists[2][i])
            else:
                out_x_list1.append(x_list[i])
                out_y_list1.append(point_lists[0][i])

        fig = plt.figure()
        plt.plot(out_x_list1, out_y_list1, "o", color=colors[0], alpha=0.7)
        plt.plot(out_x_list2, out_y_list2, "o", color=colors[1], alpha=0.7)
        plt.plot(out_x_list3, out_y_list3, "o", color=colors[2], alpha=0.7)
        plt.xlabel(key)
        plt.ylabel(tag)
        plt.savefig(os.path.join(out_dir, f"{tag}_{key}.png"))
        plt.close(fig)

    p1_wins = 0
    p2_wins = 0
    eq = 0

    for i in range(len(point_lists[0])):
        p1_score = point_lists[0][i]
        p2_score = point_lists[1][i]
        thresh = eps

        if sign * ((p2_score - thresh) - p1_score) < 0:
            p2_wins += 1
        elif sign * ((p2_score - thresh) - p1_score) == 0:
            eq += 1
        else:
            p1_wins += 1

    with open(os.path.join(out_dir, f"{tag}.txt"), "w+") as file:
        file.write(f"p1: {point_lists[0]}\n"
                   f"p2: {point_lists[1]}\n"
                   f"eq: {point_lists[2]}\n"
                   f"p1 wins: {p1_wins}\n"
                   f"p2 wins: {p2_wins}\n"
                   f"eq: {eq}")


def get_iqt_test_results_simple(file_path):
    plus_num_list = []

    with open(file_path) as file:
        record = 0
        for line in file:
            line = line.rstrip()
            if record == 2:
                if not line:
                    record = 0
                else:
                    # plus_num_list.append((line.count("+"), line.count("-")))

                    # we throw out KH, SH and their weighted versions
                    conf_list = line.strip().split()[4::2]
                    conf_list = [conf_list[0], conf_list[5], conf_list[6]]

                    for e in conf_list:
                        if e not in ["+", "-"]:
                            raise ValueError(f"illegal test line: {line}\n"
                                             f"{conf_list}")
                    plus_num_list.append((conf_list.count("+"), conf_list.count("-")))

            if line == "-------------------------------------------------------------------------------------------" and record == 1:
                record = 2
            elif record == 2:
                pass
            else:
                record = 0
            if line == "Tree      logL    deltaL  bp-RELL    p-KH     p-SH    p-WKH    p-WSH       c-ELW       p-AU":
                record = 1

    return plus_num_list


def get_tree_name_from_base_name(base_name):
    if "_" in base_name:
        return base_name.split("_")[0]
    else:
        return base_name.split(".")[0]


def get_tree_names_from_rf_log(log_path):
    names = []
    with open(log_path) as file:
        for line in file:
            line = line.strip()
            if line.startswith("Reading input trees from file: "):
                path = line.split("Reading input trees from file: ")[1]
                name = get_tree_name_from_base_name(os.path.basename(path))
                names.append(name)
    return names


def get_rfs_from_file(rf_file):
    rel_rfs = []
    with open(rf_file) as file:
        for line in file:
            line = line.strip()
            rel_rf = float(line.split()[-1])
            rel_rfs.append(rel_rf)
    return rel_rfs


def get_llh_from_log(log_path):
    ret = 100
    kw = "Final LogLikelihood: "

    with open(log_path) as file:
        for line in file:
            line = line.rstrip()
            if line.startswith(kw):
                ret = float(line.split(":")[1])
                return ret
    raise ValueError(f"could not extract llh from log {log_path}!")


def get_difficulty(file_path):
    with open(file_path) as file:
        for line in file:
            return float(line.strip())


def sort_result_files(file_list):

    for file in file_list:
        base_name = os.path.basename(base_name)
        dir_name = os.path.basename(os.path.dirname(file))


def get_llhs_from_file(llh_file):
    scores = collections.defaultdict(lambda: [])
    with open(llh_file) as file:
        true_name = ""
        for line in file:
            line = line.strip()
            line = line.split()
            if not true_name:
                true_name = line[0]
            else:
                if true_name != line[0]:
                    break
            llh_diff = float(line[-1])
            name = line[1]

            scores[name].append(llh_diff)
    return scores


def get_norm_quartet_dists(quartet_file):
    num_quartets = 0
    norm_dists = []
    with open(quartet_file) as file:
        first_line = True
        for line in file:
            line = line.strip()
            if first_line:
                first_line = False
                num_quartets = float(line)
                continue
            norm_dist = float(line.split()[0]) / num_quartets
            norm_dists.append(norm_dist)
    return norm_dists


def get_msa_params_from_raxml_log(raxml_log_path):
    sl = pn = gp = ntaxa = 0
    with open(raxml_log_path) as file:
        for line in file:
            if line.startswith("Alignment sites / patterns:"):
                line = line.replace(" ", "").split(":")
                # sl = int(line[1].split("/")[0])
                pn = int(line[1].split("/")[1])
            elif line.startswith("Gaps:"):
                line = line.replace(" ", "").split(":")
                gp = float(line[1].split("%")[0])
            elif "Loaded alignment with" in line:
                tres = re.findall("taxa and (.*?) sites", line)
                sl = int(tres[0])
                tres = re.findall("Loaded alignment with (.*?) taxa and", line)
                ntaxa = int(tres[0])
            elif "Alignment comprises " in line:
                tres = re.findall("Alignment comprises (.*?) taxa", line)
                if len(tres) > 0:
                    ntaxa = int(tres[0])

    if ntaxa <= 0:
        print(f"problem with {ntaxa} taxa in {raxml_log_path}")
        ntaxa = 1

    return sl, pn, gp, ntaxa



def file_to_list(file_path):
    ret_list = []
    with open(file_path) as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            ret_list.append(line)
    return ret_list


def make_value_lists(experiment_files):
    individual_values = collections.defaultdict(lambda: collections.defaultdict(lambda: []))
    averaged_values = collections.defaultdict(lambda: collections.defaultdict(lambda: []))

    for exp_id in experiment_files:
        temp_list = collections.defaultdict(lambda: collections.defaultdict(lambda: []))

        for msa_id in experiment_files[exp_id]:
            difficulty_added_for_name = {}

            # starting with rfs
            rf_file = experiment_files[exp_id][msa_id]["rf"][0]
            log_file = rf_file.replace("rfDistances", "log")
            current_names = get_tree_names_from_rf_log(log_file)
            rfs = get_rfs_from_file(rf_file)

            difficulty_file = experiment_files[exp_id][msa_id]["difficulty"][0]
            difficulty = get_difficulty(difficulty_file)

            raxml_log_file = experiment_files[exp_id][msa_id]["info"][0]
            _sl, _pn, _gp, _ntaxa = get_msa_params_from_raxml_log(raxml_log_file)
            signal = _pn / _ntaxa

            for i in range(1, len(current_names)):
                name = current_names[i]
                if name == "rand":
                    continue

                rf = rfs[i - 1]

                individual_values["rf"][name].append(rf)
                temp_list["rf"][name].append(rf)

                individual_values["difficulty"][name].append(difficulty)
                individual_values["signal"][name].append(signal)
                temp_list["difficulty"][name].append(difficulty)
                temp_list["signal"][name].append(signal)
                difficulty_added_for_name[name] = 1


            # fixed rf for max-llh-found-tree
            llh_file = experiment_files[exp_id][msa_id]["llh"][0]
            llh_scores = {}
            with open(llh_file) as file:
                true_llh = False
                for line in file:
                    split_line = line.strip().split()
                    if not true_llh:
                        llh_scores["true"] = float(split_line[2])
                    llh_scores[split_line[1]] = float(split_line[3])

            """llh_perc_files = experiment_files[exp_id][msa_id]["llh_percent"]
            for file in llh_perc_files:
                name = get_tree_name_from_base_name(os.path.basename(file))
                llh_scores[name] = get_llh_from_log(file)"""

            rf_dict = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))

            max_llh = None
            max_name = None
            for i in range(len(current_names)):
                name = current_names[i]
                llh = llh_scores[name]
                if not max_llh or max_llh < llh:
                    max_llh = llh
                    max_name = name

            c = 0
            for i in range(len(current_names) - 1):
                for j in range(i + 1, len(current_names)):
                    name1 = current_names[i]
                    name2 = current_names[j]

                    rf_dict[name1][name2] = rfs[c]
                    rf_dict[name2][name1] = rfs[c]
                    c += 1

            for i in range(len(current_names)):
                name = current_names[i]
                if name == "rand":
                    continue

                rf = rf_dict[max_name][name]

                individual_values["additional_rf"][name].append(rf)
                temp_list["additional_rf"][name].append(rf)

                if name not in difficulty_added_for_name:
                    individual_values["difficulty"][name].append(difficulty)
                    individual_values["signal"][name].append(signal)
                    temp_list["difficulty"][name].append(difficulty)
                    temp_list["signal"][name].append(signal)
                    difficulty_added_for_name[name] = 1



            # additional rfs stuff: here we calculated the distance of the trees to the pars starting trees
            """
            rf_file = experiment_files[exp_id][msa_id]["additional_rf"][0]
            current_names = ["pars", "true", "raxml", "iqt2", "ft2"]
            rfs = get_rfs_from_file(rf_file)

            difficulty_file = experiment_files[exp_id][msa_id]["difficulty"][0]
            difficulty = get_difficulty(difficulty_file)

            raxml_log_file = experiment_files[exp_id][msa_id]["info"][0]
            _sl, _pn, _gp, _ntaxa = get_msa_params_from_raxml_log(raxml_log_file)
            signal = _pn / _ntaxa

            for i in range(1, len(current_names)):
                name = current_names[i]
                if name == "rand":
                    continue

                rf = rfs[i - 1]

                individual_values["additional_rf"][name].append(rf)
                temp_list["additional_rf"][name].append(rf)

                if name not in difficulty_added_for_name:
                    individual_values["difficulty"][name].append(difficulty)
                    individual_values["signal"][name].append(signal)
                    temp_list["difficulty"][name].append(difficulty)
                    temp_list["signal"][name].append(signal)
                    difficulty_added_for_name[name] = 1
            """




            # llhs
            llh_file = experiment_files[exp_id][msa_id]["llh"][0]
            llh_scores = get_llhs_from_file(llh_file)

            for name in llh_scores:
                if name == "rand":      # exclude random since the difference there is much higher than that if the rest
                    continue

                if len(llh_scores[name]) > 1:
                    raise ValueError(f"llh scores error: {name} {llh_scores[name]}")
                individual_values["llh"][name].append(llh_scores[name][0])

                temp_list["llh"][name].append(llh_scores[name][0])

                if name not in difficulty_added_for_name:
                    individual_values["difficulty"][name].append(difficulty)
                    individual_values["signal"][name].append(signal)
                    temp_list["difficulty"][name].append(difficulty)
                    temp_list["signal"][name].append(signal)
                    difficulty_added_for_name[name] = 1


            # llh diffs in percent
            llhs = {}
            llh_file = experiment_files[exp_id][msa_id]["llh"][0]
            with open(llh_file) as file:
                for line in file:
                    split_line = line.strip().split()
                    name1 = split_line[0]
                    name2 = split_line[1]
                    if name1 != "true":
                        continue

                    llh_perc = float(split_line[-2])
                    llhs[name2] = llh_perc

            for name in llhs:
                if name == "rand" or name == "true":
                    continue

                llh_diff = llhs[name]
                individual_values["llh_percent"][name].append(llh_diff)
                temp_list["llh_percent"][name].append(llh_diff)

                if name not in difficulty_added_for_name:
                    individual_values["difficulty"][name].append(difficulty)
                    individual_values["signal"][name].append(signal)
                    temp_list["difficulty"][name].append(difficulty)
                    temp_list["signal"][name].append(signal)
                    difficulty_added_for_name[name] = 1


            # llh diffs in percent compared to highest-llh-found tree
            llhs = {}
            llh_file = experiment_files[exp_id][msa_id]["llh"][0]
            llhs_dict = collections.defaultdict(lambda: 0)
            with open(llh_file) as file:
                for line in file:
                    split_line = line.strip().split()
                    name1 = split_line[0]
                    name2 = split_line[1]
                    if name1 != "true":
                        continue

                    llhs_dict[name1] = float(split_line[2])
                    llhs_dict[name2] = float(split_line[3])

                    llh_perc = float(split_line[-2])
                    llhs[name2] = llh_perc

            for name in llhs:
                if name == "rand":
                    continue

                # llh_diff = llhs[name]

                llh_diff = ((llhs_dict[max_name] - llhs_dict[name]) / llhs_dict[max_name]) * 100
                individual_values["additional_llh_percent"][name].append(llh_diff)
                temp_list["additional_llh_percent"][name].append(llh_diff)

                if name not in difficulty_added_for_name:
                    individual_values["difficulty"][name].append(difficulty)
                    individual_values["signal"][name].append(signal)
                    temp_list["difficulty"][name].append(difficulty)
                    temp_list["signal"][name].append(signal)
                    difficulty_added_for_name[name] = 1



            # topology tests
            top_file = experiment_files[exp_id][msa_id]["top"][0]
            names_file = experiment_files[exp_id][msa_id]["top"][1]
            names = file_to_list(names_file)
            plus_num_list = get_iqt_test_results_simple(top_file)

            for i in range(len(names)):
                name = names[i]
                if name == "rand":
                    continue

                score = plus_num_list[i][0]

                individual_values["top"][name].append(score)

                temp_list["top"][name].append(score)

                if name not in difficulty_added_for_name:
                    individual_values["difficulty"][name].append(difficulty)
                    individual_values["signal"][name].append(signal)
                    temp_list["difficulty"][name].append(difficulty)
                    temp_list["signal"][name].append(signal)
                    difficulty_added_for_name[name] = 1


            # quartet dists
            quartet_file = experiment_files[exp_id][msa_id]["quartet"][0]
            names_file = experiment_files[exp_id][msa_id]["quartet"][1]
            names = file_to_list(names_file)
            norm_quartet_dists = get_norm_quartet_dists(quartet_file)

            for i in range(1, len(names)):  # ignore first one, since it is the true tree
                name = names[i]
                dist = norm_quartet_dists[i]

                individual_values["quartet"][name].append(dist)

                temp_list["quartet"][name].append(dist)

                if name not in difficulty_added_for_name:
                    individual_values["difficulty"][name].append(difficulty)
                    individual_values["signal"][name].append(signal)
                    temp_list["difficulty"][name].append(difficulty)
                    temp_list["signal"][name].append(signal)
                    difficulty_added_for_name[name] = 1


        for cat in temp_list:
            for name in temp_list[cat]:
                temp_vals = temp_list[cat][name]
                temp_mean = statistics.mean(temp_vals) if len(temp_vals) > 1 else temp_vals[0]
                averaged_values[cat][name].append(temp_mean)

    return individual_values, averaged_values


def split_value_list(values, x_key, y_key, buckets):
    bucket_value_list = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: [])))
    all_names = {}

    for bucket in buckets:
        for name in values[x_key]:
            x_values = values[x_key][name]
            y_values = values[y_key][name]
            all_names[name] = 1

            if len(x_values) != len(y_values):
                raise ValueError(f"not matching x_value and y_value lists for {bucket} {name}:\n"
                                 f"{x_key}: {x_values}\n"
                                 f"{y_key}: {y_values}\n")

            for i in range(len(x_values)):
                x_value = x_values[i]
                y_value = y_values[i]

                if bucket[0] <= y_value < bucket[1]:
                    bucket_value_list[bucket][name][x_key].append(x_value)
                    bucket_value_list[bucket][name][y_key].append(y_value)

    return bucket_value_list, list(all_names)


def make_violin_plots(out_file, bucket_value_list, all_names, x_key, buckets, title_prefix=""):
    num_buckets = len(buckets)

    fig, axs = plt.subplots(num_buckets, figsize=(2*len(all_names),4*num_buckets))
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    for i in range(len(axs)):
        ax = axs[i]
        bucket = buckets[i]
        if bucket not in bucket_value_list:
            bucket_value_list[bucket] = collections.defaultdict(lambda: collections.defaultdict(lambda: []))

        if len(bucket_value_list[bucket].keys()) == 0:
            for n in all_names:
                bucket_value_list[bucket][n][x_key] = [0]
        ax.violinplot([bucket_value_list[bucket][name][x_key] for name in all_names], showmeans=True, showextrema=True, showmedians=True)
        x = np.arange(len(all_names))
        ax.set_xticks(x + 1)
        temp_names = [TOOL_NAME_DICT[tn] if tn in TOOL_NAME_DICT else tn for tn in all_names]
        if i != len(axs)-1:
            temp_names = []

        ax.set_xticklabels(temp_names, fontsize=16)
        ax.tick_params(axis="y", labelsize=16)

        if x_key in CAT_NAME_DICT:
            #ax.set_ylabel(CAT_NAME_DICT[x_key])
            fig.suptitle(CAT_NAME_DICT[x_key], fontsize=20)
        else:
            #ax.set_ylabel(x_key)
            fig.suptitle(x_key, fontsize=20)

        axs.flat[i].set_title(f"{title_prefix} range [{bucket[0]:.2f},{bucket[1]:.2f}) - num of data sets {len(bucket_value_list[bucket][all_names[0]][x_key])}",
                              fontsize=16)
    plt.savefig(out_file, bbox_inches='tight', pad_inches = 0.1)
    plt.close()


def draw_all(root_dir, experiment_files, reuse=False):
    individual_values, averaged_values = make_value_lists(experiment_files)

    combs = [
        [(individual_values, "single"), (averaged_values, "averaged")],
        ["rf", "llh", "top",  "quartet", "llh_percent"],
        [(difficulty_buckets, "difficulty"), (signal_buckets, "signal")]
    ]
    combs[1].append("additional_rf")
    combs[1].append("additional_llh_percent")

    combs = itertools.product(*combs)

    for comb in combs:
        val = comb[0]
        cat = comb[1]
        bucket = comb[2]

        #with open(os.path.join(root_dir, f"_{cat}_{bucket[1]}_{val[1]}.json"), "w+") as file:
        #    json.dump(val[0], file, indent=4)

        if not reuse:
            bucket_value_list, all_names = split_value_list(val[0], cat, bucket[1], bucket[0])

            temp_bucket_value_list = {}
            for key in bucket_value_list:
                temp_bucket_value_list[str(key)] = bucket_value_list[key]

            #do_statistics(root_dir, temp_bucket_value_list, cat, bucket[1], val[1])        # TODO: maybe enable this again at some point

            #with open(os.path.join(root_dir, f"_{cat}_{bucket[1]}_{val[1]}.json"), "w+") as file:
            #    json.dump(temp_bucket_value_list, file, indent=4)

        else:
            bucket_value_list = {}
            all_names = {}
            with open(os.path.join(root_dir, f"_{cat}_{bucket[1]}_{val[1]}.json")) as file:
                temp_bucket_value_list = json.load(file)
                bucket_value_list = temp_bucket_value_list  # there was a purpose for this (i'm sure)
            for tbucket in bucket_value_list:
                for key in bucket_value_list[tbucket]:
                    all_names[key] = 1
            all_names = list(all_names.keys())

        make_violin_plots(os.path.join(root_dir, f"_{cat}_{bucket[1]}_{val[1]}.png"), bucket_value_list, all_names, cat, bucket[0], title_prefix=bucket[1])


def do_statistics(root_dir, bucket_value_list, cat, bucket_name, val):
    out_dict = collections.defaultdict(lambda: collections.defaultdict(lambda: (0, 0)))
    for bucket in bucket_value_list:
        tool_list = list(bucket_value_list[bucket].keys())
        for i in range(len(tool_list) - 1):
            for j in range(i + 1, len(tool_list)):
                dist1 = bucket_value_list[bucket][tool_list[i]][cat]
                dist2 = bucket_value_list[bucket][tool_list[j]][cat]

                stat, p = ks_2samp(dist1, dist2)
                out_dict[bucket][f"{tool_list[i]}-{tool_list[j]}"] = (stat, p)
    with open(os.path.join(root_dir, f"_ks_{cat}_{bucket_name}_{val}.json"), "w+") as file:
        json.dump(out_dict, file, indent=4)


def do_stuff(root_dir):
    required_files = {
        "rf": ["rf.raxml.rfDistances"],
        # "additional_rf": ["additional_rfs.raxml.rfDistances"],
        "llh": ["llh_diffs"],
        "top": ["top_test.iqtree", "top_test.names"],
        "difficulty": ["difficulty"],
        "info": ["raxml_eval.raxml.log", "tree_dict.json"],
        #"llh_percent": ["true.raxml.log", "pars_eval.raxml.log", "rand_eval.raxml.log", "raxml_eval.raxml.log",
        #                "iqt2_eval.raxml.log", "ft2_bin_eval.raxml.log"],
        "quartet": ["quartet_dists.txt", "top_test.names"]
    }
    files = collections.defaultdict(lambda: collections.defaultdict(lambda: []))
    experiment_files = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: [])))

    exp_dirs = os.listdir(root_dir)
    for exp_id in exp_dirs:
        exp_dir = os.path.join(root_dir, exp_id)
        if not os.path.isdir(exp_dir):
            continue

        msa_dirs = os.listdir(exp_dir)
        for msa_id in msa_dirs:
            msa_dir = os.path.join(root_dir, exp_id, msa_id, "default")
            if not os.path.isdir(msa_dir):
                continue

            res_files = os.listdir(msa_dir)

            found_all_for_all = []
            for key in required_files:
                found_all = all(elem in res_files for elem in required_files[key])
                found_all_for_all.append(found_all)

            if not all(f_all for f_all in found_all_for_all):
                continue

            for key in required_files:
                found_all = all(elem in res_files for elem in required_files[key])
                if found_all:
                    for i in range(len(required_files[key])):
                        req_file = required_files[key][i]
                        files[key][i].append(os.path.join(msa_dir, req_file))

                        experiment_files[exp_id][msa_id][key].append(os.path.abspath(os.path.join(msa_dir, req_file)))


    #with open(os.path.join(root_dir, "debug.json"), "w+") as file:
    #    json.dump(experiment_files, file, indent=4)

    draw_all(root_dir, experiment_files)


def do_other_stuff(root_dir):
    for file_name in os.listdir(root_dir):
        file_path = os.path.join(root_dir, file_name)
        if not (file_name.endswith(".json") and file_name.startswith("_")):
            continue

        split_name = file_name.split(".")[0].split("_")
        if "llh_percent" not in file_name:
            cat = split_name[1]
            bname = split_name[2]
            k = split_name[3]
        else:
            cat = "llh_percent"
            bname = split_name[3]
            k = split_name[4]

        if bname == "difficulty":
            buckets = difficulty_buckets
        else:
            buckets = signal_buckets

        with open(file_path) as file:
            temp_bucket_value_list = json.load(file)
            bucket_value_list = {}
            all_names = {}
            for tbucket in temp_bucket_value_list:
                for key in temp_bucket_value_list[tbucket]:
                    if (cat != "quartet" and key == "rand") or (cat != "top" and key == "true"):
                        continue

                    all_names[key] = 1
                bucket_value_list[literal_eval(tbucket)] = temp_bucket_value_list[tbucket]
            #print(temp_bucket_value_list)
            #print(bucket_value_list)
            all_names = list(all_names.keys())
        try:
            full_lists = collections.defaultdict(lambda: collections.defaultdict(lambda: []))

            for bucket in bucket_value_list:
                for name in bucket_value_list[bucket]:
                    if not "difficulty" in bucket_value_list[bucket][name]:
                        continue

                    for key in bucket_value_list[bucket][name]:
                        if key == "difficulty" and name != "raxml":
                            continue
                        val_list = bucket_value_list[bucket][name][key]

                        full_lists[key][name].extend(val_list)
                        print(f"{bucket} {name} {key}: mean {statistics.mean(val_list)} median {statistics.median(val_list)}")
                print()

            for key in full_lists:
                names = list(full_lists[key].keys())
                out_lines = [tuple(names)]
                out_lines.extend(zip(*[full_lists[key][name] for name in names]))
                with open(os.path.join(root_dir, f"test_dots_{key}.txt"), "w+") as file:
                    for line in out_lines:
                        file.write(f"{','.join(map(str, line))}\n")

            make_violin_plots(os.path.join(root_dir, f"_02_{cat}_{bname}_{k}.png"), bucket_value_list, all_names,
                          cat, buckets, title_prefix=bname)
        except:
            print(f"{cat} {bname} {k}")
            print(bucket_value_list)
            raise ValueError(f"{cat} {bname} {k}")



"""
other_other_stuff() and other_other_stuff_combined() are based on the article by Brian Mattis found at 
https://towardsdatascience.com/the-matplotlib-line-plot-that-crushes-the-box-plot-912f8d2acd49
"""
def other_other_stuff(file_path):
    key = os.path.basename(file_path).split(".")[0]
    name = "iqt2"

    df = pd.read_csv(file_path)
    df_pars = copy.deepcopy(df[[name, "difficulty"]])

    num_buckets = 50
    for i in range(len(df_pars["difficulty"])):
        df_pars.iat[i, 1] = math.floor(df_pars.iat[i, 1]*num_buckets) / num_buckets

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

    data_frames = [mean_val, median_val, val_25perc, val_75perc, val_5perc, val_95perc, max_llh, min_llh, count]
    merged_df = pd.concat(data_frames, join="outer", axis=1)

    print(merged_df)

    x = merged_df.index
    y = merged_df.mean_llh
    y_median = merged_df.median_llh
    y_25 = merged_df.llh_25perc
    y_75 = merged_df.llh_75perc
    y_5 = merged_df.llh_5perc
    y_95 = merged_df.llh_95perc

    fig, ax = plt.subplots(figsize=(6,4))
    #ax.plot(x, y, color="darkorchid", marker=".", label="mean")
    ax.plot(x, y_median, color="darkorchid", marker=".", label="median")
    ax.plot(x, y_25, color="darkorchid", linestyle="-.", label="25th %")
    ax.plot(x, y_75, color="darkorchid", linestyle="-.", label="75th %")
    ax.plot(x, y_5, color="goldenrod", linestyle="-.", label="5th %")
    ax.plot(x, y_95, color="goldenrod", linestyle="-.", label="95th %")
    ax.legend()

    ax.fill_between(x, y_25, y_75, alpha=0.1, color="darkorchid")
    ax.fill_between(x, y_5, y_95, alpha=0.1, color="goldenrod")

    ax.grid(alpha=0.2, which="both")
    ax.set_yscale("log")
    ax.set_xlabel("difficulty")
    ax.set_ylabel("")

    #df_pars.boxplot(column=["pars"], by=["difficulty"])
    #plt.yscale("log")
    plt.savefig(f"test_pd_{name}_{key}.png")
    plt.close()


def json_to_csv(file_path):
    with open(file_path) as file:
        dct = json.load(file)

    summed_dct = collections.defaultdict(lambda: [])
    for bucket in dct:
        diff_saved = False
        for name in dct[bucket]:
            for key in dct[bucket][name]:
                if key == "difficulty":
                    if not diff_saved:
                        summed_dct["difficulty"].extend(dct[bucket][name][key])
                        diff_saved = True
                else:
                    summed_dct[name].extend(dct[bucket][name][key])

    columns = list(summed_dct.keys())
    l = len(summed_dct[columns[0]])
    print(f"len {l}")
    out_lines = [",".join(columns)]
    for i in range(l):
        s = []
        for key in columns:
            s.append(str(summed_dct[key][i]))
        out_lines.append(",".join(s))

    out_path = os.path.join(os.path.dirname(file_path), os.path.basename(file_path).replace(".json", ".csv"))
    with open(out_path, "w+") as file:
        for line in out_lines:
            file.write(f"{line}\n")
    return out_path


def other_other_stuff_combined(file_path):
    key = os.path.basename(file_path).split(".")[0]
    csv_path = json_to_csv(file_path)

    names = ["pars", "raxml", "iqt2", "ft2"]
    num_buckets = 50
    df = pd.read_csv(csv_path)

    fig, axs = plt.subplots(int(len(names)/2), int(len(names)/2), figsize=(6*len(names)/2, 4*len(names)/2))
    fig.tight_layout(rect=[2, 0.06, 1, 0.95])

    shared_ax = None
    for j in range(len(names)):
        name = names[j]
        #ax = axs[int(j / (len(names)/2)), j % int(len(names)/2)]
        ax = plt.subplot(2, 2, j+1, sharex=shared_ax, sharey=shared_ax)

        df_pars = copy.deepcopy(df[[name, "difficulty"]])
        for i in range(len(df_pars["difficulty"])):
            df_pars.iat[i, 1] = math.floor(df_pars.iat[i, 1]*num_buckets) / num_buckets

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

        data_frames = [mean_val, median_val, val_25perc, val_75perc, val_5perc, val_95perc, max_llh, min_llh, count]
        merged_df = pd.concat(data_frames, join="outer", axis=1)

        print(merged_df)

        x = merged_df.index
        y = merged_df.mean_llh
        y_median = merged_df.median_llh
        y_25 = merged_df.llh_25perc
        y_75 = merged_df.llh_75perc
        y_5 = merged_df.llh_5perc
        y_95 = merged_df.llh_95perc

        #ax.plot(x, y, color="darkorchid", marker=".", label="mean")

        ax.plot(x, y_median, color="darkorchid", marker=".", label="median")
        ax.plot(x, y_25, color="darkorchid", linestyle="-.", label="25th %")
        ax.plot(x, y_75, color="darkorchid", linestyle="-.", label="75th %")
        ax.plot(x, y_5, color="goldenrod", linestyle="-.", label="5th %")
        ax.plot(x, y_95, color="goldenrod", linestyle="-.", label="95th %")

        if name=="raxml":
            ax.legend()

        ax.set_yscale("log")
        ax.set_title(f"{TOOL_NAME_DICT[name]}", fontsize=12)

        if j == 0:
            shared_ax = ax

        ax.fill_between(x, y_25, y_75, alpha=0.1, color="darkorchid")
        ax.fill_between(x, y_5, y_95, alpha=0.1, color="goldenrod")

        ax.grid(alpha=0.2, which="both")
        #ax.set_xlabel("difficulty")
        #ax.set_ylabel("")

    fig.supxlabel("difficulty", fontsize=16)
    fig.supylabel(f"{CAT_NAME_DICT[key]}", fontsize=16)

    #plt.tick_params(axis="y", which="minor")
    #plt.grid(True, which="both")

    #df_pars.boxplot(column=["pars"], by=["difficulty"])
    #plt.yscale("log")
    plt.savefig(f"test_pd_{key}.png")
    plt.close()


num_difficulty_buckets = 5
difficulty_buckets = [
    (x/num_difficulty_buckets, (x+1)/num_difficulty_buckets) for x in range(num_difficulty_buckets)
]
num_signal_buckets = 10
signal_bucket_unit = 2.5
signal_buckets = [(signal_bucket_unit * i, signal_bucket_unit * (i+1)) for i in range(0, num_signal_buckets)]
signal_buckets.append((num_signal_buckets * signal_bucket_unit, 100000))    # cover the rest by capping with huge number

contender_pairings = [
    ("raxml", "ft2"), ("raxml", "iqt2"), ("raxml", "pars"), ("iqt2", "ft2")
]


def main(root_dir, reuse=False):
    #other_other_stuff_combined(root_dir)
    #sys.exit()

    print(root_dir)
    if not reuse:
        do_stuff(root_dir)
    else:
        print("reusing stuff")
        do_other_stuff(root_dir)


if __name__ == "__main__":
    main(*sys.argv[1:])
