#!/usr/bin/env python3

import collections
import copy
import math
import os
import random
import subprocess
import sys
from typing import Optional, Dict, Any

import ete3
import numpy as np

sys.path.insert(1, os.path.join("libs", "RAxMLGroveScripts"))
import libs.RAxMLGroveScripts.org_script as rgs
import msa_parser

# ======================================================================================================================
# Benoit's stuff
def read_tree(tree):
    with open(tree) as file:
        lines = file.readlines()
    for line in lines:
        if not line.startswith(">"):
            return ete3.Tree(line, format=1)
    return None


def read_trees(tree):
    out_trees = []
    with open(tree) as file:
        lines = file.readlines()
    for line in lines:
        if not line.startswith(">"):
            out_trees.append(ete3.Tree(line, format=1))
    return out_trees


def _resolve(node):
    if len(node.children) > 2:
        children = list(node.children)
        random.shuffle(children)
        node.children = []
        next_node = root = node
        for i in range(len(children) - 2):
            next_node = next_node.add_child()
            next_node.dist = 1.0
            next_node.support = 0.0

        next_node = root
        for ch in children:
            next_node.add_child(ch)
            if ch != children[-2]:
                next_node = next_node.children[0]
        return True


def resolve_polytomies(tree):
    target = [tree]
    target.extend([n for n in tree.get_descendants()])
    multif = False
    for n in target:
        temp_res = _resolve(n)
        multif = multif or temp_res
    return multif


def make_binary(input_tree, output_tree, seed=0):
    if seed:
        random.seed(seed)

    trees = read_trees(input_tree)
    multif = False
    with open(output_tree, "w+") as file:
        for tree in trees:
            multif = multif or resolve_polytomies(tree)
            file.write(f"{tree.write(format=9)}\n")

    return multif
# ======================================================================================================================


def map_db_name(source):
    source = source.lower()
    name_map = {
        "tb": "tb_all.db",
        "treebase": "tb_all.db",
        "rg": "latest_all.db",
        "rgs": "latest_all.db"
    }
    if source not in name_map:
        raise ValueError(f"unknown db source: {source}")

    if source in name_map.values():
        return source
    else:
        return name_map[source]

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


def untar_file(file_path):
    file_dir = os.path.dirname(file_path)
    command = [
        "tar", "-xzf", f"{os.path.basename(file_path)}"
    ]
    subprocess.check_output(command, cwd=file_dir)


def download_dataset(tree_id, dest_dir, db_source="tb_all.db"):
    db_name = map_db_name(db_source)
    if db_name != "tb_all.db":
        raise ValueError(f"only TB database supported right now!")

    temp_query = f"TREE_ID = '{tree_id}'"
    temp_command = [
        "download",
        "-q", temp_query,
        "-n", db_name,
        "-o", os.path.dirname(dest_dir)
    ]
    _, msa_dirs = rgs.main(temp_command)

    msa_dir = msa_dirs[0]
    tar_path = os.path.join(msa_dir, f"{tree_id}.tar.gz")
    untar_file(tar_path)
    true_msa_path = os.path.join(msa_dir, "msa.fasta")
    true_msa_seqs = msa_parser.parse_msa_somehow(true_msa_path)
    out_msa_path = os.path.join(msa_dir, "assembled_sequences.fasta")

    # we do this because RAxML-NG (which is used for the best trees in the TB database) uses reduced alignments for
    # the tree inference, which messes up the sequence length report in its log file, which is then read and
    # saved in the RG database. so then len(true_msa) != len(reduced_msa). the partition files created by RGS will
    # contain the wrong per-partition sequence lengths, which seems to lead to segfaults in RAxML-NG (even for
    # single partitioned datasets).

    cleaned_seqs = clean_sequences(true_msa_seqs)
    no_empty_seqs = remove_empty_sites(cleaned_seqs)

    # We further remove duplicated sequences.
    # Different tools have usually different philosophies on how to manage those. At least for the scenarios and tools
    # we checked most so far (IQ-TREE, RAxML-NG, FastTree2), inclusion of duplicated sequences did not make much sense.
    no_empty_seqs = remove_duplicated_sequences(no_empty_seqs)

    # we further remove "empty" sequences (containing only undetermined characters)
    no_empty_seqs = remove_empty_sequences(no_empty_seqs)

    msa_parser.save_msa(no_empty_seqs, out_msa_path, msa_format="fasta")

    part_path = os.path.join(msa_dir, "sim_partitions.txt")
    if os.path.isfile(part_path):  # TODO: do this using tree_dict
        write_raxml_part_file(part_path, dsc_substitution_model, dsc_substitution_model_set)
        write_iqt_part_file(part_path, dsc_substitution_model, dsc_substitution_model_set)


def sim_dataset(tree_id, dest_dir, db_source="tb_all.db", use_bonk=False):
    db_name = map_db_name(db_source)

    temp_query = f"TREE_ID = '{tree_id}'"
    temp_command = [
        "generate",
        "-q", temp_query,
        "-n", db_name,
        "--generator", "alisim",
        "-o", dest_dir,
    ]
    if use_bonk:
        temp_command.append(
            "--use-bonk"
        )
    rgs.main(temp_command)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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


def read_order_file(path):
    names = []
    with open(path) as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            names.append(line)
    return names


def read_consel_name_file(path):
    names = []
    with open(path) as file:
        fl = True
        for line in file:
            line = line.strip()
            if fl:
                fl = False
                continue
            if not line:
                continue
            name = line.split()[0].replace("_slh", "").replace("slh", "")
            names.append(name)
    return names


def read_rf_file(path, name_list):
    rfs = {}
    with open(path) as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            line = line.split()
            idx1 = int(line[0])
            idx2 = int(line[1])
            rfs[f"rf_{name_list[idx1]}_{name_list[idx2]}"] = float(line[-1])
    return rfs


def read_rf_order_file(path):
    names = []
    with open(path) as file:
        for line in file:
            line = line.strip()
            if line.startswith("Reading input trees from file:"):
                temp_path = os.path.basename(line.split(":")[-1])
                names.append(temp_path.replace(".raxml.bestTree", "").replace("_eval", ""))
    return names


def parse_llh_file(path):
    dct = {}
    txt_dct = {}
    with open(path) as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            line = line.split()
            dct[f"llh_{line[0]}"] = float(line[2])
            dct[f"llh_{line[1]}"] = float(line[3])
            txt_dct[f"llh_{line[0]}"] = line[2].strip()
            txt_dct[f"llh_{line[1]}"] = line[3].strip()
    return dct, txt_dct


def read_diff_file(path):
    with open(path) as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            return float(line)


def read_ntd_file(path, name_list):
    name_list = [n.split(".")[0].split("_eval")[0] for n in name_list]
    values = []
    with open(path) as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            values.append([float(x) for x in line.split()])

    ntds = {}
    for i, n1 in enumerate(name_list):
        for j, n2 in enumerate(name_list[:i]):
            ntds[f"ntd_{n2}_{n1}"] = values[i-1][j]
    return ntds


def read_consel_test_results(path, names):
    dct = {}
    passed_tests = get_consel_test_results(path)
    for i, name in enumerate(names):
        dct[f"consel_{name}"] = passed_tests[i]
    return dct


def get_consel_test_results(path):
    passed_tests = collections.defaultdict(int)
    start = False
    with open(path) as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            split_line = line.split()
            if "rank" in line:
                start = True
                continue
            if not start:
                continue

            item = int(split_line[2]) - 1
            au_res = float(split_line[4])
            if au_res >= 0.05:
                passed_tests[item] += 1
    return passed_tests


def read_sim_part_file(part_path:str):
    """
    Expects partition file in format
        data_type, sub_model, part_name = start_index1-end_index1, ...
    e.g.,
        AA, JTT+G, partition_0 = 1-121

    Args:
        part_path: path to sim_partitions.txt as generated by RGS

    Returns:
        list of tuples (data_type, sub_model, part_name, rest)
    """

    lst = []
    with open(part_path) as file:
        for line in file:
            split_line = line.split("=")
            tmp = [s.strip() for s in split_line[0].split(",")]
            lst.append((tmp[0], tmp[1], tmp[2], split_line[1].strip()))
    return lst


def remove_empty_sites(seqs):
    """
    Removes sites that contain only undetermined characters "-", "?", "N".
    Args:
        seqs: MSA sequences (as returned by msa_parser.py)

    Returns:
        list of seqs without sites containing undetermined characters only
    """
    out_seqs = copy.deepcopy(seqs)
    empty_sites = collections.defaultdict(lambda: 0)
    for seq in seqs:
        for i in range(len(seq.sequence)):
            if seq.sequence[i].lower() in ["-", "?", "n"]:
                empty_sites[i] += 1
    for seq in out_seqs:
        temp_sequence = ""
        for i in range(len(seq.sequence)):
            if empty_sites[i] != len(out_seqs):
                temp_sequence += seq.sequence[i]
        seq.sequence = temp_sequence
    return out_seqs


def remove_empty_sequences(seqs):
    """
    Removes sequences that contain only undetermined characters "-", "?", "N".
    Args:
        seqs: MSA sequences (as returned by msa_parser.py)

    Returns:
        list of seqs without seqs containing undetermined characters only
    """
    out_seq = []
    for seq in seqs:
        num_undet = 0
        for char in seq.sequence:
            if char.lower() in ["-", "?", "n"]:
                num_undet += 1
        if num_undet < len(seq.sequence):
            out_seq.append(seq)
    return out_seq


def remove_duplicated_sequences(seqs):
    """
    Removes duplicated sequences from the MSA.
    Args:
        seqs: MSA sequences (as returned by msa_parser.py)

    Returns:
        seqs without duplicated sequences
    """
    out_seqs = []
    seq_dct = {}

    for seq in seqs:
        if seq.sequence not in seq_dct:
            seq_dct[seq.sequence] = 1
            out_seqs.append(seq)

    return out_seqs


def compute_ntd(tree1, tree2):
    tree1 = read_tree(tree1)
    tree2 = read_tree(tree2)
    name_dct = {}

    def get_node_dct(tree):
        node_dct = collections.defaultdict(list)
        def traverse(root):
            temp_list = []
            for node in root.children:
                if node.name:
                    name_dct[node.name] = 1
                    temp_list.append(node.name)
                temp_list.extend(traverse(node))
            node_dct[root] = dict([(n, 1) for n in temp_list])
            return temp_list

        traverse(tree)

        tmp_names = list(name_dct.keys())
        split_dct = {}

        for key in node_dct:
            tmp_dct = node_dct[key]
            tmp_tips = []
            for n in tmp_names:
                if n in tmp_dct:
                    tmp_tips.append(n)
            tmp_set = frozenset(tmp_tips)
            split_dct[tmp_set] = 1

        return node_dct, split_dct

    tree1_dct, split1_dct = get_node_dct(tree1)
    tree2_dct, split2_dct = get_node_dct(tree2)

    counter = 0
    for s1 in split1_dct:
        if s1 not in split2_dct:
            counter += 1
    print(counter)


def print_r_rfs(concat_trees_path):
    concat_trees_path = os.path.abspath(str(concat_trees_path))

    command = [
        "Rscript",
        "kfdist.r",
        concat_trees_path,
        "RF"
    ]

    out = subprocess.run(command, capture_output=True, text=True)
    print(out.stdout)


def print_r_kfs(concat_trees_path):
    concat_trees_path = os.path.abspath(str(concat_trees_path))

    command = [
        "Rscript",
        "kfdist.r",
        concat_trees_path,
    ]

    out = subprocess.run(command, capture_output=True, text=True)
    print(out.stdout)

    fl = True
    ret = []
    for line in str(out.stdout).split('\n'):
        if fl:
            fl = False
            continue
        if not line.strip():
            continue
        tmp = line.split()[1:]
        tmp = [float(v) for v in tmp]
        ret.append(tmp)
    return ret


def compute_pairwise_ntd(trees_path):
    name_dct = {}
    def get_node_dct(tree):
        node_dct = collections.defaultdict(list)
        def traverse(root, is_root=True):
            temp_list = []
            bl_sum = root.dist
            if root.name and root.name not in name_dct and "t" in root.name:
                name_dct[root.name] = 1

            if not root.children:
                temp_list = [root.name]

            for node in root.children:
                clist, cbl = traverse(node, is_root=False)
                temp_list.extend(clist)
                bl_sum += cbl

            bl = root.dist
            if is_root and len(root.children) == 2:
                for node in root.children:
                    bl += node.dist
                for node in root.children:
                    node_dct[node] = (node_dct[node][0], bl)
                node_dct[root] = (dict([(n, 1) for n in temp_list]), root.dist)
            else:
                node_dct[root] = (dict([(n, 1) for n in temp_list]), bl)

            return temp_list, bl_sum

        _, bl_sum = traverse(tree)

        tmp_names = list(name_dct.keys())
        split_dct = {}
        split_comb_list = []

        for key in node_dct:
            tmp_dct, tmp_bl = node_dct[key]
            tmp_tips = []
            tmp_rev_tips = []
            for n in tmp_names:
                if n in tmp_dct:
                    tmp_tips.append(n)
                else:
                    tmp_rev_tips.append(n)
            tmp_set = frozenset(tmp_tips)
            tmp_rev_set = frozenset(tmp_rev_tips)

            split_dct[tmp_set] = tmp_bl
            split_dct[tmp_rev_set] = tmp_bl
            split_comb_list.append((tmp_set, tmp_rev_set, tmp_bl))

        return node_dct, split_dct, split_comb_list, bl_sum

    def count_matching_splits(split1_dct, split2_dct, comb1, comb2, bl_sum1, bl_sum2):
        counter = 0
        bl_diff_sum = 0
        ntd = 0
        checked_splits = {}

        for s, rev, bl1 in comb1:
            if s in checked_splits or rev in checked_splits:
                continue

            bl2 = 0
            if s not in split2_dct or rev not in split2_dct:
                counter += 1
            else:
                bl2 = split2_dct[s]

            bl_diff_sum += (bl1 - bl2) ** 2
            ntd += abs(bl1/bl_sum1 - bl2/bl_sum2)

            checked_splits[s] = 1
            checked_splits[rev] = 1

        for s, rev, bl2 in comb2:
            if s in checked_splits or rev in checked_splits:
                continue

            bl1 = 0
            if s not in split1_dct or rev not in split1_dct:
                counter += 1
            else:
                bl1 = split1_dct[s]

            bl_diff_sum += (bl1 - bl2) ** 2
            ntd += abs(bl1 / bl_sum1 - bl2 / bl_sum2)

            checked_splits[s] = 1
            checked_splits[rev] = 1

        return counter, math.sqrt(bl_diff_sum), ntd/2

    trees_list = []
    with open(trees_path) as file:
        for line in file:
            trees_list.append(ete3.Tree(line, format=1))

    output1 = []
    for i, tree1 in enumerate(trees_list[:]):
        _, splits1, comb1, bl_sum1 = get_node_dct(tree1)
        tmp_out = []
        for j, tree2 in enumerate(trees_list[:i]):
            _, splits2, comb2, bl_sum2 = get_node_dct(tree2)
            score = count_matching_splits(splits1, splits2, comb1, comb2, bl_sum1, bl_sum2)
            tmp_out.append(score[2])

        if tmp_out:
            output1.append(tmp_out)

    def compare_outputs(o1, o2):
        for i, l1 in enumerate(o1):
            for j, val1 in enumerate(l1):
                val2 = o2[i][j]
                if val1 - val2 > 0.000000001:
                    print(f'{val1} != {val2}')

    for sub_list in output1:
        str_list = [str(val) for val in sub_list]
        print(f'{" ".join(str_list)}')

    return output1


def ntd_to_file(trees_path, out_path):
    ntd_list = compute_pairwise_ntd(trees_path)

    with open(out_path, "w+") as file:
        for sub_list in ntd_list:
            str_list = [str(val) for val in sub_list]
            file.write(f'{" ".join(str_list)}\n')


def clean_sequences(seqs):
    out_seqs = copy.deepcopy(seqs)
    for seq in out_seqs:
        seq.sequence = seq.sequence.replace("*", "-")
    return out_seqs


if __name__ == "__main__":
    globals()[sys.argv[1]](*sys.argv[2:])

