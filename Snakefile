import collections
import json
import itertools
import numpy as np
import os
import random
import shutil
import statistics
import sys
import traceback
from io import StringIO

import ete3
from Bio import Phylo, SeqIO, Seq, SeqRecord

sys.path.insert(1, os.path.join("libs", "RAxMLGroveScripts"))
sys.path.insert(1, os.path.join("libs", "PyPythia"))

import inference_tools
import libs.RAxMLGroveScripts.org_script as rgs
from libs.PyPythia.prediction_no_install import predict_difficulty as pythia_predict_difficulty
from util import *

configfile: "config.yaml"


BASE_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
dsc_substitution_model = "GTR+G"   # TODO: maybe set this somewhere else
current_dsc = config["data_sets"]["used_dsc"]
dsc_source = config["data_sets"][current_dsc]["source"]
dsc_sort_by = config["data_sets"]["sort_by"][dsc_source]
dsc_query = config["data_sets"][current_dsc]["query"]
dsc_num_points = config["data_sets"][current_dsc]["num_points"]
dsc_num_repeats = config["data_sets"][current_dsc]["num_repeats"]
dsc_filter_file = "broken_tb_msas.txt"                               # ids from this file will be removed from analysis

raxml_ng_path = os.path.abspath(config["software"]["raxml_ng"]["command"])
iqt2_path = os.path.abspath(config["software"]["iqtree2"]["command"])
fasttree2_path = os.path.abspath(config["software"]["fasttree2"]["command"])
tqdist_path = os.path.abspath(config["software"]["tqdist"]["command"])
pythia_predictor_path = os.path.join("libs", "PyPythia", "pypythia", "predictors", "predictor_sklearn_rf_v0.0.1.pckl")
rgs_db_path = os.path.abspath(os.path.join("libs", "RAxMLGroveScripts", "latest_all.db"))




# ======================================================================================================================
# The interesting part. Here one can add tools to the pipeline
out_dir = os.path.join("out", current_dsc)
tool_list = [
    inference_tools.RAxMLPars(raxml_ng_path, prefix="pars"),
    inference_tools.RAxMLNG(raxml_ng_path, prefix="raxml"),
    inference_tools.IQTREE2(iqt2_path, prefix="iqt2"),
    inference_tools.FastTree2(fasttree2_path, prefix="ft2")
]
if dsc_source == "TB":
    tool_list.append(inference_tools.BigRAxMLNG(raxml_ng_path, prefix="bigraxml", check_existing="tree_best.newick"))

tools_dict = dict([(t.get_prefix(), t) for t in tool_list])
# ======================================================================================================================





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


def read_filter_file(file_path: str):
    data_map = {}
    with open(file_path) as file:
        for line in file:
            line = line.strip()
            data_map[line] = 1
    return data_map


def untar_file(file_path):
    file_dir = os.path.dirname(file_path)
    command = [
        "tar", "-xzf", f"{file_path}"
    ]
    subprocess.check_output(command, cwd=file_dir)


def select_representatives(out_file: str, source: str, query: str, sort_by: str = dsc_sort_by,
                           filter_file: str = "", num_splits=dsc_num_points):
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
        command.extend([
            "-n", "tb_dna.db",       # TODO: change once AA is supported
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
    if filter_file:
        ignored_msas = read_filter_file(filter_file)
    sorted_results = list(filter(lambda x: x["TREE_ID"] not in ignored_msas, sorted_results))

    result_dict = {}
    l = len(sorted_results)
    for i in range(num_splits):
        temp_range = sorted_results[int(l * i / num_splits): int(l * (i + 1) / num_splits)]
        result_dict[i] = temp_range

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


def select_dataset(repr_file: str, target_file: str, data_set_num: int):
    target_file = str(target_file)
    with open(repr_file) as file:
        representatives = json.load(file)
    temp_repr = random.choice(representatives[data_set_num])
    tree_id = temp_repr["TREE_ID"]

    create_dir_if_needed(os.path.dirname(target_file))
    with open(target_file, "w+") as file:
        file.write(f"{tree_id}")


def prepare_rgs_dataset(repr_path: str, msa_path: str, source: str):
    repr_path = os.path.abspath(str(repr_path))
    msa_path = os.path.abspath(str(msa_path))
    msa_dir = os.path.dirname(msa_path)
    exp_dir = os.path.dirname(msa_dir)

    repr_id = "-1"
    with open(repr_path) as file:
        for line in file:
            repr_id = line.strip()
            break
    query = f"TREE_ID = '{repr_id}'"

    generate_command = [
        "justgimmeatree",
        "-q", query,
        "-o", exp_dir,
        "--generator", "alisim"
    ]
    if source == "RGS":
        db_path = "latest.db"
        if os.path.isfile(rgs_db_path):
            db_path = rgs_db_path

        generate_command.extend([
            "-n", db_path,
            "--use-bonk",
            "--insert-matrix-gaps",
            "--avoid-empty-sequences"
        ])
    elif source == "TB":
        generate_command.extend([
            "-n", "tb_dna.db",          # TODO: change once AA is supported
            "--no-simulation",
            #"--use-local-db", TB_MIRROR_PATH
        ])
    elif source == "RGS_TB":
        generate_command.extend([
            "-n", "tb_dna.db",          # TODO: change once AA is supported
            "--use-bonk",
            "--insert-matrix-gaps",
            "--avoid-empty-sequences",
            #"--use-local-db", TB_MIRROR_PATH
        ])
    else:
        raise ValueError(f"unknown data source: {source}")

    # download and simulate file
    _, _ = rgs.main(generate_command)
    if source == "TB":
        tar_path = os.path.join(msa_dir, f"{repr_id}.tar.gz")
        untar_file(tar_path)
        true_msa_path = os.path.join(msa_dir, "msa.fasta")
        shutil.copy(true_msa_path,os.path.join(msa_dir, "assembled_sequences.fasta"))

    part_path = os.path.join(msa_dir, "sim_partitions.txt")
    if os.path.isfile(part_path):
        write_raxml_part_file(part_path, dsc_substitution_model)

    # compute difficulty
    """predictor = DifficultyPredictor(open(pythia_predictor_path, "rb"))
    raxmlng = PythiaRAxMLNG(raxml_ng_path)
    msa = MSA(os.path.join(msa_dir, "assembled_sequences.fasta"))

    msa_features = get_all_features(raxmlng, msa, dsc_substitution_model)
    difficulty = predictor.predict(msa_features)"""
    difficulty = pythia_predict_difficulty(os.path.join(msa_dir, "assembled_sequences.fasta"),
        pythia_predictor_path, raxml_ng_path)

    with open(os.path.join(msa_dir, "difficulty"), "w+") as file:
        file.write(f"{difficulty}\n")


def run_raxml_evaluate(msa_path: str, tree_path: str, model: str, prefix: str, threads: int = 1,
                       redo: bool = True, retry: int = 0):
    new_prefix = f"{prefix}"
    folder_path = os.path.dirname(msa_path)
    tree_name = os.path.basename(tree_path)
    msa_name = os.path.basename(msa_path)
    command = f"{raxml_ng_path} --evaluate --msa {msa_name} --model {model} --tree {tree_name} --prefix {new_prefix} " \
              f"--threads {threads} --force perf_threads"
    if redo:
        command += " --redo"

    try:
        out = subprocess.run(command.split(), cwd=folder_path)
    except Exception as e:
        print(e)

    # in case it went wrong because the tree was multifurcated
    if not retry and not os.path.isfile(os.path.join(folder_path, f"{new_prefix}.raxml.bestTree")):
        new_tree_path = os.path.join(folder_path, f"temp_bin_{new_prefix}.newick")
        make_binary(tree_path, new_tree_path)
        run_raxml_evaluate(msa_path, new_tree_path, model, prefix, threads=threads, redo=redo, retry=1)


def compute_rfs(true_tree_path: str, tree_paths: list[str], prefix: str = "rf"):
    tree_paths = list(map(lambda p: os.path.abspath(str(p)), tree_paths))
    tree_paths = [os.path.abspath(str(true_tree_path))] + tree_paths
    tree_dir = os.path.dirname(tree_paths[0])

    command = [
        raxml_ng_path, "--rfdist",
        "--tree", f"{','.join(tree_paths)}",
        "--redo"
    ]
    if prefix:
        command.extend([
            "--prefix", f"{prefix}"
        ])
    subprocess.run(command, cwd=tree_dir)


def compute_llh_diffs(true_tree_path: str, log_paths: list[str], out_path: str):
    log_paths = [os.path.abspath(str(p)) for p in log_paths]
    log_paths = [os.path.abspath(str(true_tree_path))] + log_paths

    names = [get_tree_name_from_base_name(os.path.basename(path)) for path in log_paths]

    llhs = {}
    out_lines = []
    for i in range(len(names)):
        llhs[names[i]] = get_llh_from_log(log_paths[i])

    for i in range(len(names) - 1):
        llh1 = llhs[names[i]]
        for j in range(i + 1, len(names)):
            llh2 = llhs[names[j]]
            diff = llh1 - llh2
            out_lines.append(f"{names[i].ljust(10)} {names[j].ljust(10)} {llh1} {llh2} {str(diff/llh1).ljust(10)} {diff}\n")

    with open(out_path, "w+") as file:
        for line in out_lines:
            file.write(line)


def concat_tree_files(file_names: list[str], out_path: str):
    with open(out_path, 'w+') as outfile:
        for fname in file_names:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)


def get_tree_name_from_base_name(base_name: str) -> str:
    if "_" in base_name:
        return base_name.split("_")[0]
    else:
        return base_name.split(".")[0]


def run_iqt2_topology_test(true_tree_path: str, tree_paths: list[str], msa_path: str, concat_path: str, prefix: str = "top_test",
                           threads: int = 1, substitution_model: str = dsc_substitution_model):
    tree_paths = [os.path.abspath(str(p)) for p in tree_paths]
    tree_paths = [os.path.abspath(str(true_tree_path))] + tree_paths

    concat_path = os.path.abspath(str(concat_path))
    root_path = os.path.dirname(concat_path)
    msa_path = os.path.abspath(str(msa_path))

    concat_tree_files(tree_paths, concat_path)
    names_file = os.path.join(root_path, f"{prefix}.names")
    part_file = os.path.join(root_path, "sim_partitions.txt")
    names = [get_tree_name_from_base_name(os.path.basename(p)) for p in tree_paths]
    with open(names_file, "w+") as file:
        for name in names:
            file.write(f"{name}\n")

    command = [
        iqt2_path,
        "-s", msa_path,
        "-m", substitution_model,
        "-n", "0",
        "-z", concat_path,
        "-zb", "10000",
        "-au",
        "-zw",
        "-redo",
        "-nt", f"{threads}",
        "--prefix", prefix
    ]
    if os.path.isfile(part_file):
        command.extend([
            "-p", os.path.basename(part_file)
        ])

    try:
        out = subprocess.run(command, cwd=root_path, check=True, stdout=subprocess.PIPE).stdout
    except Exception as e:
        print(f"Exception in iqt_topology_test: {e}")


def compute_quartet_dists(concat_trees_path: str, out_path: str):
    concat_trees_path = str(concat_trees_path)
    out_path = str(out_path)

    command = [
        tqdist_path, concat_trees_path
    ]

    output = subprocess.check_output(command)
    with open(out_path, "wb+") as file:
        file.write(output)


def write_raxml_part_file(part_path: str, substitution_model: str):
    msa_dir = os.path.dirname(part_path)
    part_out_path = os.path.join(msa_dir, "raxml_partitions.txt")
    part_lines = []
    with open(part_path) as file:
        for line in file:
            part_lines.append(line)
    with open(part_out_path,"w+") as file:
        for line in part_lines:
            file.write(f"{substitution_model}, {line}")


class TrueTree(inference_tools.InferenceTool):
    def run_inference(self, msa_path: str, substitution_model: str = "GTR+G", part_path: str = None, dsc_source: str = "", **kwargs) -> str:
        if not dsc_source:
            raise ValueError("dsc not specified!")

        msa_dir = os.path.dirname(str(msa_path))
        if dsc_source == "RGS":
            tree_path = os.path.join(msa_dir, "tree_best.newick")
            run_raxml_evaluate(msa_path, tree_path, substitution_model, "true", threads=1)
        else:
            max_llh = "None"
            max_prefix = ""
            trees = kwargs.pop("trees")
            for tree_path in trees:
                tree_path = os.path.abspath(str(tree_path))
                basename = os.path.basename(tree_path)
                prefix = basename.split(".")[0]
                log_path = os.path.join(msa_dir, f"{prefix}.raxml.log")
                llh = get_llh_from_log(log_path)

                if max_llh == "None" or max_llh < llh:
                    max_llh = llh
                    max_prefix = prefix

            print(f"highest llh tree: {max_prefix}")
            for fn in os.listdir(msa_dir):
                if fn.startswith(f"{max_prefix}.raxml"):
                    file_path = os.path.join(msa_dir, fn)
                    suffix = fn.split(".")[-1]
                    target_path = os.path.join(msa_dir, f"true.raxml.{suffix}")

                    shutil.copy(file_path, target_path)

            return os.path.join(msa_dir, "true.raxml.bestTree")



rule all:
    input:
        expand("{out_dir}/{data_set_num}/msa_{msa_num}/default/rf.raxml.rfDistances",
            out_dir=[out_dir],
            data_set_num=list(range(dsc_num_points)),
            msa_num=list(range(dsc_num_repeats))),
        expand("{out_dir}/{data_set_num}/msa_{msa_num}/default/top_test.iqtree",
           out_dir=[out_dir],
           data_set_num=list(range(dsc_num_points)),
            msa_num=list(range(dsc_num_repeats))),
        expand("{out_dir}/{data_set_num}/msa_{msa_num}/default/quartet_dists.txt",
           out_dir=[out_dir],
           data_set_num=list(range(dsc_num_points)),
            msa_num=list(range(dsc_num_repeats))),
        expand("{out_dir}/{data_set_num}/msa_{msa_num}/default/llh_diffs",
            out_dir=[out_dir],
            data_set_num=list(range(dsc_num_points)),
            msa_num=list(range(dsc_num_repeats))),

rule select_representatives:
    output:
        representatives = "{out_dir}/representatives_all.txt"
    run:
        select_representatives(output.representatives, dsc_source, dsc_query, sort_by=dsc_sort_by,
                               filter_file=dsc_filter_file, num_splits=dsc_num_points)

rule select_dataset:
    input:
        representatives = "{out_dir}/representatives_all.txt"
    output:
        id_file = "{out_dir}/{data_set_num}/tree_id"
    run:
        select_dataset(input.representatives, output.id_file, wildcards.data_set_num)

rule setup_dataset:
    input:
        id_file = "{out_dir}/{data_set_num}/tree_id"
    output:
        msa = "{out_dir}/{data_set_num}/msa_{msa_num}/default/assembled_sequences.fasta",
        difficulty = "{out_dir}/{data_set_num}/msa_{msa_num}/default/difficulty"
    run:
        prepare_rgs_dataset(input.id_file, output.msa, dsc_source)

rule run_inference:
    input:
        msa = "{out_dir}/{data_set_num}/msa_{msa_num}/default/assembled_sequences.fasta"
    output:
        tree = "{out_dir}/{data_set_num}/msa_{msa_num}/default/{prefix}_eval.raxml.bestTree",
        log = "{out_dir}/{data_set_num}/msa_{msa_num}/default/{prefix}_eval.raxml.log"
    threads: 4      # TODO: set this somewhere else
    run:
        inf_tree = tools_dict[wildcards.prefix].run_inference(input.msa, substitution_model=dsc_substitution_model,
            threads=threads)
        run_raxml_evaluate(input.msa, inf_tree, dsc_substitution_model, f"{wildcards.prefix}_eval")

rule find_true_tree:
    input:
        trees = expand("{{out_dir}}/{{data_set_num}}/msa_{{msa_num}}/default/{prefix}_eval.raxml.bestTree", prefix=[t.get_prefix() for t in tool_list]),
        msa = "{out_dir}/{data_set_num}/msa_{msa_num}/default/assembled_sequences.fasta"
    output:
        true_tree = "{out_dir}/{data_set_num}/msa_{msa_num}/default/true.raxml.bestTree",
        log = "{out_dir}/{data_set_num}/msa_{msa_num}/default/true.raxml.log"
    threads: 4  # TODO: set this somewhere else
    run:
        inf_tree = TrueTree("", prefix="true").run_inference(input.msa, trees=input.trees, substitution_model=dsc_substitution_model, threads=threads, dsc_source=dsc_source)

rule compute_rfs:
    input:
        trees = expand("{{out_dir}}/{{data_set_num}}/msa_{{msa_num}}/default/{prefix}_eval.raxml.bestTree", prefix=[t.get_prefix() for t in tool_list]),
        true_tree= "{out_dir}/{data_set_num}/msa_{msa_num}/default/true.raxml.bestTree",
    output:
        rfs = "{out_dir}/{data_set_num}/msa_{msa_num}/default/rf.raxml.rfDistances"
    run:
        compute_rfs(input.true_tree, input.trees, prefix="rf")

rule compute_llh_diffs:
    input:
        logs = expand("{{out_dir}}/{{data_set_num}}/msa_{{msa_num}}/default/{prefix}_eval.raxml.log", prefix=[t.get_prefix() for t in tool_list]),
        true_tree_log = "{out_dir}/{data_set_num}/msa_{msa_num}/default/true.raxml.log",
    output:
        llh_diffs = "{out_dir}/{data_set_num}/msa_{msa_num}/default/llh_diffs"
    run:
        compute_llh_diffs(input.true_tree_log, input.logs, output.llh_diffs)

rule compute_tops:
    input:
        trees = expand("{{out_dir}}/{{data_set_num}}/msa_{{msa_num}}/default/{prefix}_eval.raxml.bestTree", prefix=[t.get_prefix() for t in tool_list]),
        true_tree= "{out_dir}/{data_set_num}/msa_{msa_num}/default/true.raxml.bestTree",
        msa = "{out_dir}/{data_set_num}/msa_{msa_num}/default/assembled_sequences.fasta"
    output:
        concat_trees = "{out_dir}/{data_set_num}/msa_{msa_num}/default/concat.bestTrees",
        test_results = "{out_dir}/{data_set_num}/msa_{msa_num}/default/top_test.iqtree"
    threads: 1
    run:
        run_iqt2_topology_test(input.true_tree, input.trees, input.msa, output.concat_trees, prefix="top_test", threads=threads,
                               substitution_model=dsc_substitution_model)

rule compute_quartet_dists:
    input:
        concat_trees = "{out_dir}/{data_set_num}/msa_{msa_num}/default/concat.bestTrees"
    output:
        quartet_dists = "{out_dir}/{data_set_num}/msa_{msa_num}/default/quartet_dists.txt"
    run:
        compute_quartet_dists(input.concat_trees, output.quartet_dists)

