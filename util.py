import collections
import copy
import random

import ete3

# ======================================================================================================================
# Benoit's stuff
def read_tree(tree):
    lines = open(tree).readlines()
    for line in lines:
        if not line.startswith(">"):
            return ete3.Tree(line, format=1)
    return None


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

    tree = read_tree(input_tree)
    tree.resolve_polytomy()
    tree = read_tree(input_tree)
    multif = resolve_polytomies(tree)
    with open(output_tree, "w") as writer:
        tree.write(outfile=output_tree)
    return multif
# ======================================================================================================================

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
    Removes sites that contain only undetermined characters.
    Args:
        seqs: MSA sequences (as returned by msa_parser.py)

    Returns:

    """
    out_seqs = copy.deepcopy(seqs)
    empty_sites = collections.defaultdict(lambda: 0)
    for seq in seqs:
        for i in range(len(seq.sequence)):
            if seq.sequence[i] in ["-", "?"]:
                empty_sites[i] += 1
    for seq in out_seqs:
        temp_sequence = ""
        for i in range(len(seq.sequence)):
            if empty_sites[i] != len(out_seqs):
                temp_sequence += seq.sequence[i]
        seq.sequence = temp_sequence
    return out_seqs

