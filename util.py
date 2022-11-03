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

