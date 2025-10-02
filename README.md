# PhyloSmew

The "Phylogenetic Snakemake evaluation workflow" is a pipeline to assess the accuracy of phylogenetic tree inference tools on 
empirical DNA MSAs from [TreeBASE](https://www.treebase.org/treebase-web/home.html) and simulated DNA MSAs based on 
[RAxMLGrove](https://github.com/angtft/RAxMLGrove) datasets. 


---

## Overview
PhyloSmew automates end‑to‑end benchmarking of tree inference tools. It can:
- Automatically select and download MSAs, or ingest custom MSAs (empirical or simulated),
- Run a configurable set of inference tools with different thread counts,
- Compute statistics on the results, such as log‑likelihood differences, RF/NTD distances, quartet distances (optional), AU tests, and runtime
- Aggregate results and visualize them (plots or interactive Dash app).

The project includes small wrappers for RAxML‑NG, IQ‑TREE (v2 / v3), FastTree2, and VeryFastTree; new tools can be added declaratively via `config.yaml` or by implementing a custom class in `inference_tools.py`.

---

## Quick start
In any case, we recommend the usage of conda or mamba (you can get an installer at https://github.com/conda-forge/miniforge).

```bash
# 1) Clone (with submodules, required for RAxMLGroveScripts & PyPythia)
git clone --recursive https://github.com/angtft/PhyloSmew.git
cd PhyloSmew

# 2) (Recommended) create a fresh conda env
conda create -n phylosmew
conda activate phylosmew

# 3) Install runtime deps (CLI tools + Python libs)
#   scikit-learn==1.0.2 is required for the bundled PyPythia model
conda install -c conda-forge -c bioconda biopython matplotlib numpy pandas scikit-learn==1.0.2 snakemake tqdist ete3 scikit-optimize consel

# 4) Run the workflow (see config options below)
snakemake --cores 8 --config used_dsc="smew_test"       # "smew_test" is the name of a predefined test run in the 'config.yaml'

# 5) Aggregate results
python scripts.py create_csv out/smew_test

# (Optional) Dash app for interactive analysis
conda install -c conda-forge dash plotly statsmodels

# (Optional) Visualize (Dash app)
python dash_app.py  # then open the shown URL and drop the CSV into the uploader
```

> Tip: the default `used_dsc` in `config.yaml` is set to `smew_test` for a quick sanity check.

---

## Installation notes
- Executables for *RAxML‑NG* and *IQ‑TREE* are vendored via the `libs/RAxMLGroveScripts/tools` directory; you can override their paths in `config.yaml`.
- For quartet distances (optional rules), install *tqDist* and ensure `tqdist.command` points to `all_pairs_quartet_dist` (or install via conda and keep the config entry as is).

---

## Configuration (`config.yaml`)
Key sections:

### 1) `software`
Paths to core executables used for evaluation or auxiliary steps:
```yaml
software:
  raxml_ng:
    command: "libs/RAxMLGroveScripts/tools/raxml-ng_v1.1.0_linux_x86_64/raxml-ng"
  iqtree2:
    command: "libs/RAxMLGroveScripts/tools/iqtree-2.2.0-beta-Linux/bin/iqtree2"
  tqdist:       # optional, only if quartet distances are enabled
    command: "all_pairs_quartet_dist"
```

### 2) `tools`
Define how each inference tool is run. You can:
- Provide a complete command template (`command`) and an optional `command_partitioned` variant (used when a `partitions.txt` exists), **or**
- Specify an `inference_class` implemented in `inference_tools.py`.

Common placeholders available in `command` strings:
`{exe_path}`, `{msa_path}`, `{model}`, `{threads}`, `{prefix}`, `{part_file_path}`.
Note that the substitution model in the `{model}` placeholder might not be recognized by some inference tools (or mean something different, e.g., "GTR+G" for RAxML-NG 
should be "GTR+G+FO" for IQ-TREE2). In such cases you should implement a class in `inference_tools.py` and translate the model strings for the tools.

Example (abridged):
```yaml
tools:
  raxml1:
    path: "libs/RAxMLGroveScripts/tools/raxml-ng_v1.1.0_linux_x86_64/raxml-ng"
    command: "{exe_path} --msa {msa_path} --model {model} --threads {threads} --prefix {prefix} --force perf_threads"
    out_tree: "{prefix}.raxml.bestTree"

  iqtree3:
    path: "../iqtree-3.0.1-Linux/bin/iqtree3"
    inference_class: "IQTREE2"   # here, we use the implemented IQ-TREE2 class in 'inference_tools.py'; CLIs are compatible for our use here

  raxml2_fast:
    path: "../raxml-ng_v2.0/bin/raxml-ng"
    reference: "raxml1"
    add_flags: "--fast"       # concatenates "add_flags" to "command" and "command_partitioned" of the reference entry
```

You can also derive a tool from another via `reference` and append flags with `add_flags`.

### 3) Which tools & threads to run
```yaml
tool_list: ["raxml1", "iqtree2"]
num_threads: [4, 8]   # integer or list
```

### 4) Dataset selection criteria
Select which dataset group to process (also becomes the output directory name under `./out/`). The repo ships with examples such as `smew_test`.

```yaml
used_dsc: "smew_test"
```

---

## MSAs

In the current version, PhyloSmew can automatically select and download empirical MSAs from [TreeBASE](https://www.treebase.org/treebase-web/home.html) and simulate MSAs based on 
[RAxMLGrove](https://github.com/angtft/RAxMLGrove) (using the simulation tool [AliSim](https://github.com/iqtree/iqtree2/wiki/AliSim)). 
Predefined sets and examples are present in the `config.yaml`. In case a custom set of MSAs should be used, you can try the following:

### 1.) Create a custom dsc 
Create a new dsc for the custom MSA in the `data_sets` section of the `config.yaml` which should look like this:
```yaml
data_sets:
  # ...
  data_id: # some identifier
    source: "RGS_TB"                # set this to "RGS_TB", ignore the meaning (legacy stuff, will be renamed/redone in future)
    substitution_model: "GTR+G"     # set this!
```

### 2.) Set up the MSA directories
To set up the MSA directories, there are two options.

#### a.) Easy
In case the MSAs are stored in an easy to recognize structure, it might be sufficient to set the `custom_msas` value in the dsc definition,
such that it would look like this:
```yaml
data_sets:
  # ...
  dsc_name:                         # name it somehow
    source: "RGS_TB"                # set this to "RGS_TB", ignore the meaning (legacy stuff, will be renamed/redone in future)
    substitution_model: "GTR+G"     # set this!
    custom_msas: "path/to/msas/"    # set this!
```
This would run the `copy_datasets()` function in `scripts.py` to copy the MSAs. Thus, if something goes wrong, you can refer to the 
documentation and implementation there.

#### b.) Kind of annoying
You would need to set up the directories in the following structure in the `./out/` directory:

```
./out/<dsc-name>/<dataset-id>/
  ├─ assembled_sequences.fasta     # required
  ├─ partitions.txt                # optional, RAxML-style partition file
  └─ tree_best.newick              # optional (true tree for simulated datasets)
```

Then, execute:
```
python scripts.py create_repr_files out/{dsc-name}
```

---

## Preprint

D. Höhler, J. Haag,  A. M. Kozlov, A. Stamatakis, (2022). 
**A representative Performance Assessment of Maximum Likelihood based Phylogenetic Inference Tools** 
*bioRxiv*.
[https://doi.org/10.1101/2022.10.31.514545](https://doi.org/10.1101/2022.10.31.514545)
