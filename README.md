# PhyloSmew

The "Phylogenetic Snakemake evaluation workflow" is a pipeline to assess the accuracy of phylogenetic tree inference tools on 
empirical DNA MSAs from [TreeBASE](https://www.treebase.org/treebase-web/home.html) and simulated DNA MSAs based on 
[RAxMLGrove](https://github.com/angtft/RAxMLGrove) datasets. 


---

## Overview
PhyloSmew automates end‑to‑end benchmarking of tree inference tools. It can:
- Automatically select and download MSAs, or ingest custom MSAs (empirical or simulated).
- Run a configurable set of inference tools with different thread counts.
- Compute statistics on the results, such as log‑likelihood differences, RF- and NT-distances, quartet distances (optional), AU tests, and runtime.
- Aggregate results and visualize them (plots or interactive Dash app).

The project includes small wrappers for RAxML‑NG, IQ‑TREE (v2 / v3), FastTree2, and VeryFastTree; new tools can be added declaratively via `config.yaml` or by implementing a custom class in `inference_tools.py`.

---

## Quick start
On any OS, we recommend the usage of conda or mamba (you can get an installer at https://github.com/conda-forge/miniforge). 

> Note that [RAxML-NG](https://github.com/amkozlov/raxml-ng) is needed for the evaluation part of the pipeline. There is no 
Windows binary available; the easiest way to run everything would be by using [WSL](https://learn.microsoft.com/en-us/windows/wsl/install).
For macOS there is a RAxML-NG binary, but you need to download it and set the path in the `config.yaml`.


### 1.) Clone (with submodules, required for RAxMLGroveScripts & PyPythia)
```bash
git clone --recursive https://github.com/angtft/PhyloSmew.git
cd PhyloSmew
```

### 2.) (Recommended) Create a fresh conda environment
```bash
conda create -n phylosmew
conda activate phylosmew
```

### 3.) Install runtime dependencies (CLI tools + Python libs)
> Note: scikit-learn==1.0.2 is required for the bundled PyPythia model
```bash
conda install -c conda-forge -c bioconda biopython matplotlib numpy pandas scikit-learn==1.0.2 snakemake tqdist ete3 scikit-optimize consel ebg
```

### 4.) Run the workflow (see config options below)
```bash
snakemake --cores 8 --config used_dsc="smew_test"       # "smew_test" is the name of a predefined test run in the 'config.yaml'
```

### 5.) Aggregate results
```bash
python scripts.py make_csv out/smew_test
```

### (Optional) Dash app for interactive analysis
```bash
conda install -c conda-forge dash plotly statsmodels
```

### (Optional) Visualize (Dash app)
```bash
python dash_app.py 
```
then open the printed URL and upload the CSV.

> Tip: the default `used_dsc` in `config.yaml` is set to `smew_test` for a quick sanity check.

---

## Installation notes
- Linux executables for *RAxML‑NG* and *IQ‑TREE* are vendored via the `libs/RAxMLGroveScripts/tools` directory; you can override their paths in `config.yaml`.
- For quartet distances (optional rules), install *tqDist* and ensure `tqdist.command` points to `all_pairs_quartet_dist` (or install via conda and keep the config entry as is).

---

## Configuration (`config.yaml`)
Most likely you will need to touch at least some of these. Key sections:

### 1.) `software`
Paths to core executables used for evaluation or auxiliary steps:
```yaml
software:
  raxml_ng:
    command: "libs/RAxMLGroveScripts/tools/raxml-ng_v1.1.0_linux_x86_64/raxml-ng"
  iqtree2:
    command: "libs/RAxMLGroveScripts/tools/iqtree-2.2.0-beta-Linux/bin/iqtree2"
  tqdist:       # optional, only if quartet distances are uncommented in the rule "all" in the Snakefile
    command: "all_pairs_quartet_dist"
```

### 2.) `tools`
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

### 3.) Which tools and threads to run
```yaml
tool_list: ["raxml1", "iqtree2"]
num_threads: [4, 8]   # integer or list
```

### 4.) Dataset selection criteria
Select which dataset group to process (also becomes the output directory name under `./out/`). The repo ships with examples such as `smew_test`.

```yaml
used_dsc: "smew_test"
```

### 5.) Optional statistics
There are some optional statistics that have been disabled by default in the Snakefile, which might be interesting to some. 
To get them to work, you would need to uncomment the corresponding line in the rule "all" in the `Snakefile`.

#### a.) Quartet distances
Self-explanatory: You can compute quartet distances between all inferred trees using tqDist (just uncomment the line). 

#### b.) Predicted bootstrap support using EBG
EBG is "a machine learning-based tool that predicts SBS branch support values for a given input phylogeny" (https://doi.org/10.1093/molbev/msae215). 
You can uncomment the EBG support line in the `Snakefile` to run [EBG](https://github.com/wiegertj/EBG). 
This will create a directory for every dataset and tool containing the inferred tree with assigned predicted branch support values:
```
./out/<dsc-name>/<dataset-id>/ebg_<tool-name>/
  ├─ tmp                                                     # contains parsimony trees used for predictions
  ├─ ebg_<tool-name>_median_support_prediction.newick        # tree with assigned support values
  └─ ...
```
Currently, we only compute the mean and median branch supports per tool, which is why we disabled the EBG by default. 
The execution time is dominated by the creation of 1200 parsimony starting-trees with RAxML-NG. Including the tree inference, 
EBG is on average about 9.4 times faster than [UFBoot2](https://doi.org/10.1093/molbev/msx281). Since we would have the 
inferred trees already, running EBG should be pretty cheap. The mean absolute error of predictions is 5 (in range 0 to 100).

---

## MSAs

In the current version, PhyloSmew can automatically select and download empirical MSAs from [TreeBASE](https://www.treebase.org/treebase-web/home.html) and simulate MSAs based on 
[RAxMLGrove](https://github.com/angtft/RAxMLGrove) (using the simulation tool [AliSim](https://github.com/iqtree/iqtree2/wiki/AliSim)). 
Predefined sets and examples are present in the `config.yaml`. In case a custom set of MSAs should be used, you can try the following:

### 1.) Create a custom dataset configuration/selection criteria (dsc) 
Create a new dsc for the custom MSA in the `data_sets` section of the `config.yaml` which should look like this:
```yaml
data_sets:
  # ...
  dsc_name:                         # name it somehow
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

#### b.) Manual setup
You would need to set up the directories in the following structure in the `./out/` directory:

```
./out/<dsc-name>/<dataset-id>/
  ├─ assembled_sequences.fasta     # required
  ├─ partitions.txt                # optional, RAxML-style partition file
  └─ tree_best.newick              # optional (true tree for simulated datasets)
```

Then, execute:
```bash
python scripts.py create_repr_files out/{dsc-name}
```

---

## Results

The results of the pipeline execution can be collected into a `.csv` file using
```bash
python scripts.py make_csv out/{dsc-name}
```
The CSV should contain the [Pythia](https://doi.org/10.1093/molbev/msac254)-difficulty of the MSA, 
log-likelihood differences to "true" tree (or "best-known" tree, if no true tree available), RF- and NT-distances to true tree, 
statistical tests implemented in [CONSEL](https://github.com/shimo-lab/consel) (AU, KH, SH, wKH, wSH), 
and execution times.

You can run the Dash app for some simple visualization of the results.
```bash
python dash_app.py 
```

---

## Simple use cases

### 1. "Test run"
For a simple test run, refer to the "Quick start" above. If everything is installed, you can simply run (assuming you have 8 cores):
```bash
snakemake --cores 8 --config used_dsc="smew_test" num_threads=[1,4]
```
This should start a run with 231 jobs, which should finish quite fast, since the datasets are pretty small. 
You should find two directories in "./out/": "smew_test_t1" (contains runs using 1 thread) and "smew_test_t4" (contains runs using 4 threads).

### 2. "Recycling"
In case you're interested in using our precomputed inferences from the preprint, including RAxML-NG 1.1, IQ-TREE 2, FastTree 2, 
BIONJ (as implemented in IQ-TREE 2), and parsimony starting trees (from RAxML-NG 1.1), and compare your set of tools 
against them, you can do the following:

#### a.) Download our datasets
Download our archive from https://cme.h-its.org/exelixis/material/accuracy-study/data.tar.gz (30GB!). There should be at least two separate 
archives containing the 5k TreeBASE datasets (`tb_5k_dna.tar.gz`) and the 20k simulated datasets using RG and AliSim (`sim_20k_dna.tar.gz`). 
Extract those in the ./out/ directory.

#### b.) Setup config.yaml
Set the `tool_list` to ["raxml", "iqt2", "ft2", "pars", "bionj"] (or to whichever tools you'd like to include from our runs), 
define your own tools (see above) and add your own tools to this list. Set the `used_dsc` according to the name of the extracted directory.

#### c.) Delete existing evaluation files
Most of the evaluation is bound to knowing the "best" tree. Thus, you'd need to delete the already computed evaluation files using 
```bash
python scripts.py reset_evaluation out/{datasets}
```

### 3. "Bring Your Own Dataset"
In case you'd like to use your own datasets, refer to the description in "MSAs" above. For example, you could use this to 
create MSAs with different aligners and investigate their impact on the final tree inference.

---

## Preprint

D. Höhler, J. Haag,  A. M. Kozlov, A. Stamatakis, (2022). 
**A representative Performance Assessment of Maximum Likelihood based Phylogenetic Inference Tools** 
*bioRxiv*.
[https://doi.org/10.1101/2022.10.31.514545](https://doi.org/10.1101/2022.10.31.514545)
