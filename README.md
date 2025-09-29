# PhyloSmew

The "Phylogenetic Snakemake evaluation workflow" is a pipeline to assess the accuracy of phylogenetic tree inference tools on 
empirical DNA MSAs from [TreeBASE](https://www.treebase.org/treebase-web/home.html) and simulated DNA MSAs based on 
[RAxMLGrove](https://github.com/angtft/RAxMLGrove) datasets. 


## Requirements
The scripts use the following packages and tools:
- Biopython
- matplotlib
- NumPy
- pandas
- scikit-learn (1.0.2)
- scikit-optimize
- Snakemake
- [tqDist](https://birc.au.dk/~cstorm/software/tqdist/) (for Quartet distances, if used)
- ete3
- consel
- dash (used in dash_app.py)
- plotly (used in dash_app.py)

Since we are currently including an older version of [Pythia](https://github.com/tschuelia/PyPythia), the version 1.0.2 of scikit-learn is actually mandatory. We recommend using a [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) environment.
Using conda, you can set up the environment with the following:
```
conda create --name phylosmew
conda activate phylosmew
conda install biopython matplotlib numpy pandas scikit-learn==1.0.2 snakemake tqdist ete3 scikit-optimize consel -c conda-forge -c bioconda
```

The dependencies for the dash app (for the visualisation of the results) can be installed using
```
conda install dash plotly statsmodels -c conda-forge
```


## Evaluated Tools

Currently, [RAxML-NG 1.1](https://github.com/amkozlov/raxml-ng), [IQ-TREE2](https://github.com/iqtree/iqtree2) are part of the pipeline and can be used for reference. New tools should be comparably easy to add by either adding the command in the "config.yaml" or (if the tool is somewhat difficult to use) by implementing a new class in the "inference_tools.py" which  needs to execute the inference (see for example "RAxMLPars(InferenceTool)" class) and return 
the path to the inferred tree. An instance of the newly implemented class can be added to the "tool_list" at the beginning of the Snakefile or specified in the config file. 


## Configuration File

In the "config.yaml" you can set the paths to the executables of RAxML-NG, IQ-TREE2 in the software category (if using non-default paths). 
Even if you want to exclude these tools from the analysis, RAxML-NG and IQ-TREE2 are needed for their functionality in the 
evaluation part of the pipeline. Then, you can set the "dataset selection criteria" (dsc) in the "data_sets" category. 

Further, the configuration file provides examples of inference tool, thread number and dataset specification.

## MSAs

In the current version, PhyloSmew can automatically select and download empirical MSAs from [TreeBASE](https://www.treebase.org/treebase-web/home.html) and simulated MSAs based on 
[RAxMLGrove](https://github.com/angtft/RAxMLGrove). In case a custom set of MSAs should be used, you can try to setup the directories in the following structure in ./out/:
```
- out/{dsc-name}/
          - {msa-id}
            - assembled_sequences.fasta	# MSA to be analyzed
            - partitions.txt			# (optional) RAxML-style partition file
            - tree_best.newick		# true tree used for simulations
```
Then, execute:
```
python scripts.py create_repr_files out/{dsc-name}
```

Alternatively, you can try to run 
```
python scripts.py copy_datasets {source-MSA-directory}
```
to set up the required structure automatically (in case your MSAs are already stored in an easy to understand structure -- refer to documentation of scripts.py copy_datasets()).


## Usage

After configuring the configs and implementing a new inference tool class (see above) simply run 
```
snakemake --cores {cores}
```
with appropriately set number of cores to be used. By default, the dsc in the config file is set to "smew_test", which can be used to quickly check if everything runs as it should.

Alternatively, 
```
snakemake --cores {cores} --config used_dsc="{dsc}"
```
can be used to select the used the dsc without changing the config file every time. The dsc still needs to be defined once in the config file though.

When the run is finished, you can generate some plots (like we used in the preprint) with 
```
python scripts.py create_plots out/{dsc}
```

Alternatively, you can try to use the Dash-app (see requirements!), by creating the .csv file for the dsc (if there are different directories named {dsc}_t{n} for different threads, they will all be saved in the same .csv) with
```
python scripts.py create_csv out/{dsc}
```
then starting the Dash-app, opening the displayed address (probably http://127.0.0.1:8050/) in your browser. Then, just drop the .csv in the box.
```
python dash_app.py
```


## Preprint

D. Höhler, J. Haag,  A. M. Kozlov, A. Stamatakis, (2022). 
**A representative Performance Assessment of Maximum Likelihood based Phylogenetic Inference Tools** 
*bioRxiv*.
[https://doi.org/10.1101/2022.10.31.514545](https://doi.org/10.1101/2022.10.31.514545)
