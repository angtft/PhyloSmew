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
- Snakemake
- [tqDist](https://birc.au.dk/~cstorm/software/tqdist/)


## Evaluated Tools

Currently, [RAxML-NG](https://github.com/amkozlov/raxml-ng), [IQ-TREE2](https://github.com/iqtree/iqtree2), 
and [FastTree2](http://www.microbesonline.org/fasttree/) are part of the pipeline and can be used for reference. New tools should be comparably easy 
to add by implementing a new class in the "inference_tooly.py" which would execute the inference (see for example "RAxMLPars(InferenceTool)" class) and return 
the path to the inferred tree. An instance of the newly implemented class should be added to the "tool_list" at the beginning of the Snakefile. 


## Configuration File

In the "config.yaml" one should set the paths to the executables of RAxML-NG, IQ-TREE2, and FastTree2 in the software category. 
Even if one wants to exclude these tools from the analysis, at least RAxML-NG and IQ-TREE2 are needed for their functionality in the 
evaluation part of the pipeline. Then, one should set the "dataset selection criteria" (dsc) in the "data_sets" category. 


## Usage

After configuring the configs and implementing a new inference tool class (see above) simply run 
```
snakemake --cores {cores}
```
with appropriately set number of cores to be used.

Alternatively, 
```
snakemake --cores {cores} --config used_dsc="{dsc}"
```
can be used to select the used the dsc without changing the config file. The dsc in the input still needs to be defined in the config though.

When the run is finished, one can generate some plots with 
```
python scripty.py create_plots out/{output_directory}
```
The output directory should be the named after the "used_dsc" in the config file.

In order to re-run the analysis, you can use the standard snakemake commands, e.g.
```
snakemake --delete-all-output
snakemake --cores {cores} --config used_dsc="{dsc}"
```


## Preprint

D. Höhler, J. Haag,  A. M. Kozlov, A. Stamatakis, (2022). 
**A representative Performance Assessment of Maximum Likelihood based Phylogenetic Inference Tools** 
*bioRxiv*.
[https://doi.org/10.1101/2022.10.31.514545](https://doi.org/10.1101/2022.10.31.514545)
