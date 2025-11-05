# User plugin for PhyloSmew: extend/override inference tools here.
# This file is copied into your project on `phylosmew init`.
# It will be loaded automatically if present.

import os
import subprocess

# We import the packaged base module as `_base` and selectively re-expose what we need.
from phylosmew.legacy import inference_tools as _base

# Re-export base classes so users can subclass them right here:
InferenceTool   = _base.InferenceTool
RAxMLPars       = _base.RAxMLPars
RAxMLNG         = _base.RAxMLNG
BigRAxMLNG      = _base.BigRAxMLNG
IQTREE2         = _base.IQTREE2
IQTREE3_Fast    = _base.IQTREE3_Fast
IQTREE2_BIONJ   = getattr(_base, "IQTREE2_BIONJ", None)

# --- Example: customize behavior or add your own tool class -------------
# class MyIQTREE3(IQTREE2):
#     def run_inference(self, msa_path: str, substitution_model: str = "GTR+G", threads: int = 1, **kwargs) -> str:
#         # Example tweak: always enable --fast and add a seed flag
#         return super().run_inference(msa_path, substitution_model, threads=threads, **kwargs)

# NOTE on config.yaml usage:
# In your `tools:` block, set e.g.:
#   inference_class: "MyIQTREE3"
# ------------------------------------------------------------------------




# Further example which makes RAxML-NG infer 2000 trees (usage not recommended)
# add something like the following to the "tools" section of the "config.yaml"
# raxml2000:
#     path: "raxml-ng"
#     inference_class: "RAxMLNG2000"
class RAxMLNG2000(InferenceTool):
    def run_inference(self, msa_path: str, substitution_model: str = "GTR+G", threads: int = 2000,
                      num_pars: int = 2000, num_rand: int = 0, **kwargs) -> str:
        # cleaning up paths
        msa_path = str(msa_path)
        msa_dir = os.path.dirname(msa_path)
        model_param = substitution_model

        # use partition file, if available.
        # here you can also try to translate the substitution model for your tool
        part_path = os.path.join(os.path.abspath(msa_dir), "raxml_partitions.txt")
        if os.path.isfile(part_path):
            model_param = os.path.basename(part_path)

        # preparing the run command
        command = [
            self.executable_path,
            "--msa", os.path.basename(msa_path),
            "--model", model_param,
            "--prefix", self.get_prefix(),
            "--threads", f"{threads}",
            "--force",
            "--tree", f"pars{{{2000}}}"
        ]

        subprocess.run(command, cwd=msa_dir)
        # return the path to the inferred tree
        return self.get_out_tree_name(msa_dir)

    # Make this return the path to the inferred tree
    def get_out_tree_name(self, msa_dir):
        return os.path.join(msa_dir,f"{self.get_prefix()}.raxml.bestTree")





def resolve_tools(raw):
    """Re-export helper to keep compatibility if used elsewhere."""
    return _base.resolve_tools(raw)

def prepare_tools(raw):
    """
    Prepare and instantiate tools. We deliberately resolve classes from a
    merged namespace:
      - everything from the *base* module
      - everything defined in *this* user file

    That way, your new classes are available by name in config.yaml (inference_class).
    """
    tools = _base.resolve_tools(raw)
    out = {}

    # Merge base + user namespace for class lookup by name
    ns = dict(vars(_base))
    ns.update(globals())

    for tool_name, settings in tools.items():
        cls_name = settings.get("inference_class", "SimpleGeneric")
        try:
            cls = ns[cls_name]
        except KeyError:
            raise KeyError(f"inference_class '{cls_name}' not found "
                           f"(available: {sorted([k for k,v in ns.items() if isinstance(v, type)])})")
        out[tool_name] = cls(settings["path"], settings["prefix"], settings=settings)
    return out
