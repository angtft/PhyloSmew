import os
import shutil
import subprocess

from util import *


class InferenceTool:
    def __init__(self, executable_path: str, prefix: str = None, **kwargs):
        self.executable_path = os.path.abspath(executable_path)
        self.prefix = prefix if prefix else self.__class__.__name__
        self.seed = None
        if "seed" in kwargs:
            self.seed = kwargs["seed"]

    def run_inference(self, msa_path: str, substitution_model: str = "GTR+G",
                      part_path: str = None, **kwargs) -> str:
        raise NotImplementedError(self.__class__.__name__)

    def get_prefix(self):
        return self.prefix


class RAxMLPars(InferenceTool):
    def run_inference(self, msa_path: str, substitution_model: str = "GTR+G", part_path: str = None,
                      num_pars: int = 1, **kwargs) -> str:
        msa_path = str(msa_path)
        msa_dir = os.path.dirname(msa_path)
        model_param = substitution_model

        part_path = os.path.join(msa_dir,"raxml_partitions.txt")
        if os.path.isfile(part_path):
            model_param = os.path.basename(part_path)

        command = [
            self.executable_path,
            "--start",
            "--msa", os.path.basename(msa_path),
            "--prefix", self.get_prefix(),
            "--tree", f"pars{{{num_pars}}}",
            "--model", model_param
        ]
        subprocess.run(command, cwd=msa_dir)
        return os.path.join(msa_dir,f"{self.get_prefix()}.raxml.startTree")


class RAxMLNG(InferenceTool):
    def run_inference(self, msa_path: str, substitution_model: str = "GTR+G", threads: int = 1,
                      num_pars: int = 10, num_rand: int = 10, **kwargs) -> str:
        msa_path = str(msa_path)
        msa_dir = os.path.dirname(msa_path)
        model_param = substitution_model

        part_path = os.path.join(msa_dir, "raxml_partitions.txt")
        if os.path.isfile(part_path):
            model_param = os.path.basename(part_path)

        command = [
            self.executable_path,
            "--msa", os.path.basename(msa_path),
            "--model", model_param,
            "--prefix", self.get_prefix(),
            "--threads", f"{threads}",
            "--force", "perf_threads"
        ]
        if num_pars != 10 or num_rand != 10:
            command.extend([
                "--tree", f"pars{{{num_pars}}},rand{{{num_rand}}}",
            ])

        subprocess.run(command, cwd=msa_dir)
        return os.path.join(msa_dir, f"{self.get_prefix()}.raxml.bestTree")


class BigRAxMLNG(RAxMLNG):
    def run_inference(self, msa_path: str, substitution_model: str = "GTR+G", threads: int = 1,
                      num_pars: int = 10, num_rand: int = 10, precomputed_tree_name: str = "", **kwargs) -> str:
        msa_dir = os.path.dirname(msa_path)
        if precomputed_tree_name:
            precomputed_tree_path = os.path.join(msa_dir, precomputed_tree_name)
            if os.path.isfile(precomputed_tree_path):
                print("big raxml tree already there! copying...")
                out_path = os.path.join(msa_dir, f"{self.get_prefix()}.raxml.bestTree")
                shutil.copy(precomputed_tree_path, out_path)
                return out_path

        return super(BigRAxMLNG, self).run_inference(msa_path, substitution_model=substitution_model, threads=threads, num_pars=50, num_rand=50, **kwargs)


class IQTREE2(InferenceTool):
    def run_inference(self, msa_path: str, substitution_model: str = "GTR+G", threads: int = 1, **kwargs) -> str:
        msa_path = str(msa_path)
        msa_dir = os.path.dirname(msa_path)
        msa_name = os.path.basename(msa_path)

        command = [
            self.executable_path,
            "-s", msa_name,
            "-m", substitution_model,
            "-nt", f"{threads}",
            "--prefix", self.get_prefix(),
        ]
        part_file = os.path.join(msa_dir, "iqt_partitions.txt")
        if part_file and os.path.isfile(part_file):
            command.extend([
                "-p", os.path.basename(part_file)
            ])
        subprocess.run(command, cwd=msa_dir)
        return os.path.join(msa_dir, f"{self.get_prefix()}.treefile")


class IQTREE2_BIONJ(InferenceTool):
    def run_inference(self, msa_path: str, substitution_model: str = "GTR+G", threads: int = 1, **kwargs) -> str:
        msa_path = str(msa_path)
        msa_dir = os.path.dirname(msa_path)
        msa_name = os.path.basename(msa_path)

        command = [
            self.executable_path,
            "-s", msa_name,
            "-m", substitution_model,
            "-nt", f"{threads}",
            "--prefix", self.get_prefix(),
            "-t", "BIONJ",
            "--tree-fix"
        ]
        part_file = os.path.join(msa_dir, "iqt_partitions.txt")
        if part_file and os.path.isfile(part_file):
            command.extend([
                "-p", os.path.basename(part_file)
            ])
        subprocess.run(command, cwd=msa_dir)
        return os.path.join(msa_dir, f"{self.get_prefix()}.treefile")


class FastTree2(InferenceTool):
    def get_model_flags(self, sub_model):
        allowed_models = ["JC", "GTR", "JTT", "LG", "WAG"]
        nuc_models = ["JC", "GTR"]
        allowed_mods = ["G", "CAT"]
        tmp = sub_model.split("+")
        flags = ""

        if tmp[0] not in allowed_models:
            raise ValueError(f"{sub_model} possibly not supported by FastTree2")
        else:
            flags += f"-{tmp[0].lower()}"
        if len(tmp) > 1:
            if tmp[1] not in allowed_mods:
                raise ValueError(f"{sub_model} possibly not supported by FastTree2")
            else:
                flags += f" -{tmp[1].lower()}"
        if tmp[0] in nuc_models:
            flags += f" -nt"

        return flags

    def run_inference(self, msa_path: str, substitution_model: str = "-gtr -gamma", part_path: str = None, **kwargs) -> str:
        msa_name = os.path.basename(msa_path)
        folder_path = os.path.dirname(msa_path)
        log_path = os.path.join(folder_path, f"{self.get_prefix()}.log")
        mf_out_path = os.path.join(folder_path, f"{self.get_prefix()}.mf.tree")
        bin_out_path = os.path.join(folder_path, f"{self.get_prefix()}.bin.tree")
        model = self.get_model_flags(substitution_model)
        command = f"{self.executable_path} {model} {msa_name}"

        try:
            proc = subprocess.Popen(command.split(), cwd=folder_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = proc.communicate()
        except Exception as e:
            raise e

        with open(mf_out_path, "wb+") as file:
            file.write(stdout)

        with open(log_path, "wb+") as file:
            file.write(stderr)

        make_binary(mf_out_path, bin_out_path)
        return bin_out_path
