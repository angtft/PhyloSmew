#!/usr/bin/env python3
import collections
from Bio import Phylo, SeqIO, Seq, SeqRecord


"""
Our cheap MSA parser stuff 
"""
class MSAParser:
    def __init__(self):
        pass

    def parse(self, msa_path):
        return []

    def parse_and_fix(self, msa_path, data_type):
        sequences = self.parse(msa_path)
        if data_type != "DNA":
            return sequences

        for sequence in sequences:
            sequence.sequence = sequence.sequence.upper()
            sequence.sequence = sequence.sequence.replace("N", "-")
            sequence.sequence = sequence.sequence.replace("?", "-")
            sequence.sequence = sequence.sequence.replace("X", "-")
            sequence.sequence = sequence.sequence.replace("U", "T")
        return sequences

    def __str__(self):
        return type(self).__name__


class Sequence:
    def __init__(self, id, sequence, comment=""):
        self.id = id
        self.sequence = sequence
        self.comment = comment


class PhylipsParser(MSAParser):
    def parse(self, msa_path):
        with open(msa_path) as file:
            first_line = True
            num_taxa = sequence_len = 0
            sequences = []

            for line in file:
                line = line.strip()
                temp_line = line.split()

                if first_line:
                    num_taxa = int(temp_line[0])
                    sequence_len = int(temp_line[1])
                    first_line = False
                    continue
                if not line:
                    continue

                sequence = temp_line[-1]
                taxon = " ".join(temp_line[:-1])

                if len(sequence) != sequence_len:
                    raise ValueError(f"unexpected sequence length: {len(sequence)} vs {sequence_len}")
                sequences.append(Sequence(taxon, sequence))
            if len(sequences) != num_taxa:
                raise ValueError(f"unexpected number of sequences: {len(sequences)} vs {num_taxa}")
        return sequences


class FastaParser(MSAParser):
    def parse(self, msa_path):
        with open(msa_path) as file:
            sequences = []
            current_taxon = ""
            current_sequence = ""
            current_comment = ""

            first_line = True

            for line in file:
                line = line.strip()
                if not line:
                    continue

                if first_line:
                    if not line.startswith(">"):
                        raise ValueError(f"taxon name is expected to start with '>'")
                    first_line = False

                if line.startswith(">"):
                    sequence = Sequence(current_taxon, current_sequence, comment=current_comment)
                    if current_sequence:
                        sequences.append(sequence)

                    temp_line = line.split()
                    current_taxon = "".join(list(temp_line[0])[1:])
                    current_comment = " ".join(temp_line[1:])
                    current_sequence = ""
                else:
                    current_sequence += line

            if current_sequence:
                sequence = Sequence(current_taxon, current_sequence, comment=current_comment)
                sequences.append(sequence)

            sequence_len = 0
            for sequence in sequences:
                if not sequence_len:
                    sequence_len = len(sequence.sequence)
                if sequence_len != len(sequence.sequence):
                    raise ValueError(f"unexpected sequence length: {len(sequence.sequence)} vs {sequence_len}")
        return sequences


class PhylipParser(MSAParser):
    def parse(self, msa_path):
        with open(msa_path) as file:
            first_line = True
            seq_names = []
            sequence_dict = collections.defaultdict(lambda: "")
            sequences = []
            num_taxa = seq_len = counter = 0

            for line in file:
                line = line.strip()
                if first_line:
                    num_taxa, seq_len = map(int, line.strip().split())
                    first_line = False
                    continue
                if not line:
                    continue

                if len(seq_names) < num_taxa:
                    temp_line = line.split()
                    seq_name = temp_line[0]
                    seq = "".join(temp_line[1:])

                    seq_names.append(seq_name)
                else:
                    temp_line = line.split()
                    seq = "".join(temp_line)

                sequence_dict[seq_names[counter]] += seq

                counter = (counter + 1) % num_taxa

        if len(seq_names) != num_taxa:
            raise ValueError(f"unexpected number of sequences: {len(seq_names)} vs {num_taxa}")

        for i in range(len(seq_names)):
            taxon = seq_names[i]
            sequence = sequence_dict[taxon]
            if len(sequence) != seq_len:
                raise ValueError(f"unexpected sequence length: {len(sequence)} vs {seq_len}")
            sequences.append(Sequence(taxon, sequence))

        return sequences


def parse_msa_somehow(msa_path):
    parsers = [
        FastaParser, PhylipsParser, PhylipParser
    ]
    num_successful = 0
    sequences = []
    sequences_dict = {}

    error_log = ""

    for parser_class in parsers:
        try:
            parser = parser_class()
            sequences = parser.parse(msa_path)
            sequences_dict[str(parser)] = sequences

            num_successful += 1
        except Exception as e:
            error_log += f"{e}\n"

    if num_successful > 1:
        pass
        # print(f"found {num_successful} different possible msa formats for file {msa_path}!")
    if num_successful == 0:
        print(f"failed to parse {msa_path}")
        print(error_log)

    return sequences


def parse_and_fix_msa_somehow(msa_path, data_type):
    parsers = [
        FastaParser, PhylipsParser, PhylipParser
    ]
    num_successful = 0
    sequences = []
    sequences_dict = {}

    error_log = ""

    for parser_class in parsers:
        try:
            parser = parser_class()
            sequences = parser.parse_and_fix(msa_path, data_type)
            sequences_dict[str(parser)] = sequences

            num_successful += 1
        except Exception as e:
            error_log += f"{e}\n"

    if num_successful > 1:
        pass
        # print(f"found {num_successful} different possible msa formats for file {msa_path}!")
    if num_successful == 0:
        print(f"failed to parse {msa_path}")
        print(error_log)

    return sequences


def save_msa(sequences, out_path, msa_format="fasta"):
    with open(out_path, "w+") as file:
        for sequence in sequences:
            new_record = SeqRecord.SeqRecord(Seq.Seq(sequence.sequence), id=sequence.id, description="")
            SeqIO.write(new_record, file, msa_format)


