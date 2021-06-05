import torch
from pathlib import Path
from typing import List
from otu_identifier.encoders.dna_encoder import KmerEncoder
from otu_identifier.encoders.domain_encoder import DomainEncoder


class SeqDataset(torch.utils.data.Dataset):
    """ Dataset to store seq database data """

    def __init__(self, filename: Path):
        x, y = self.read_fasta(filename=Path(filename))
        self.target = 256
        self.X = x
        self.y = y

    def read_fasta(self, filename: Path) -> List:
        """
        Method do read fasta defined in struct init.

        Parameters
        ----------
        filename: Path, required
            Filename with fasta sequences used to train model

        Returns
        -------
        dataset : List[List[str, str]], required
            Dataset list with each sequence and it's header.
        """
        x = []
        y = []
        with open(filename, "r") as file:
            lines = file.readlines()
        for i in range(0, len(lines), 2):
            domain = lines[i].split()[1].replace("\n", "")
            domain_encoded = DomainEncoder().encode(domain=domain)
            sequence = lines[i+1].replace("\n", "")
            try:
                seq_encoded = KmerEncoder(4).\
                    obtain_kmer_feature_for_one_sequence(
                        sequence,
                        write_number_of_occurrences=False
                )
            except Exception as e:
                print(e)
                continue
            seq_encoded = torch.tensor(seq_encoded, dtype=torch.float32)

            x.append(seq_encoded)
            y.append(domain_encoded)

        return x, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return self.X[idx], self.y[idx]


def seq_to_input(seq):
    """"""
    seq_encoded = KmerEncoder(4).obtain_kmer_feature_for_one_sequence(
        seq,
        write_number_of_occurrences=False
    )
    return torch.tensor(seq_encoded, dtype=torch.float32)
