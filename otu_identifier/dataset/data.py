import pandas as pd
from pathlib import Path
from typing import Any, List
from otu_identifier.encoders.dna_encoder import DnaEncoder
from otu_identifier.encoders.family_encoder import FamilyEncoder


class Database:
    """ Class to store FASTA database data """

    def __init__(self, filename):
        self.path = Path(filename)

    def read_fasta(self) -> List:
        """
        Method do read fasta defined in struct init.

        Parameters
        ----------

        Returns
        -------
        dataset : List[List[str, str]], required
            Dataset list with each sequence and it's header.
        """
        dataset = []
        with open(self.path, "r") as file:
            lines = file.readlines()
        for i in range(0, len(lines), 2):
            header = lines[i].replace("\n", "").replace(">", "")
            header_encoded = FamilyEncoder().encode(family=header)
            sequence = lines[i+1].replace("\n", "")
            seq_encoded = DnaEncoder().encode(sequence=sequence)
            dataset.append([header_encoded, seq_encoded])

        return dataset

    def file_to_df(self, dataset: List) -> Any:
        """Method to transform dataset into Pandas DataFrame.

        Parameters
        ----------
        dataset : List[List[str, str]], required
            Dataset list with each sequence and it's header.

        Returns
        -------
        df
            Pandas dataframe with every dataset sequence and family.
        """
        df = pd.DataFrame(dataset, columns=["family", "sequence"])
        return df


teste = Database(filename="../16s_family.fasta")
dataset = teste.read_fasta()
df = teste.file_to_df(dataset)
print(df.head())
