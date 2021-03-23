import numpy as np
from sklearn.preprocessing import OneHotEncoder
from typing import Any


LABEL_ENCODE_DNA = {
    "A": 0,
    "T": 1,
    "C": 2,
    "G": 3,
    "N": 4,
}


class DnaEncoder:

    def encode(self, sequence: str) -> Any:
        """Encodes DNA sequence into One Hot Enconding to feed DL model.

        Parameters
        ----------
        sequence : str, required
            DNA Sequence used for one hot encoding.

        Returns
        -------
        Array
            Array struct with DNA sequence one hot encoded.
        """
        seq_array = np.array(list(sequence))
        label_encoded_seq = []
        for nucl in seq_array:
            encoded_nucl = LABEL_ENCODE_DNA[nucl]
            label_encoded_seq.append(encoded_nucl)
        label_encoded_seq = np.array(label_encoded_seq).reshape(
            len(label_encoded_seq),
            1
        )

        onehot_encoder = OneHotEncoder(sparse=False, dtype=np.int8)
        onehot_encoded_seq = onehot_encoder.fit_transform(label_encoded_seq)
        return onehot_encoded_seq


teste = DnaEncoder()
print(teste.encode("ATCG"))
