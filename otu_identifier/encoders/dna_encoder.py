import numpy as np
from sklearn.preprocessing import OneHotEncoder
from typing import Any

LABEL_ENCODE_DNA = {
    "A": 0,
    "T": 1,
    "C": 2,
    "G": 3,
    "N": 4,
    "M": 5,
    "S": 6,
    "R": 7,
    "H": 8,
    "K": 9,
    "Y": 10,
    "W": 11,
    "B": 12,
    "V": 13,
    "D": 14,
}


class DnaEncoder:

    def encode(self, sequence: str) -> Any:
        """
        Encodes DNA sequence into One Hot Enconding to feed DL model.

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


class KmerEncoder:

    def __init__(self, k):
        """
        seqs: a list of DNA sequences
        k: the "k" in k-mer
        """
        self.k = k
        self.letters = ['A', 'T', 'C', 'G']
        self.multiplyBy = 4 ** np.arange(k-1, -1, -1)
        self.n = 4**k

    def obtain_kmer_feature_for_a_list_of_sequences(
        self,
        seqs,
        write_number_of_occurrences=False
    ):
        """
        Given a list of m DNA sequences, return a 2-d array with shape
        (m, 4**k) for the 1-hot representation of the kmer features.
        Args:
        write_number_of_occurrences:
        a boolean. If False, then in the 1-hot representation, the percentage
        of the occurrence of a kmer will be recorded; otherwise the number of
        occurrences will be recorded. Default False.
        """
        kmer_features = []
        for seq in seqs:
            this_kmer_feature = self.\
                obtain_kmer_feature_for_one_sequence(
                    seq.upper(),
                    write_number_of_occurrences=write_number_of_occurrences
                )
            kmer_features.append(this_kmer_feature)

        kmer_features = np.array(kmer_features)

        return kmer_features

    def obtain_kmer_feature_for_one_sequence(
        self,
        seq,
        write_number_of_occurrences=False
    ):
        """
        Given a DNA sequence, return the 1-hot representation of its kmer
        feature.
        Args:
      seq:
        a string, a DNA sequence
        write_number_of_occurrences:
        a boolean. If False, then in the 1-hot representation, the percentage0
        of the occurrence of a kmer will be recorded; otherwise the number of
        occurrences will be recorded. Default False.
        """
        number_of_kmers = len(seq) - self.k + 1

        kmer_feature = np.zeros(self.n)

        for i in range(number_of_kmers):
            this_kmer = seq[i:(i+self.k)]
            this_numbering = self.kmer_numbering_for_one_kmer(this_kmer)
            kmer_feature[this_numbering] += 1

        if not write_number_of_occurrences:
            kmer_feature = kmer_feature / number_of_kmers

        kmer_feature = kmer_feature.astype(np.float32)

        return kmer_feature

    def kmer_numbering_for_one_kmer(self, kmer):
        """
        Given a k-mer, return its numbering (the 0-based position
        in 1-hot representation)
        """
        digits = []
        for letter in kmer:
            digits.append(self.letters.index(letter))

        digits = np.array(digits)

        numbering = (digits * self.multiplyBy).sum()

        return numbering
