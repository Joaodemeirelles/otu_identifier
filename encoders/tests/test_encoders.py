import unittest
import numpy as np


from dna_encoder import DnaEncoder


class TestDnaEncoder(unittest.TestCase):

    def test_false_input(self):
        """ Test if encode doesn't accept wrong inputs """
        encoder = DnaEncoder()
        protein_seq = "MNLLLTLLTN"
        with self.assertRaises(Exception):
            encoder.encode(protein_seq)

    def test_right_encoding(self):
        """ Test if encoder is making the right encoding"""
        encoder = DnaEncoder()
        sequence = "ATCG"
        expected_result = np.array([
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]
                        ])
        expected_result = np.array(expected_result)

        encoded_seq = encoder.encode(sequence)
        equal = np.array_equal(encoded_seq, expected_result)
        self.assertTrue(equal)


if __name__ == '__main__':
    unittest.main()
