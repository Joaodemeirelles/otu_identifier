import unittest
import numpy as np


from encoders.dna_encoder import DnaEncoder
from encoders.family_encoder import FamilyEncoder


class TestDnaEncoder(unittest.TestCase):
    """ Class to test DnaEnconder class """

    def test_false_input(self):
        """ Test if encoder doesn't accept wrong inputs """
        encoder = DnaEncoder()
        protein_seq = "MNLLLTLLTN"
        with self.assertRaises(Exception):
            encoder.encode(protein_seq)

    def test_right_encoding(self):
        """ Test if encoder is making the right encoding """
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


class TestFamilyEncoder(unittest.TestCase):
    """ Class to test FamilyEncoder class """

    def test_false_input(self):
        """ Test if encoder doesn't accept wrong inputs """
        encoder = FamilyEncoder()
        wrong_category = "123"
        with self.assertRaises(Exception):
            encoder.encode(wrong_category)

    def test_right_encoding(self):
        """ Test if encoder is making the right encoding """
        encoder = FamilyEncoder()
        category = "Pa"
        expected_result = [
            np.array(
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
                ),
            np.array(
                [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
                ),
        ]

        encoded_cat = encoder.encode(category)
        equal = np.array_equal(encoded_cat, expected_result)
        self.assertTrue(equal)


if __name__ == '__main__':
    unittest.main()
