import torch
from typing import Any


class DomainEncoder:

    def encode(self, domain: str) -> Any:
        """
        Encodes Domain taxonomy into One Hot Enconding to feed DL model.

        Parameters
        ----------
        domain : str, required
            domain category string.
            Example:
              "Eukaryota"

        Returns
        -------
        Tensor
            Tensor with domain category one hot encoded.
        """
        if domain == "Eukaryota":
            onehot = torch.tensor(1, dtype=torch.float32)
        elif domain == "Bacteria":
            onehot = torch.tensor(0, dtype=torch.float32)
        else:
            raise ValueError(f"Domain {domain} not right, needs to be Eukaryota\
                or Bacteria")

        return onehot
