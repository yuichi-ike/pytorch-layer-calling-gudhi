import unittest 

import torch
from src.layer.dtm import DtmFiltration

class TestLayerDtm(unittest.TestCase):
    def test_init(self):
        kNN = 2
        m = DtmFiltration(1,10, kNN)
    def test_forward(self):
        kNN = 2
        m = DtmFiltration(1,10, kNN)
        x = torch.Tensor([[0,1,2],[1,0,3],[2,3,0]])
        y = m(x)

if __name__ == "__main__":
    unittest.main()     