import torch

from model import Encoder

if __name__ == '__main__':

    bert = Encoder()
    x = torch.rand((32, 300, 512))
    print(bert)
    y = bert(x)