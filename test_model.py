from pyitcast.transformer_utils import (
    Batch,
    LabelSmoothing,
    SimpleLossCompute,
    get_std_opt,
)

from model import make_model
import torch
import numpy as np
from torch.autograd import Variable


def data_generator(V, batch, num_batch):
    for i in range(num_batch):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        source = Variable(data, requires_grad=False)
        target = Variable(data, requires_grad=False)

        yield Batch(source, target) # generator


V = 11
batch = 20
num_batch = 30

if __name__ == "__main__":
    res = data_generator(V, batch, num_batch)
    print(res) # we will get a generator object

model = make_model(V, V, N=2)
model_optimizer = get_std_opt(model)
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
loss = SimpleLossCompute(model.generator, criterion, model_optimizer)
