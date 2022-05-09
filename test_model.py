import numpy as np
import torch
from pyitcast.transformer_utils import (
    Batch,
    LabelSmoothing,
    SimpleLossCompute,
    get_std_opt,
    greedy_decode,
    run_epoch,
)
from torch.autograd import Variable

from model import make_model


def data_generator(V, batch, num_batch):
    for i in range(num_batch):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        source = Variable(data, requires_grad=False)
        target = Variable(data, requires_grad=False)

        yield Batch(source, target)  # generator


V = 11
batch = 20
num_batch = 30

# if __name__ == "__main__":
#     res = data_generator(V, batch, num_batch)
#     print(res) # we will get a generator object

model = make_model(V, V, N=2)
model_optimizer = get_std_opt(model)
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
loss = SimpleLossCompute(model.generator, criterion, model_optimizer)

# import pyplot as plt

# crit = LabelSmoothing(size=5, padding_idx=0, smoothing=0.5)
# predict = Variable(
#     torch.LongTensor(
#         [
#             [0, 0.2, 0.7, 0.1, 0],
#             [0, 0.2, 0.7, 0.1, 0],
#             [0, 0.2, 0.7, 0.1, 0],
#         ]
#     )
# )
# target = Variable(torch.LongTensor([2, 1, 0]))
# crit(predict, target)
# plt.imshow(crit.true_dist)


def run(model, loss, epochs=10):
    for epoch in range(epochs):
        model.train()
        run_epoch(data_generator(V, 8, 20), model, loss)

        model.eval()
        run_epoch(data_generator(V, 8, 5), model, loss)

    model.eval()
    source = Variable(torch.LongTensor([[1, 3, 2, 5, 4, 6, 7, 8, 9, 10]]))
    source_mask = Variable(torch.ones(1, 1, 10))
    result = greedy_decode(model, source, source_mask, max_len=10, start_symbol=1)
    print(result)


# if __name__ == "__main__":
#     run(model, loss)
