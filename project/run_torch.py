import torch
import datasets
import matplotlib.pyplot as plt

PTS = 250
DATASET = datasets.Xor(PTS)
HIDDEN = 10
RATE = 0.5


# Model with
class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Submodules
        self.layer1 = Linear(2, HIDDEN)
        self.layer2 = Linear(HIDDEN, HIDDEN)
        self.layer3 = Linear(HIDDEN, 1)

    def forward(self, x):
        h = self.layer1.forward(x).relu()
        h = self.layer2.forward(h).relu()
        return self.layer3.forward(h).sigmoid()


class Linear(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weight = torch.nn.Parameter(2 * (torch.rand(in_size, out_size) - 0.5))
        self.bias = torch.nn.Parameter(2 * (torch.rand(out_size) - 0.5))

    def forward(self, x):
        return x @ self.weight + self.bias


model = Network()

# Dataset
data = DATASET

losses = []
for epoch in range(500):

    # Forward
    out = model.forward(torch.tensor(data.X, requires_grad=True)).view(data.N)
    y = torch.tensor(data.y)
    probs = (out * y) + (out - 1.0) * (y - 1.0)
    loss = -probs.log().sum()

    # Update
    loss.view(1).backward()

    for p in model.parameters():
        if p.grad is not None:
            p.data = p.data - RATE * (p.grad / float(data.N))
            p.grad.zero_()

    # Logging
    pred = out > 0.5
    correct = ((y == 1) * (pred)).sum() + ((y == 0) * (~pred)).sum()
    losses.append(loss)

    if epoch % 10 == 0:
        print("Epoch ", epoch, " loss ", loss, "correct", correct)
        im = f"graph epoch: {epoch} loss: {loss}"

    if epoch % 50 == 0:

        def check(x):
            return model.forward(torch.tensor(x).view(1, 2))[0, 0]

        data.graph(im, check)
        plt.plot(losses, c="blue")
        data.vis.matplot(plt, win="loss")
