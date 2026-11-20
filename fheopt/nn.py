"""Camadas compatíveis com CKKS reconhecidas pelo motor."""

import torch
from torch import nn


class PolyAct(nn.Module):
    """Ativação polinomial de grau 2: a·x² + b·x + c (consome 1 nível no CKKS).

    Com `learnable=True` os coeficientes são treinados junto com a rede; a inicialização
    (0.125, 0.5, 0.25) é a aproximação de mínimos quadrados da ReLU em [-2, 2].
    """

    def __init__(self, a=0.125, b=0.5, c=0.25, learnable=True):
        super().__init__()
        coef = torch.tensor([a, b, c], dtype=torch.float32)
        if learnable:
            self.coef = nn.Parameter(coef)
        else:
            self.register_buffer("coef", coef)

    def forward(self, x):
        a, b, c = self.coef
        return a * x * x + b * x + c

    def extra_repr(self):
        a, b, c = (float(v) for v in self.coef)
        return f"a={a:.4f}, b={b:.4f}, c={c:.4f}"


def mlp_poly(sizes=(64, 32, 32, 10), **act_kw):
    """MLP com PolyAct entre as camadas lineares (mesma topologia do baseline TFHE)."""
    layers = []
    for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(n_in, n_out))
        if i < len(sizes) - 2:
            layers.append(PolyAct(**act_kw))
    return nn.Sequential(*layers)


def train(model, X, y, seed=0, epochs=200, lr=1e-2):
    """Treino full-batch com Adam e seed fixa."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt, yt = torch.tensor(X), torch.tensor(y)
    for _ in range(epochs):
        opt.zero_grad()
        nn.functional.cross_entropy(model(Xt), yt).backward()
        opt.step()
    return model
