"""Estudo da ativação polinomial: acurácia em claro do MLP de referência (5 seeds).

No CKKS a ReLU não é avaliada exatamente; ela é trocada por um polinômio de grau 2 e a rede é
retreinada. Compara a ReLU (referência) com x², com 0,125x² + 0,5x + 0,25 fixo (aproximação da
ReLU em [-2, 2], adotada pelo motor) e com a·x² + b·x + c treinável.

python -m benchmark.ativacao   →   results/ativacao.json
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from benchmark.scenarios import load_dataset
from fheopt.nn import mlp_poly, train

OUT = Path(__file__).resolve().parents[1] / "results" / "ativacao.json"
SEEDS, EPOCHS = [0, 1, 2, 3, 4], 400
VARIANTS = {
    "relu": lambda: nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 10)),
    "x2": lambda: mlp_poly(a=1.0, b=0.0, c=0.0, learnable=False),
    "relu_fix": lambda: mlp_poly(learnable=False),
    "poly_learn": lambda: mlp_poly(learnable=True),
}


def main():
    X_tr, y_tr, X_val, y_val, X_te, y_te = load_dataset("digits", seed=0)
    acc = lambda m, X, y: float((m(torch.tensor(X)).argmax(1).numpy() == y).mean())  # noqa: E731
    runs = []
    for name, make in VARIANTS.items():
        for seed in SEEDS:
            torch.manual_seed(seed)
            m = train(make(), X_tr, y_tr, seed=seed, epochs=EPOCHS)
            with torch.no_grad():
                runs.append({"variant": name, "seed": seed, "acc_val": acc(m, X_val, y_val),
                             "acc_test": acc(m, X_te, y_te)})
            print(runs[-1], flush=True)
    summary = {}
    for name in VARIANTS:
        t = np.array([r["acc_test"] for r in runs if r["variant"] == name])
        v = np.array([r["acc_val"] for r in runs if r["variant"] == name])
        summary[name] = {"acc_val_mean": v.mean(), "acc_val_std": v.std(), "acc_test_mean": t.mean(),
                         "acc_test_std": t.std()}
    OUT.write_text(json.dumps({"epochs": EPOCHS, "seeds": SEEDS, "summary": summary, "runs": runs}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
