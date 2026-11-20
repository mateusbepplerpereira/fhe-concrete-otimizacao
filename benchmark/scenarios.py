"""Cenários fixos do benchmark: mesmos dados e mesma especificação de modelo para todas as abordagens.

Cada cenário descreve O QUE é avaliado (família, dataset, arquitetura/hiperparâmetros); cada
abordagem decide COMO avaliá-lo (Concrete ML, motor CKKS, ...). Não altere um cenário existente:
isso invalida a comparação entre entregas. Para mudar algo, crie um cenário novo com outro ID.

Famílias cobertas (as quatro do Concrete ML): linear, tree, nn, knn.
"""

from dataclasses import dataclass, field

import numpy as np

DATASETS = ("digits", "breast_cancer")


@dataclass(frozen=True)
class Scenario:
    id: str
    family: str  # linear | tree | nn | knn
    dataset: str  # ver load_dataset
    model: dict = field(default_factory=dict)  # especificação independente da abordagem
    n_lat: int = 5  # amostras para medir latência (após 1 de aquecimento)
    description: str = ""


SCENARIOS = [
    # --- modelos lineares
    Scenario("linear/logreg_cancer", "linear", "breast_cancer", {"kind": "logreg"},
             description="Regressão logística binária, 30 atributos"),
    Scenario("linear/logreg_digits", "linear", "digits", {"kind": "logreg"},
             description="Regressão logística multiclasse (10), 64 atributos"),
    Scenario("linear/linsvc_cancer", "linear", "breast_cancer", {"kind": "linsvc"},
             description="SVM linear binário, 30 atributos"),
    # --- árvores
    Scenario("tree/dtree_cancer", "tree", "breast_cancer", {"kind": "dtree", "max_depth": 5},
             description="Árvore de decisão, profundidade 5"),
    Scenario("tree/rf_cancer", "tree", "breast_cancer", {"kind": "rf", "n_estimators": 20, "max_depth": 4},
             description="Random forest, 20 árvores, profundidade 4"),
    Scenario("tree/xgb_cancer", "tree", "breast_cancer", {"kind": "xgb", "n_estimators": 20, "max_depth": 3},
             description="XGBoost, 20 árvores, profundidade 3"),
    # --- redes neurais (MLP)
    Scenario("nn/mlp_digits_32x2", "nn", "digits", {"kind": "mlp", "hidden": [32, 32]},
             description="MLP 64→32→32→10 (modelo de referência da pesquisa)"),
    Scenario("nn/mlp_digits_64x2", "nn", "digits", {"kind": "mlp", "hidden": [64, 64]},
             description="MLP 64→64→64→10 (mais largo)"),
    Scenario("nn/mlp_cancer_16x2", "nn", "breast_cancer", {"kind": "mlp", "hidden": [16, 16]},
             description="MLP 30→16→16→2"),
    # --- vizinhos mais próximos
    Scenario("knn/knn_cancer", "knn", "breast_cancer", {"kind": "knn", "k": 3, "n_train": 100}, n_lat=2,
             description="KNN k=3, 100 pontos de treino (custo cresce com o tamanho do treino)"),
]

BY_ID = {s.id: s for s in SCENARIOS}
FAMILIES = ("linear", "tree", "nn", "knn")


def select(ids=None, families=None):
    out = SCENARIOS
    if families:
        out = [s for s in out if s.family in families]
    if ids:
        missing = set(ids) - set(BY_ID)
        if missing:
            raise KeyError(f"cenários desconhecidos: {sorted(missing)}")
        out = [s for s in out if s.id in ids]
    return out


def load_dataset(name: str, seed: int = 0):
    """Retorna (X_tr, y_tr, X_val, y_val, X_te, y_te), features padronizadas no treino (float32)."""
    from sklearn.datasets import load_breast_cancer, load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    loaders = {"digits": load_digits, "breast_cancer": load_breast_cancer}
    if name not in loaders:
        raise KeyError(f"dataset desconhecido: {name} (disponíveis: {', '.join(DATASETS)})")
    X, y = loaders[name](return_X_y=True)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=seed, stratify=y_tmp)
    sc = StandardScaler().fit(X_tr)
    f = lambda a: sc.transform(a).astype(np.float32)  # noqa: E731
    i = lambda a: np.asarray(a, dtype=np.int64)  # noqa: E731
    return f(X_tr), i(y_tr), f(X_val), i(y_val), f(X_te), i(y_te)
