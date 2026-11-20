"""Abordagens avaliadas pelo benchmark.

Linhas de base (Concrete ML, TFHE):
  concrete_ml      modelos prontos do Concrete ML com os padrões da biblioteca
                   (redes neurais: NeuralNetClassifier com treino ciente da quantização)
  concrete_ml_ptq  redes neurais: PTQ 6 bits + arredondamento aproximado + global_p_error=0,1

Abordagens deste projeto:
  fheopt_diagonal  CKKS, método diagonal clássico (referência, sem os passes do motor)
  fheopt           motor CKKS, modo latência (layout replicado + rotate-and-sum)
  fheopt_vazao     motor CKKS, modo vazão (1 amostra por slot, fusão da ativação)
  tfhe_auto        Concrete ML com seleção automática de bits e p_error (árvores e KNN)

As abordagens do motor rodam em duas fases, em processos separados: `prepare` treina o modelo em
claro (scikit-learn/PyTorch) e exporta as camadas; `execute` roda o CKKS só com numpy + OpenFHE
(os runtimes OpenMP do PyTorch e do OpenFHE entram em deadlock no mesmo processo).
Para avaliar uma nova versão do motor, registre-a em APPROACHES com outro nome.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

SEED = 0


# ------------------------------------------------------------------------------------------------
# base

class Approach:
    """Uma forma de executar um cenário sob FHE.

    `phases` são executadas em ordem, cada uma num subprocesso próprio (o runner cuida disso).
    A última fase devolve o dicionário de resultado; fases anteriores podem gravar arquivos
    intermediários em `case_dir`.
    """

    name = "base"
    description = ""
    phases: tuple[str, ...] = ("run",)

    def supports(self, sc) -> tuple[bool, str]:
        return True, ""

    def run_phase(self, phase: str, sc, case_dir, opts: dict) -> dict | None:
        raise NotImplementedError


def time_calls(fn, inputs):
    """Latência de fn(x) para cada x (após uma chamada de aquecimento com inputs[0])."""
    fn(inputs[0])
    lat, outs = [], []
    for x in inputs:
        t0 = time.perf_counter()
        outs.append(fn(x))
        lat.append(time.perf_counter() - t0)
    return np.array(lat), outs


def latency_fields(lat):
    return {"lat_mean_s": float(lat.mean()), "lat_std_s": float(lat.std()), "lat_samples_s": lat.tolist(),
            "n_lat": len(lat)}


# ------------------------------------------------------------------------------------------------
# concrete_ml



def _seed():
    import torch

    np.random.seed(SEED)
    torch.manual_seed(SEED)


def _circuit_stats(circuit):
    st = circuit.statistics
    return {"pbs": int(st.get("programmable_bootstrap_count", 0)), "complexity": float(st.get("complexity", 0)),
            "max_bit_width": int(circuit.graph.maximum_integer_bit_width()),
            "bsk_MB": st.get("size_of_bootstrap_keys", 0) / 2**20, "ksk_MB": st.get("size_of_keyswitch_keys", 0) / 2**20}


def _load(sc):
    from benchmark.scenarios import load_dataset

    X_tr, y_tr, X_val, y_val, X_te, y_te = load_dataset(sc.dataset, seed=SEED)
    if sc.model.get("n_train"):
        X_tr, y_tr = X_tr[: sc.model["n_train"]], y_tr[: sc.model["n_train"]]
    return X_tr, y_tr, X_te, y_te


def _subset(X, y, n):
    return (X, y) if not n else (X[:n], y[:n])


class ConcreteMLBuiltin(Approach):
    name = "concrete_ml"
    description = "Concrete ML, modelos prontos com os padrões da biblioteca (NN: NeuralNetClassifier com QAT)"

    def _model(self, sc, n_features):
        import torch
        from concrete.ml import sklearn as cml

        m = sc.model
        kind = m["kind"]
        if kind == "logreg":
            return cml.LogisticRegression()
        if kind == "linsvc":
            return cml.LinearSVC()
        if kind == "dtree":
            return cml.DecisionTreeClassifier(max_depth=m["max_depth"], random_state=SEED)
        if kind == "rf":
            return cml.RandomForestClassifier(n_estimators=m["n_estimators"], max_depth=m["max_depth"],
                                              random_state=SEED)
        if kind == "xgb":
            return cml.XGBClassifier(n_estimators=m["n_estimators"], max_depth=m["max_depth"], random_state=SEED)
        if kind == "knn":
            return cml.KNeighborsClassifier(n_neighbors=m["k"])
        if kind == "mlp":
            hidden = m["hidden"]
            assert len(set(hidden)) == 1, "NeuralNetClassifier exige camadas ocultas de mesma largura"
            return cml.NeuralNetClassifier(
                module__n_layers=len(hidden) + 1,
                # largura oculta = int(n_features * multiplicador)
                module__n_hidden_neurons_multiplier=(hidden[0] + 0.5) / n_features,
                module__activation_function=torch.nn.ReLU,
                max_epochs=m.get("epochs_qat", 100), verbose=0)
        raise KeyError(kind)

    def run_phase(self, phase, sc, case_dir, opts):
        _seed()
        X_tr, y_tr, X_te, y_te = _load(sc)
        model = self._model(sc, X_tr.shape[1])
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        train_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        model.compile(X_tr)
        compile_s = time.perf_counter() - t0

        acc_clear = float((model.predict(X_te, fhe="disable") == y_te).mean())
        X_acc, y_acc = _subset(X_te, y_te, opts.get("n_acc") or sc.model.get("n_acc"))
        acc_fhe = float((model.predict(X_acc, fhe="simulate") == y_acc).mean())

        t0 = time.perf_counter()
        model.fhe_circuit.keygen(force=True)
        keygen_s = time.perf_counter() - t0
        n_lat = opts.get("n_lat") or sc.n_lat
        X_lat = X_te[:n_lat]
        lat, outs = time_calls(lambda x: model.predict(x[None], fhe="execute")[0], X_lat)
        sim = model.predict(X_lat, fhe="simulate")
        return {"acc_clear": acc_clear, "acc_clear_mode": "quantizado (fhe=disable)", "acc_fhe": acc_fhe,
                "acc_fhe_mode": "simulação FHE (fhe=simulate)", "n_acc": len(y_acc),
                "fidelity": float((np.array(outs) == sim).mean()), "fidelity_ref": "simulação",
                **latency_fields(lat), "train_s": train_s, "compile_s": compile_s, "keygen_s": keygen_s,
                "stats": _circuit_stats(model.fhe_circuit), "config": repr(model.get_params())[:500]}


class ConcreteMLPTQ(Approach):
    name = "concrete_ml_ptq"
    description = ("Concrete ML, MLP PyTorch com PTQ 6 bits + arredondamento aproximado + global_p_error=0,1 "
                   "(melhor configuração TFHE encontrada para o MLP de referência)")

    def supports(self, sc):
        return (sc.model["kind"] == "mlp", "apenas redes neurais")

    def run_phase(self, phase, sc, case_dir, opts):
        import torch
        from torch import nn

        from concrete.ml.torch.compile import compile_torch_model
        from fheopt.nn import train

        _seed()
        X_tr, y_tr, X_te, y_te = _load(sc)
        sizes = [X_tr.shape[1], *sc.model["hidden"], int(y_tr.max()) + 1]
        layers = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        t0 = time.perf_counter()
        net = train(nn.Sequential(*layers), X_tr, y_tr, seed=SEED, epochs=sc.model.get("epochs", 200))
        train_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        qm = compile_torch_model(net, X_tr, n_bits=6, rounding_threshold_bits={"n_bits": 6, "method": "APPROXIMATE"},
                                 global_p_error=0.1)
        compile_s = time.perf_counter() - t0

        with torch.no_grad():
            acc_float = float((net(torch.tensor(X_te)).argmax(1).numpy() == y_te).mean())
        acc_clear = float((qm.forward(X_te, fhe="disable").argmax(1) == y_te).mean())
        X_acc, y_acc = _subset(X_te, y_te, opts.get("n_acc"))
        acc_fhe = float((qm.forward(X_acc, fhe="simulate").argmax(1) == y_acc).mean())

        t0 = time.perf_counter()
        qm.fhe_circuit.keygen(force=True)
        keygen_s = time.perf_counter() - t0
        X_lat = X_te[: opts.get("n_lat") or sc.n_lat]
        lat, outs = time_calls(lambda x: int(qm.forward(x[None], fhe="execute").argmax(1)[0]), X_lat)
        sim = qm.forward(X_lat, fhe="simulate").argmax(1)
        return {"acc_float": acc_float, "acc_clear": acc_clear, "acc_clear_mode": "quantizado (fhe=disable)",
                "acc_fhe": acc_fhe, "acc_fhe_mode": "simulação FHE (fhe=simulate)", "n_acc": len(y_acc),
                "fidelity": float((np.array(outs) == sim).mean()), "fidelity_ref": "simulação",
                **latency_fields(lat), "train_s": train_s, "compile_s": compile_s, "keygen_s": keygen_s,
                "stats": _circuit_stats(qm.fhe_circuit)}


# ------------------------------------------------------------------------------------------------
# fheopt

SUPPORTED = {"logreg", "linsvc", "mlp"}


def _linear_layers(clf):
    W, b = np.asarray(clf.coef_, dtype=np.float64), np.asarray(clf.intercept_, dtype=np.float64)
    if W.shape[0] == 1:  # binário: logits [0, s] -> argmax equivale a s > 0
        W, b = np.vstack([np.zeros_like(W), W]), np.concatenate([[0.0], b])
    return [{"W": W, "b": b, "act": None}]


class FheOptCKKS(Approach):
    phases = ("prepare", "execute")

    def __init__(self, name, plan, description):
        self.name, self.plan, self.description = name, plan, description

    def supports(self, sc):
        k = sc.model["kind"]
        return (k in SUPPORTED, "" if k in SUPPORTED else "comparações (árvores/KNN) não suportadas pelo motor CKKS")

    def run_phase(self, phase, sc, case_dir, opts):
        return self._prepare(sc, Path(case_dir)) if phase == "prepare" else self._execute(sc, Path(case_dir), opts)

    # --- fase 1: treino em claro
    def _prepare(self, sc, case_dir):
        from benchmark.scenarios import load_dataset
        from fheopt.compiler import save_layers

        X_tr, y_tr, _, _, X_te, y_te = load_dataset(sc.dataset, seed=SEED)
        kind = sc.model["kind"]
        t0 = time.perf_counter()
        if kind in ("logreg", "linsvc"):
            from sklearn.linear_model import LogisticRegression
            from sklearn.svm import LinearSVC

            clf = (LogisticRegression(max_iter=5000) if kind == "logreg" else LinearSVC(max_iter=20000))
            clf.fit(X_tr, y_tr)
            layers = _linear_layers(clf)
        else:
            import torch

            from fheopt.compiler import extract
            from fheopt.nn import mlp_poly, train

            torch.manual_seed(SEED)
            sizes = (X_tr.shape[1], *sc.model["hidden"], int(y_tr.max()) + 1)
            net = train(mlp_poly(sizes, learnable=False), X_tr, y_tr, seed=SEED, epochs=sc.model.get("epochs", 400))
            layers = extract(net)
        train_s = time.perf_counter() - t0

        h = X_te.astype(np.float64)  # referência em claro (float64), mesma semântica do programa CKKS
        for l in layers:
            h = h @ l["W"].T + l["b"]
            if l["act"]:
                a, b, c = l["act"]
                h = a * h * h + b * h + c
        save_layers(case_dir / "model.npz", layers, X=X_te.astype(np.float64), y=y_te, clear=h)
        info = {"train_s": train_s, "acc_clear": float((h.argmax(1) == y_te).mean())}
        (case_dir / "prepare.json").write_text(json.dumps(info))
        return info

    # --- fase 2: execução CKKS
    def _execute(self, sc, case_dir, opts):
        from fheopt.backend import OpenFHEBackend
        from fheopt.compiler import PLANS, load_layers

        prep = json.loads((case_dir / "prepare.json").read_text())
        layers, d = load_layers(case_dir / "model.npz")
        X, y, clear = d["X"], d["y"], d["clear"]

        t0 = time.perf_counter()
        plan = PLANS[self.plan](layers)
        plan.check(X[:20], clear[:20], tol=1e-6)
        compile_s = time.perf_counter() - t0
        be = OpenFHEBackend(plan.slots, plan.depth, plan.rotations)
        t0 = time.perf_counter()
        pts = plan.prepare(be)
        encode_s = time.perf_counter() - t0

        n_lat = opts.get("n_lat") or sc.n_lat
        lat, _ = time_calls(lambda x: plan.infer(be, pts, x), X[:n_lat])
        n_acc = opts.get("n_acc") or 0
        Xa, ya, ca = (X, y, clear) if not n_acc else (X[:n_acc], y[:n_acc], clear[:n_acc])
        logits = np.array([plan.infer(be, pts, x) for x in Xa])
        return {"acc_clear": prep["acc_clear"], "acc_clear_mode": "float64 (mesmo modelo, sem criptografia)",
                "acc_fhe": float((logits.argmax(1) == ya).mean()), "acc_fhe_mode": "execução CKKS real",
                "n_acc": len(ya), "fidelity": float((logits.argmax(1) == ca.argmax(1)).mean()),
                "fidelity_ref": "claro", "max_abs_err_logits": float(np.abs(logits - ca).max()),
                **latency_fields(lat), "train_s": prep["train_s"], "compile_s": compile_s, "keygen_s": be.keygen_s,
                "encode_s": encode_s,
                "stats": {**plan.summary(), "ring_dim": be.ring_dim, "rot_keys": be.n_rot_keys}}


class FheOptVazao(FheOptCKKS):
    """Motor em modo vazão: 1 amostra por slot, sem rotações, ativação com 1 nível (ver fheopt/vazao.py).

    Latência reportada = custo AMORTIZADO por amostra com o lote cheio (S = N/2 amostras, o conjunto
    de teste repetido até encher os slots). A latência do lote inteiro fica em batch_latency_s.
    """

    def __init__(self):
        super().__init__("fheopt_vazao", "vazao",
                         "CKKS, motor em modo vazão: 1 amostra/slot, sem rotações (custo amortizado por amostra)")

    def _execute(self, sc, case_dir, opts):
        from fheopt.compiler import load_layers
        from fheopt.vazao import VazaoCKKS, fold, reference

        prep = json.loads((case_dir / "prepare.json").read_text())
        layers, d = load_layers(case_dir / "model.npz")
        X, y, clear = d["X"], d["y"], d["clear"]
        assert np.allclose(reference(fold(layers), X), clear), "reescrita da ativação alterou o modelo"

        # escala de 36 bits; o primeiro módulo reserva margem para o maior logit (+6 bits)
        head = int(np.ceil(np.log2(np.abs(clear).max() + 1))) + 6
        t0 = time.perf_counter()
        v = VazaoCKKS(layers, scale_bits=36, first_mod_bits=min(60, max(48, 36 + head)))
        compile_s = time.perf_counter() - t0
        Xb = np.resize(X, (v.slots, X.shape[1]))
        v.run_batch(Xb[:64])  # aquecimento
        reps = max(1, min(opts.get("n_lat") or 3, 3))
        tot, logits, parts = [], None, []
        for _ in range(reps):
            logits, t = v.run_batch(Xb)
            tot.append(t["total_s"])
            parts.append(t)
        tot = np.array(tot)
        lg = logits[: len(X)]
        per = tot / v.slots
        return {"acc_clear": prep["acc_clear"], "acc_clear_mode": "float64 (mesmo modelo, sem criptografia)",
                "acc_fhe": float((lg.argmax(1) == y).mean()), "acc_fhe_mode": "execução CKKS real",
                "n_acc": len(y), "fidelity": float((lg.argmax(1) == clear.argmax(1)).mean()),
                "fidelity_ref": "claro", "max_abs_err_logits": float(np.abs(lg - clear).max()),
                "lat_kind": "amortizada", "batch_size": v.slots,
                "batch_latency_s": float(tot.mean()), "batch_parts_s": {k: float(np.mean([p[k] for p in parts]))
                                                                        for k in parts[0]},
                **latency_fields(per), "train_s": prep["train_s"], "compile_s": compile_s,
                "keygen_s": v.keygen_s, "stats": v.stats}


# ------------------------------------------------------------------------------------------------
# tfhe_auto

GRID = {"tree": [(nb, gpe) for nb in (6, 5, 4, 3, 2) for gpe in (None, 0.1)],
        "knn": [(nb, gpe) for nb in (2, 3) for gpe in (None, 0.1)]}
TOL = 0.01


class TfheAuto(Approach):
    name = "tfhe_auto"
    description = ("Concrete ML com otimizador automático: varre bits e p_error, escolhe em treino + validação "
                   "(≤1 p.p. de perda) a configuração de menor custo")

    def supports(self, sc):
        return (sc.family in GRID, "aplicado a árvores e KNN")

    def _build(self, sc, n_bits, X_tr, y_tr, gpe):
        model = ConcreteMLBuiltin()._model(sc, X_tr.shape[1])
        model.set_params(n_bits=n_bits)
        model.fit(X_tr, y_tr)
        model.compile(X_tr, **({"global_p_error": gpe} if gpe else {}))
        return model

    def run_phase(self, phase, sc, case_dir, opts):
        from benchmark.scenarios import load_dataset

        _seed()
        X_tr, y_tr, X_val, y_val, X_te, y_te = load_dataset(sc.dataset, seed=SEED)
        if sc.model.get("n_train"):
            X_tr, y_tr = X_tr[: sc.model["n_train"]], y_tr[: sc.model["n_train"]]
        default_nb = 2 if sc.family == "knn" else 6
        X_sel, y_sel = np.vstack([X_tr, X_val]), np.concatenate([y_tr, y_val])
        trials, t0 = [], time.perf_counter()
        for nb, gpe in GRID[sc.family]:
            try:
                m = self._build(sc, nb, X_tr, y_tr, gpe)
                acc = float((m.predict(X_sel, fhe="simulate") == y_sel).mean())
                trials.append({"n_bits": nb, "global_p_error": gpe, "acc_sel": acc,
                               "complexity": float(m.fhe_circuit.statistics["complexity"]),
                               "pbs": int(m.fhe_circuit.statistics["programmable_bootstrap_count"])})
            except Exception as e:  # noqa: BLE001
                trials.append({"n_bits": nb, "global_p_error": gpe, "erro": f"{type(e).__name__}: {e}"[:200]})
        search_s = time.perf_counter() - t0
        ok = [t for t in trials if "erro" not in t]
        if not ok:
            raise RuntimeError(f"nenhuma configuração compilou ({trials[0].get('erro')})")
        ref = next((t for t in ok if t["n_bits"] == default_nb and t["global_p_error"] is None), max(ok, key=lambda t: t["acc_sel"]))
        best = min((t for t in ok if t["acc_sel"] >= ref["acc_sel"] - TOL), key=lambda t: t["complexity"])

        _seed()
        m = self._build(sc, best["n_bits"], X_tr, y_tr, best["global_p_error"])
        acc_clear = float((m.predict(X_te, fhe="disable") == y_te).mean())
        acc_fhe = float((m.predict(X_te, fhe="simulate") == y_te).mean())
        t0 = time.perf_counter()
        m.fhe_circuit.keygen(force=True)
        keygen_s = time.perf_counter() - t0
        X_lat = X_te[: opts.get("n_lat") or sc.n_lat]
        lat, outs = time_calls(lambda x: m.predict(x[None], fhe="execute")[0], X_lat)
        sim = m.predict(X_lat, fhe="simulate")
        return {"acc_clear": acc_clear, "acc_clear_mode": "quantizado (fhe=disable)", "acc_fhe": acc_fhe,
                "acc_fhe_mode": "simulação FHE (fhe=simulate)", "n_acc": len(y_te),
                "fidelity": float((np.array(outs) == sim).mean()), "fidelity_ref": "simulação",
                **latency_fields(lat), "keygen_s": keygen_s, "search_s": search_s,
                "chosen": best, "reference": ref, "trials": trials, "stats": _circuit_stats(m.fhe_circuit)}


APPROACHES = {a.name: a for a in [
    ConcreteMLBuiltin(),
    ConcreteMLPTQ(),
    FheOptCKKS("fheopt_diagonal", "diagonal", "CKKS, método diagonal clássico (referência, sem passes do motor)"),
    FheOptCKKS("fheopt", "replicated", "CKKS, motor fheopt com os passes atuais (layout replicado + rotate-and-sum)"),
    FheOptVazao(),
    TfheAuto(),
]}

BASELINES = ("concrete_ml", "concrete_ml_ptq")  # linhas de base TFHE (Concrete ML)
OURS = ("fheopt", "fheopt_diagonal", "fheopt_vazao", "tfhe_auto")  # abordagens deste projeto
