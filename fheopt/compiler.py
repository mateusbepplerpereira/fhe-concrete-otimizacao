"""Compilação de um MLP PyTorch (Linear + PolyAct) para um programa CKKS sobre slots.

Dois planos de layout:

DiagonalPlan (referência, método diagonal híbrido):
  cada camada m×n faz m_p multiplicações por diagonais da matriz com m_p−1 rotações da entrada,
  mais log2(n_p/m_p) rotações para dobrar o resultado. Entre camadas, a saída precisa ser
  reempacotada (máscara: +1 nível; replicação: +1 rotação) para o formato cíclico da próxima.

ReplicatedPlan (proposto, layout replicado + rotate-and-sum, sem reempacotamento):
  alterna dois layouts para que a saída de uma camada já esteja no formato da próxima:
    E (expandido, bloco B): slot(i·B + j) = x_i    -> soma sobre i com passos B, 2B, ...  -> saída compacta
    T (ladrilhado, período P): slot(k) = x_{k mod P} -> soma dentro do bloco com passos 1, 2, ... -> saída com passo P
  A conversão compacta→T e passo-P→E é feita por rotate-and-sum (sem nível), e a máscara que
  zera o lixo entre slots é fundida aos coeficientes da ativação (sem nível extra).
  Custo por camada: log2(n_p) rotações + log2(m_p) para preparar a próxima.
"""

from __future__ import annotations

import numpy as np

from .backend import SlotSim


def pow2(n: int) -> int:
    return 1 << max(0, int(np.ceil(np.log2(n))))


def extract(model) -> list[dict]:
    """Lista de camadas: {"W", "b", "act": (a, b, c) | None}.

    Aceita um nn.Sequential (Linear + PolyAct) ou a própria lista já extraída (ver `save_layers`).
    O torch só é importado aqui: o processo que executa CKKS não deve carregar o torch, cujo
    runtime OpenMP entra em deadlock com o do OpenFHE no mesmo processo.
    """
    if isinstance(model, list):
        return model
    from torch import nn

    from .nn import PolyAct

    layers = []
    for mod in model:
        if isinstance(mod, nn.Linear):
            layers.append({"W": mod.weight.detach().double().numpy(),
                           "b": mod.bias.detach().double().numpy(), "act": None})
        elif isinstance(mod, PolyAct):
            layers[-1]["act"] = tuple(float(v) for v in mod.coef.detach())
        else:
            raise TypeError(f"camada não suportada pelo motor: {type(mod).__name__}")
    return layers


def save_layers(path, layers, **extra):
    """Salva as camadas extraídas (e arrays extras) em .npz."""
    arrs = {}
    for i, l in enumerate(layers):
        arrs[f"W{i}"], arrs[f"b{i}"] = l["W"], l["b"]
        if l["act"] is not None:
            arrs[f"act{i}"] = np.array(l["act"])
    np.savez(path, n_layers=len(layers), **arrs, **extra)


def load_layers(path):
    """Inverso de `save_layers`: (camadas, dict com os arrays extras)."""
    z = np.load(path)
    n = int(z["n_layers"])
    layers = [{"W": z[f"W{i}"], "b": z[f"b{i}"],
               "act": tuple(float(v) for v in z[f"act{i}"]) if f"act{i}" in z else None} for i in range(n)]
    keys = {f"W{i}" for i in range(n)} | {f"b{i}" for i in range(n)} | {f"act{i}" for i in range(n)} | {"n_layers"}
    return layers, {k: z[k] for k in z.files if k not in keys}


class Plan:
    name = "base"

    def __init__(self, model, slots=None):
        self.layers = extract(model)
        if slots is None:
            self._autosize()
        else:
            self.slots = slots
            self.vecs: dict[str, np.ndarray] = {}
            self.build()
        self._analyze()

    def _autosize(self, max_slots=1 << 16):
        """Menor potência de 2 de slots em que o programa reproduz a referência numpy."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(4, self.layers[0]["W"].shape[1]))
        ref = self.reference(X)
        slots = 8
        while slots <= max_slots:
            self.slots, self.vecs = slots, {}
            try:
                self.build()
                self.check(X, ref, tol=1e-8)
                return
            except (IndexError, ValueError, AssertionError):
                slots *= 2
        raise ValueError(f"{self.name}: modelo não cabe em {max_slots} slots")

    def reference(self, X):
        """Inferência em claro (float64) com a mesma semântica do programa."""
        h = np.asarray(X, dtype=np.float64)
        for l in self.layers:
            h = h @ l["W"].T + l["b"]
            if l["act"]:
                a, b, c = l["act"]
                h = a * h * h + b * h + c
        return h

    # --- interface dos planos
    def build(self): ...
    def encode_input(self, x) -> np.ndarray: ...
    def run(self, be, pts, ct): ...
    out_index: np.ndarray

    # --- comum
    def _vec(self, key, values: dict | np.ndarray):
        v = np.zeros(self.slots)
        if isinstance(values, dict):
            for k, val in values.items():
                v[k] = val
        else:
            v[: len(values)] = values
        assert len(v) == self.slots, f"{key}: layout excede {self.slots} slots"
        self.vecs[key] = v

    def prepare(self, be):
        """Codifica todos os plaintexts (pesos, bias, máscaras) uma vez, fora da medição."""
        return {k: be.encode(v) for k, v in self.vecs.items()}

    def infer(self, be, pts, x):
        ct = be.encrypt(self.encode_input(x))
        out = self.run(be, pts, ct)
        return be.decrypt(out, int(self.out_index.max()) + 1)[self.out_index]

    def _analyze(self):
        """Executa no simulador: conta operações, rotações distintas e profundidade."""
        sim = _RecordingSim(self.slots)
        pts = self.prepare(sim)
        n_in = self.layers[0]["W"].shape[1]
        out = self.run(sim, pts, sim.encrypt(self.encode_input(np.zeros(n_in))))
        self.ops = dict(sim.ops)
        self.rotations = sorted(sim.rot_idx)
        self.depth = out.level

    def check(self, X, ref, tol=1e-6):
        """Confere o programa (em claro, no simulador) contra a saída do PyTorch."""
        sim = SlotSim(self.slots)
        pts = self.prepare(sim)
        got = np.array([self.infer(sim, pts, x) for x in X])
        err = np.abs(got - ref).max()
        assert err < tol, f"{self.name}: erro {err:.2e} contra o PyTorch"
        return err

    def _act(self, be, pts, acc, i, masked):
        a, b, c = self.layers[i]["act"]
        if masked:  # máscara fundida aos coeficientes: x·(a·m⊙x + b·m) + c·m
            t = be.add_pt(be.mul_pt(acc, pts[f"a{i}"]), pts[f"b{i}"])
            return be.add_pt(be.mul_ct(t, acc), pts[f"c{i}"])
        t = be.add_scalar(be.mul_scalar(acc, a), b)
        return be.add_scalar(be.mul_ct(t, acc), c)

    def summary(self):
        return {"plan": self.name, "depth": self.depth, "rotations_per_inference": self.ops.get("rot", 0),
                "distinct_rotation_keys": len(self.rotations), "ops": self.ops, "slots": self.slots}


class _RecordingSim(SlotSim):
    def __init__(self, slots):
        super().__init__(slots)
        self.rot_idx = set()

    def rot(self, a, k):
        if k % self.slots:
            self.rot_idx.add(k)
        return super().rot(a, k)


class DiagonalPlan(Plan):
    name = "diagonal"

    def build(self):
        self.geo = []
        L = self.layers
        for i, l in enumerate(L):
            m, n = l["W"].shape
            m_p = pow2(m)
            n_p = max(pow2(n), m_p)
            Wp = np.zeros((m_p, n_p))
            Wp[:m, :n] = l["W"]
            k = np.arange(n_p)
            for d in range(m_p):
                self._vec(f"W{i}_{d}", Wp[k % m_p, (k + d) % n_p])
            self._vec(f"bias{i}", l["b"])
            if i < len(L) - 1:
                self._vec(f"mask{i}", np.ones(m))
            self.geo.append((m, n_p, m_p))
        self.out_index = np.arange(L[-1]["W"].shape[0])

    def encode_input(self, x):
        n_p = self.geo[0][1]
        xp = np.zeros(n_p)
        xp[: len(x)] = x
        return np.concatenate([xp, xp])  # cíclico: rot(x, d) válido para d < n_p

    def run(self, be, pts, ct):
        L = self.layers
        for i, (m, n_p, m_p) in enumerate(self.geo):
            acc = None
            for d in range(m_p):
                term = be.mul_pt(be.rot(ct, d), pts[f"W{i}_{d}"])
                acc = term if acc is None else be.add(acc, term)
            s = n_p // 2
            while s >= m_p:  # dobra: soma os blocos de tamanho m_p
                acc = be.add(acc, be.rot(acc, s))
                s //= 2
            acc = be.add_pt(acc, pts[f"bias{i}"])
            if L[i]["act"]:
                acc = self._act(be, pts, acc, i, masked=False)
            if i < len(L) - 1:  # reempacota para o formato cíclico da próxima camada
                acc = be.mul_pt(acc, pts[f"mask{i}"])
                acc = be.add(acc, be.rot(acc, -self.geo[i + 1][1]))
            ct = acc
        return ct


class ReplicatedPlan(Plan):
    name = "replicated"

    def build(self):
        L = self.layers
        self.geo = []
        stride = None  # passo da saída da camada anterior (layout T)
        for i, l in enumerate(L):
            W = l["W"]
            m, n = W.shape
            m_p, n_p = pow2(m), pow2(n)
            last = i == len(L) - 1
            if i % 2 == 0:  # entrada E (bloco B), saída compacta
                B = m_p if i == 0 else stride
                assert B >= m_p, "bloco menor que a saída: layout não suportado"
                self._vec(f"W{i}", {ii * B + j: W[j, ii] for ii in range(n) for j in range(m)})
                valid = np.arange(m)
                self.geo.append(("E", B, n_p, m_p))
            else:  # entrada T (período P), saída com passo P
                P = n_p
                self._vec(f"W{i}", {j * P + ii: W[j, ii] for j in range(m) for ii in range(n)})
                valid = np.arange(m) * P
                stride = P
                self.geo.append(("T", P, n_p, m_p))
            self._vec(f"bias{i}", dict(zip(valid, l["b"])))
            if l["act"]:
                a, b, c = l["act"]
                for key, coef in (("a", a), ("b", b), ("c", c)):
                    self._vec(f"{key}{i}", {s: coef for s in valid})
            elif not last:
                self._vec(f"mask{i}", {s: 1.0 for s in valid})
            if last:
                self.out_index = valid

    def encode_input(self, x):
        _, B, n_p, _ = self.geo[0]
        v = np.zeros(n_p * B)
        for i, xi in enumerate(x):
            v[i * B: (i + 1) * B] = xi
        return v

    def run(self, be, pts, ct):
        L = self.layers
        for i, (kind, size, n_p, m_p) in enumerate(self.geo):
            acc = be.mul_pt(ct, pts[f"W{i}"])
            steps = [size * 2 ** t for t in range(int(np.log2(n_p)))] if kind == "E" \
                else [2 ** t for t in range(int(np.log2(size)))]
            for s in steps:  # rotate-and-sum
                acc = be.add(acc, be.rot(acc, s))
            acc = be.add_pt(acc, pts[f"bias{i}"])
            if L[i]["act"]:
                acc = self._act(be, pts, acc, i, masked=True)
            elif i < len(L) - 1:
                acc = be.mul_pt(acc, pts[f"mask{i}"])
            if i < len(L) - 1:  # prepara o layout da próxima camada (só rotações)
                nxt_kind, nxt_size, _, nxt_m_p = self.geo[i + 1]
                if nxt_kind == "T":  # compacta -> ladrilhada: replica nxt_m_p blocos de período P
                    s = nxt_size
                    while s < nxt_m_p * nxt_size:
                        acc = be.add(acc, be.rot(acc, -s))
                        s *= 2
                else:  # passo P -> expandida: preenche nxt_m_p posições de cada bloco
                    s = 1
                    while s < nxt_m_p:
                        acc = be.add(acc, be.rot(acc, -s))
                        s *= 2
            ct = acc
        return ct


PLANS = {"diagonal": DiagonalPlan, "replicated": ReplicatedPlan}
