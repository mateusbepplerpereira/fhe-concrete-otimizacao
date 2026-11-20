"""Backends de execução do motor.

Os passes geram um programa sobre slots (soma, multiplicação, rotação). O mesmo programa roda em:
  - SlotSim: numpy, sem criptografia; valida layouts e conta operações/profundidade;
  - OpenFHEBackend: CKKS real (OpenFHE).
Convenção de rotação (igual ao OpenFHE): rot(v, k)[i] = v[i + k] (k > 0 gira para a esquerda).
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass
class SimCt:
    v: np.ndarray
    level: int = 0


class SlotSim:
    """Simulador em claro: mesma semântica de slots do CKKS, conta operações e níveis."""

    def __init__(self, slots: int):
        self.slots = slots
        self.ops = Counter()

    def encrypt(self, vec):
        v = np.zeros(self.slots)
        v[: len(vec)] = vec
        return SimCt(v)

    def decrypt(self, ct, n):
        return ct.v[:n].copy()

    def encode(self, vec):
        v = np.zeros(self.slots)
        v[: len(vec)] = vec
        return v

    def add(self, a, b):
        self.ops["add"] += 1
        return SimCt(a.v + b.v, max(a.level, b.level))

    def add_pt(self, a, p):
        self.ops["add_pt"] += 1
        return SimCt(a.v + p, a.level)

    def add_scalar(self, a, s):
        self.ops["add_scalar"] += 1
        return SimCt(a.v + s, a.level)

    def mul_pt(self, a, p):
        self.ops["mul_pt"] += 1
        return SimCt(a.v * p, a.level + 1)

    def mul_scalar(self, a, s):
        self.ops["mul_scalar"] += 1
        return SimCt(a.v * s, a.level + 1)

    def mul_ct(self, a, b):
        self.ops["mul_ct"] += 1
        return SimCt(a.v * b.v, max(a.level, b.level) + 1)

    def rot(self, a, k):
        if k % self.slots == 0:
            return a
        self.ops["rot"] += 1
        return SimCt(np.roll(a.v, -k), a.level)


class OpenFHEBackend:
    """CKKS (RNS) com OpenFHE, segurança de 128 bits (HEStd_128_classic)."""

    def __init__(self, slots, depth, rotations, scale_bits=40, first_mod_bits=60, ring_dim=None):
        import openfhe as fhe

        self.fhe = fhe
        self.slots = slots
        p = fhe.CCParamsCKKSRNS()
        p.SetMultiplicativeDepth(depth)
        p.SetScalingModSize(scale_bits)
        p.SetFirstModSize(first_mod_bits)
        p.SetBatchSize(slots)
        p.SetSecurityLevel(fhe.SecurityLevel.HEStd_128_classic)
        if ring_dim:
            p.SetRingDim(ring_dim)
        cc = fhe.GenCryptoContext(p)
        for feat in (fhe.PKESchemeFeature.PKE, fhe.PKESchemeFeature.KEYSWITCH,
                     fhe.PKESchemeFeature.LEVELEDSHE):
            cc.Enable(feat)
        t0 = time.perf_counter()
        self.keys = cc.KeyGen()
        cc.EvalMultKeyGen(self.keys.secretKey)
        rots = sorted({int(r) for r in rotations if r % slots})
        if rots:
            cc.EvalRotateKeyGen(self.keys.secretKey, rots)
        self.keygen_s = time.perf_counter() - t0
        self.cc = cc
        self.ring_dim = cc.GetRingDimension()
        self.n_rot_keys = len(rots)
        self.ops = Counter()

    def encrypt(self, vec):
        pt = self.cc.MakeCKKSPackedPlaintext(list(map(float, vec)))
        return self.cc.Encrypt(self.keys.publicKey, pt)

    def decrypt(self, ct, n):
        pt = self.cc.Decrypt(ct, self.keys.secretKey)
        pt.SetLength(n)
        return np.array(pt.GetRealPackedValue())

    def encode(self, vec):
        return self.cc.MakeCKKSPackedPlaintext(list(map(float, vec)))

    def add(self, a, b):
        self.ops["add"] += 1
        return self.cc.EvalAdd(a, b)

    def add_pt(self, a, p):
        self.ops["add_pt"] += 1
        return self.cc.EvalAdd(a, p)

    def add_scalar(self, a, s):
        self.ops["add_scalar"] += 1
        return self.cc.EvalAdd(a, float(s))

    def mul_pt(self, a, p):
        self.ops["mul_pt"] += 1
        return self.cc.EvalMult(a, p)

    def mul_scalar(self, a, s):
        self.ops["mul_scalar"] += 1
        return self.cc.EvalMult(a, float(s))

    def mul_ct(self, a, b):
        self.ops["mul_ct"] += 1
        return self.cc.EvalMult(a, b)

    def rot(self, a, k):
        if k % self.slots == 0:
            return a
        self.ops["rot"] += 1
        return self.cc.EvalRotate(a, int(k))
