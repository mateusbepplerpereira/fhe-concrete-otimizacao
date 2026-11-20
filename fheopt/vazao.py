"""Modo vazão: uma amostra por slot (empacotamento por atributo), sem rotações.

Layout: o ciphertext j guarda o atributo j de S amostras (S = N/2 slots). Uma camada densa vira,
para cada neurônio k, uma soma ponderada de ciphertexts por escalares (Σ_j W[k,j]·ct_j + b_k):
nenhuma rotação e nenhuma chave de rotação.

Passe de profundidade: a ativação a·h² + b·h + c é reescrita como a·g + c, com g = h·(h + b/a).
O fator a e o termo c são absorvidos nos pesos e no viés da camada seguinte, de modo que cada
ativação custa 1 nível (em vez de 2). Profundidade total: 2·(camadas ocultas) + 1.
"""

from __future__ import annotations

import time

import numpy as np


def fold(layers):
    """Reescreve as camadas absorvendo (a, c) de cada ativação na camada seguinte.

    Retorna lista de (W, b, beta) em que a camada calcula h = W·x + b e, se beta não for None,
    a saída é g = h·(h + beta).
    """
    out, scale, shift = [], None, None
    for l in layers:
        W, b = np.asarray(l["W"], np.float64), np.asarray(l["b"], np.float64)
        if scale is not None:  # entrada desta camada é a·g + c
            b = b + W @ shift
            W = W * scale[None, :]
        if l["act"]:
            a, bb, c = l["act"]
            if a == 0:
                raise ValueError("ativação linear (a=0) não suportada neste passe")
            out.append((W, b, bb / a))
            scale, shift = np.full(W.shape[0], a), np.full(W.shape[0], c)
        else:
            out.append((W, b, None))
            scale = shift = None
    return out


def reference(folded, X):
    h = np.asarray(X, np.float64)
    for W, b, beta in folded:
        h = h @ W.T + b
        if beta is not None:
            h = h * (h + beta)
    return h


def depth(folded):
    return sum(1 + (beta is not None) for _, _, beta in folded)


class VazaoCKKS:
    def __init__(self, layers, scale_bits=40, first_mod_bits=60):
        import openfhe as fhe

        self.folded = fold(layers)
        self.depth = depth(self.folded)
        p = fhe.CCParamsCKKSRNS()
        p.SetMultiplicativeDepth(self.depth)
        p.SetScalingModSize(scale_bits)
        p.SetFirstModSize(first_mod_bits)
        p.SetSecurityLevel(fhe.SecurityLevel.HEStd_128_classic)
        cc = fhe.GenCryptoContext(p)
        for feat in (fhe.PKESchemeFeature.PKE, fhe.PKESchemeFeature.KEYSWITCH, fhe.PKESchemeFeature.LEVELEDSHE,
                     fhe.PKESchemeFeature.ADVANCEDSHE):
            cc.Enable(feat)
        t0 = time.perf_counter()
        self.keys = cc.KeyGen()
        cc.EvalMultKeyGen(self.keys.secretKey)
        self.keygen_s = time.perf_counter() - t0
        self.cc = cc
        self.ring_dim = cc.GetRingDimension()
        self.slots = self.ring_dim // 2
        n_scalar = sum(W.size for W, _, _ in self.folded)
        n_ctct = sum(W.shape[0] for W, _, beta in self.folded if beta is not None)
        self.stats = {"layout": "vazão (1 amostra/slot)", "depth": self.depth, "ring_dim": self.ring_dim,
                      "slots": self.slots, "rotations_per_inference": 0, "rot_keys": 0,
                      "scalar_mults_per_batch": n_scalar, "ct_ct_mults_per_batch": n_ctct,
                      "input_ciphertexts": self.folded[0][0].shape[1]}

    def encrypt(self, Xb):
        cc, pk = self.cc, self.keys.publicKey
        return [cc.Encrypt(pk, cc.MakeCKKSPackedPlaintext(Xb[:, j].tolist())) for j in range(Xb.shape[1])]

    def evaluate(self, cts):
        cc = self.cc
        for W, b, beta in self.folded:
            nxt = []
            for k in range(W.shape[0]):
                h = cc.EvalAdd(cc.EvalLinearWSum(cts, W[k].tolist()), float(b[k]))
                if beta is not None:
                    h = cc.EvalMult(h, cc.EvalAdd(h, float(beta)))
                nxt.append(h)
            cts = nxt
        return cts

    def decrypt(self, cts, n):
        out = []
        for ct in cts:
            pt = self.cc.Decrypt(ct, self.keys.secretKey)
            pt.SetLength(n)
            out.append(pt.GetRealPackedValue())
        return np.array(out).T

    def run_batch(self, Xb):
        """Cifra, avalia e decifra um lote de até `slots` amostras; retorna (logits, tempos)."""
        t0 = time.perf_counter()
        cts = self.encrypt(Xb)
        t1 = time.perf_counter()
        out = self.evaluate(cts)
        t2 = time.perf_counter()
        logits = self.decrypt(out, len(Xb))
        t3 = time.perf_counter()
        return logits, {"encrypt_s": t1 - t0, "eval_s": t2 - t1, "decrypt_s": t3 - t2, "total_s": t3 - t0}
