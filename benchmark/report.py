"""Relatórios do benchmark.

python -m benchmark.report entrega1                 # regenera results/entrega1/report.md
python -m benchmark.report entrega1 entrega2 ...    # comparativo entre entregas (results/comparativo.md)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_ = Path(__file__).resolve().parents[1]
if str(ROOT_) not in sys.path:
    sys.path.insert(0, str(ROOT_))
from benchmark.approaches import BASELINES, OURS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "results"
FAMILY_NAMES = {"linear": "Modelos lineares", "tree": "Árvores", "nn": "Redes neurais", "knn": "Vizinhos mais próximos"}
STATUS = {"nao_suportado": "não suportado", "erro": "erro", "timeout": "timeout"}


def _ms(s, ref=None):
    ref = s if ref is None else ref
    if ref < 1e-2:
        return f"{s * 1e3:.3f}"
    return f"{s * 1e3:,.1f}".replace(",", " ") if ref < 10 else f"{s:,.2f} s".replace(",", " ")


def _lat(c):
    m = c["lat_mean_s"]
    return f"{_ms(m)} ± {_ms(c['lat_std_s'], m)}" if c["status"] == "ok" else "—"


def _best_baseline(cases):

    ok = [c for c in cases if c["approach"] in BASELINES and c["status"] == "ok"]
    return min(ok, key=lambda c: c["lat_mean_s"]) if ok else None


def render(data: dict) -> str:
    meta, cases = data["meta"], data["cases"]
    v = meta.get("versions", {})
    out = [f"# Benchmark: {meta['tag']}", "",
           f"Atualizado em {meta['updated_at']} · commit `{meta['commit']}` · {meta['machine']} · "
           f"{meta['omp_threads']} threads · concrete-ml {v.get('concrete.ml', '?')} · openfhe {v.get('openfhe', '?')}",
           "",
           "Latência: média ± desvio por amostra (cifrar, avaliar e decifrar), após aquecimento. Acurácia FHE: "
           "simulação FHE no Concrete ML, execução real no CKKS. Ganho: em relação à linha de base Concrete ML "
           "mais rápida do mesmo cenário.", ""]
    by_sc = {}
    for c in cases:
        by_sc.setdefault(c["scenario"], []).append(c)

    summary, summary_v = [], []
    for fam, fam_name in FAMILY_NAMES.items():
        scs = [s for s, cs in by_sc.items() if cs[0]["family"] == fam]
        if not scs:
            continue
        out += [f"## {fam_name}", "",
                "| Cenário | Abordagem | Acurácia (claro) | Acurácia (FHE) | Latência (ms) | Ganho | Observação |",
                "|---|---|---|---|---|---|---|"]
        for s in scs:
            cs = sorted(by_sc[s], key=lambda c: c["approach"])
            base = _best_baseline(cs)
            for c in cs:
                if c["status"] == "ok":
                    gain = f"{base['lat_mean_s'] / c['lat_mean_s']:.1f}×" if base else "—"
                    st = c.get("stats", {})
                    obs = _obs(c, st)
                    row = [s, c["approach"], f"{c['acc_clear']:.3f}", f"{c['acc_fhe']:.3f}", _lat(c), gain, obs]
                else:
                    row = [s, c["approach"], "—", "—", "—", "—", f"{STATUS[c['status']]}: {c.get('reason', '')}"]
                out.append("| " + " | ".join(row) + " |")
            for kind, bucket in (("consulta", summary), ("amortizada", summary_v)):
                ours = [c for c in cs if c["approach"] in OURS and c["status"] == "ok"
                        and (c.get("lat_kind") == "amortizada") == (kind == "amortizada")]
                if base and ours:
                    bucket.append((s, base, min(ours, key=lambda c: c["lat_mean_s"])))
        out.append("")

    for title, rows, note in (
            ("Resumo: latência por consulta (1 amostra)", summary,
             "Melhor abordagem deste projeto × melhor linha de base Concrete ML, uma consulta por vez."),
            ("Resumo: vazão (custo amortizado por amostra)", summary_v,
             "Modo vazão do motor CKKS (lote de N/2 amostras) × melhor linha de base Concrete ML. O Concrete ML "
             "processa uma amostra por vez (sem empacotamento SIMD), então o custo dele por amostra é a latência.")):
        if not rows:
            continue
        out += [f"## {title}", "", note, "",
                "| Cenário | Melhor Concrete ML | Latência (ms) | Melhor deste projeto | Latência (ms) | Ganho | Δ acurácia (FHE) |",
                "|---|---|---|---|---|---|---|"]
        for s, base, b in rows:
            out.append(f"| {s} | {base['approach']} | {_lat(base)} | {b['approach']} | {_lat(b)} | "
                       f"{base['lat_mean_s'] / b['lat_mean_s']:,.1f}× | {(b['acc_fhe'] - base['acc_fhe']) * 100:+.1f} p.p. |"
                       .replace(",", " "))
        out.append("")
    return "\n".join(out)


def _obs(c, st):
    if c.get("lat_kind") == "amortizada":
        return (f"vazão: lote de {c['batch_size']} em {c['batch_latency_s']:.2f} s; N={st['ring_dim']}, "
                f"prof. {st['depth']}, 0 rotações")
    if "ring_dim" in st:
        return f"N={st['ring_dim']}, prof. {st['depth']}, {st['rotations_per_inference']} rotações"
    if "chosen" in c:
        ch, ref = c["chosen"], c["reference"]
        gpe = ch["global_p_error"] if ch["global_p_error"] is not None else "padrão"
        return f"{st.get('pbs', '?')} PBS; escolhido n_bits={ch['n_bits']}, p_error={gpe} (padrão: {ref['pbs']} PBS)"
    return f"{st.get('pbs', '?')} PBS"


def compare(tags: list[str]) -> str:
    runs = {t: json.loads((OUT_ROOT / t / "results.json").read_text()) for t in tags}
    keys = []
    for t in tags:
        for c in runs[t]["cases"]:
            k = (c["scenario"], c["approach"])
            if k not in keys:
                keys.append(k)
    out = ["# Comparativo entre entregas", "", "Latência por amostra (ms) e acurácia FHE de cada caso em cada rodada.", "",
           "| Cenário | Abordagem | " + " | ".join(tags) + " |", "|---|---|" + "---|" * len(tags)]
    for s, a in sorted(keys):
        cells = []
        for t in tags:
            c = next((c for c in runs[t]["cases"] if c["scenario"] == s and c["approach"] == a), None)
            cells.append("—" if not c else (f"{_ms(c['lat_mean_s'])} ({c['acc_fhe']:.3f})" if c["status"] == "ok"
                                            else STATUS[c["status"]]))
        out.append(f"| {s} | {a} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main():
    tags = sys.argv[1:]
    if not tags:
        sys.exit(__doc__)
    if len(tags) == 1:
        data = json.loads((OUT_ROOT / tags[0] / "results.json").read_text())
        (OUT_ROOT / tags[0] / "report.md").write_text(render(data))
        print(f"results/{tags[0]}/report.md")
    else:
        (OUT_ROOT / "comparativo.md").write_text(compare(tags))
        print("results/comparativo.md")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
