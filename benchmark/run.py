"""Runner do benchmark: avalia cenários fixos × abordagens e gera resultados por entrega.

Exemplos:
  python -m benchmark.run --tag entrega1                       # tudo
  python -m benchmark.run --tag entrega1 --families nn linear  # só algumas famílias
  python -m benchmark.run --tag entrega2 --approaches fheopt   # só a nova versão do motor
  python -m benchmark.run --tag teste --quick                  # rodada rápida de verificação
  python -m benchmark.run --list                               # lista cenários e abordagens

Cada fase de cada caso roda em subprocesso isolado (com timeout), chamando este mesmo módulo com
--worker. Casos já concluídos numa tag são reaproveitados (use --force para refazer).
Saída: results/<tag>/{results.json, report.md, cases/}.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "results"


def _git_commit():
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "fheopt", "benchmark"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return rev.strip() + ("-modificado" if dirty else "")
    except OSError:
        return "?"


def _env():
    env = {**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "8"), "PYTHONUNBUFFERED": "1",
           "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    tools = ROOT / "tools" / "bin"  # contorno do linker no macOS (ver env.sh)
    if tools.exists():
        env["PATH"] = f"{tools}{os.pathsep}{env['PATH']}"
    return env


def _versions():
    out = {"python": platform.python_version()}
    for mod in ("concrete.ml", "openfhe", "sklearn", "torch"):
        code = f"import {mod} as m; print(getattr(m, '__version__', '?'))"
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=_env())
        out[mod] = r.stdout.strip() or "não instalado"
    return out


def run_case(sc, appr, case_dir: Path, args) -> dict:
    base = {"scenario": sc.id, "family": sc.family, "dataset": sc.dataset, "model": sc.model,
            "description": sc.description, "approach": appr.name, "approach_description": appr.description}
    ok, why = appr.supports(sc)
    if not ok:
        return {**base, "status": "nao_suportado", "reason": why}
    case_dir.mkdir(parents=True, exist_ok=True)
    t_case = time.perf_counter()
    for phase in appr.phases:
        cmd = [sys.executable, "-m", "benchmark.run", "--worker", "--scenarios", sc.id, "--approaches", appr.name,
               "--phase", phase, "--case-dir", str(case_dir), "--n-lat", str(args.n_lat), "--n-acc", str(args.n_acc)]
        log = case_dir / f"{phase}.log"
        try:
            with open(log, "w") as fh:
                r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=_env(), cwd=ROOT,
                                   timeout=args.timeout)
        except subprocess.TimeoutExpired:
            return {**base, "status": "timeout", "reason": f"fase {phase} excedeu {args.timeout}s"}
        if r.returncode != 0:
            tail = log.read_text().strip().splitlines()[-1:] or ["?"]
            return {**base, "status": "erro", "reason": f"fase {phase}: {tail[0][:300]}"}
    res = json.loads((case_dir / f"{appr.phases[-1]}.out.json").read_text())
    return {**base, "status": "ok", "wall_s": time.perf_counter() - t_case, **res}


def main():
    from benchmark import report
    from benchmark.approaches import APPROACHES
    from benchmark.scenarios import FAMILIES, SCENARIOS, select

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", help="nome da rodada (ex.: entrega1); define a pasta de saída")
    ap.add_argument("--approaches", nargs="*", default=list(APPROACHES), choices=list(APPROACHES))
    ap.add_argument("--families", nargs="*", choices=FAMILIES)
    ap.add_argument("--scenarios", nargs="*")
    ap.add_argument("--n-lat", type=int, default=0, help="amostras de latência (0 = padrão do cenário)")
    ap.add_argument("--n-acc", type=int, default=0, help="amostras de acurácia FHE (0 = teste inteiro)")
    ap.add_argument("--timeout", type=int, default=1800, help="limite por fase, em segundos")
    ap.add_argument("--force", action="store_true", help="refaz casos já concluídos nesta tag")
    ap.add_argument("--quick", action="store_true", help="verificação rápida: n_lat=2, n_acc=40")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--phase", help=argparse.SUPPRESS)
    ap.add_argument("--case-dir", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:  # executa UMA fase de UM caso (chamado pelo próprio runner)
        import warnings

        warnings.filterwarnings("ignore")
        from benchmark.scenarios import BY_ID

        case_dir = Path(args.case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)
        out = APPROACHES[args.approaches[0]].run_phase(args.phase, BY_ID[args.scenarios[0]], case_dir,
                                                       {"n_lat": args.n_lat, "n_acc": args.n_acc}) or {}
        (case_dir / f"{args.phase}.out.json").write_text(json.dumps(out, indent=2, default=str))
        return

    if args.list:
        print("Cenários:")
        for s in SCENARIOS:
            print(f"  {s.id:24s} {s.family:7s} {s.dataset:14s} {s.description}")
        print("Abordagens:")
        for a in APPROACHES.values():
            print(f"  {a.name:18s} {a.description}")
        return
    if not args.tag:
        ap.error("--tag é obrigatório")
    if args.quick:
        args.n_lat, args.n_acc = args.n_lat or 2, args.n_acc or 40

    out_dir = OUT_ROOT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    res_file = out_dir / "results.json"
    prev = json.loads(res_file.read_text()) if res_file.exists() else {"cases": []}
    done = {(c["scenario"], c["approach"]): c for c in prev["cases"]}

    scenarios = select(args.scenarios, args.families)
    cases = [(s, APPROACHES[a]) for s in scenarios for a in args.approaches]
    print(f"[{args.tag}] {len(cases)} casos | commit {_git_commit()} | OMP_NUM_THREADS={_env()['OMP_NUM_THREADS']}",
          flush=True)
    for i, (sc, appr) in enumerate(cases, 1):
        key = (sc.id, appr.name)
        if key in done and done[key]["status"] in ("ok", "nao_suportado") and not args.force:
            print(f"  ({i}/{len(cases)}) {sc.id} × {appr.name}: reaproveitado ({done[key]['status']})", flush=True)
            continue
        t0 = time.perf_counter()
        case_dir = out_dir / "cases" / f"{sc.id.replace('/', '__')}__{appr.name}"
        r = run_case(sc, appr, case_dir, args)
        r["finished_at"] = dt.datetime.now().isoformat(timespec="seconds")
        done[key] = r
        msg = (f"lat={r['lat_mean_s'] * 1e3:.1f}ms acc_fhe={r['acc_fhe']:.3f}" if r["status"] == "ok"
               else r.get("reason", ""))
        print(f"  ({i}/{len(cases)}) {sc.id} × {appr.name}: {r['status']} {msg} [{time.perf_counter() - t0:.0f}s]",
              flush=True)
        _save(res_file, args, done)

    _save(res_file, args, done)
    data = json.loads(res_file.read_text())
    (out_dir / "report.md").write_text(report.render(data))
    print(f"\nrelatório: {(out_dir / 'report.md').relative_to(ROOT)}")


def _save(res_file, args, done):
    meta = {"tag": args.tag, "updated_at": dt.datetime.now().isoformat(timespec="seconds"), "commit": _git_commit(),
            "machine": f"{platform.machine()} {platform.mac_ver()[0] or platform.platform()}",
            "omp_threads": _env()["OMP_NUM_THREADS"], "n_lat": args.n_lat, "n_acc": args.n_acc}
    if not res_file.exists() or "versions" not in json.loads(res_file.read_text()).get("meta", {}):
        meta["versions"] = _versions()
    else:
        meta["versions"] = json.loads(res_file.read_text())["meta"]["versions"]
    res_file.write_text(json.dumps({"meta": meta, "cases": list(done.values())}, indent=2, default=str))


if __name__ == "__main__":
    main()
