"""
EXP-3 — OpenMP: escalabilidad completa (partes d/e)
=====================================================
Corre ./omp_pt (compilado con la mejor estrategia de EXP-2) en las 3
configuraciones × 2 escenas × p in {1,2,4,8}.

Necesita exp1_baseline.json para calcular speedup/eficiencia.

El binario omp_pt acepta args:
    ./omp_pt <scene> <out.ppm> <W> <H> <D> <N> <threads>

⚠  Antes de correr: ajustar BEST_SCHED y BEST_CHUNK según EXP-2.
   También recompilar omp_pt con esa estrategia hardcodeada, o usar
   omp_pt_sched pasándole los args correspondientes.

Tiempo estimado: 2–3 horas.

Uso:
    python exp3_omp_scaling.py
"""

from common import *

# ── Ajustar según resultado de EXP-2 ─────────────────────────────────────────
# Si usas omp_pt_sched en vez de omp_pt, cambia BIN y agrega BEST_SCHED/CHUNK a cmd.
BIN        = bin("omp_pt")   # o "omp_pt" si lo recompilaste con la mejor estrategia hardcodeada
BEST_SCHED = "guided"   # ← cambiar según EXP-2
BEST_CHUNK = None       # ← cambiar (None → no se pasa arg)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not BIN.exists():
        print(f"ERROR: no se encontró {BIN}  →  compilar con: make omp_pt")
        return

    results = {}   # results[scene][cfg_name][str(p)] = segundos

    for scene in SCENES:
        results[scene] = {}
        print(f"\n{'='*60}")
        print(f"  Escena: {scene}")
        print(f"{'='*60}")

        for W, H, S, N, cfg_name in CONFIGS:
            results[scene][cfg_name] = {}
            print(f"\n  Config {cfg_name}: {W}x{H}  S={S}  N={N}")

            for p in THREADS:
                print(f"    p={p}", end="  ", flush=True)

                cmd = [BIN, scene, OUT/"out_omp.ppm", W, H, DEPTH, N, p]
                if BEST_SCHED and BIN.name == "omp_pt_sched":
                    cmd.append(BEST_SCHED)
                    if BEST_CHUNK is not None:
                        cmd.append(BEST_CHUNK)

                out, wall = run(cmd)
                t = parse_time(out) or wall
                results[scene][cfg_name][str(p)] = t

    save("exp3_omp_scaling", results)

    # ── Speedup y eficiencia ──────────────────────────────────────────────────
    try:
        baseline = load("exp1_baseline")
    except FileNotFoundError:
        print("\nAviso: exp1_baseline.json no encontrado, no se calcula speedup")
        return

    print("\n\n" + "="*60)
    print("  SPEEDUP OpenMP S(p) = Ts / T(p)")
    print("="*60)
    for scene in SCENES:
        print(f"\n  {scene}:")
        for _, _, _, _, cfg_name in CONFIGS:
            Ts = baseline[scene][cfg_name]["serial_pt"]
            row = f"    {cfg_name:<8}  Ts={Ts:.1f}s  →"
            for p in THREADS:
                T  = results[scene][cfg_name][str(p)]
                sp = Ts / T
                ef = sp / p
                row += f"  p={p}: S={sp:.2f} E={ef:.2f}"
            print(row)

if __name__ == "__main__":
    main()
