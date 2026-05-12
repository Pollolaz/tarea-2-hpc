"""
EXP-4 — Numba: escalabilidad completa (partes c/d/e)
======================================================
Corre numba_pt.py en las 3 configuraciones × 2 escenas × p in {1,2,4,8}.
Hace un warmup previo para forzar la compilación JIT de Numba y no contaminar
los tiempos medidos.

numba_pt.py debe aceptar args:
    python numba_pt.py <scene> <out.png> <W> <H> <D> <N>

El número de threads se controla con la variable de entorno NUMBA_NUM_THREADS.

⚠  El tiempo que reporta el binario DEBE excluir la compilación JIT.
   Medir solo el tiempo del render() ya compilado.
   Si numba_pt.py no hace eso, ajustar el script para descartar la primera
   corrida (warmup) y usar la segunda como medición real.

Tiempo estimado: 1–2 horas (más ~30s de warmup JIT).

Uso:
    python exp4_numba_scaling.py
"""

from common import *

NUMBA_SCRIPT = SRC / "numba_pt.py"

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not NUMBA_SCRIPT.exists():
        print(f"ERROR: no se encontró {NUMBA_SCRIPT}")
        return

    # ── Warmup: fuerza compilación JIT antes de medir ─────────────────────────
    print("Warmup Numba (compilación JIT, puede tardar ~30s)...")
    W0, H0, S0, N0, _ = CONFIGS[0]   # config small, 1 thread
    run(["python", NUMBA_SCRIPT, "scene.txt", OUT/"warmup.png", W0, H0, DEPTH, N0],
        env_extra={"NUMBA_NUM_THREADS": "1"})
    print("  Warmup listo.\n")

    results = {}   # results[scene][cfg_name][str(p)] = segundos

    for scene in SCENES:
        results[scene] = {}
        print(f"\n{'='*60}")
        print(f"  Escena: {scene}")
        print(f"{'='*60}")

        for W, H, S, N, cfg_name in CONFIGS:
            results[scene][cfg_name] = {}
            print(f"\n  Config {cfg_name}: {W}x{H}  N={N}")

            for p in THREADS:
                print(f"    p={p}", end="  ", flush=True)

                out, wall = run(
                    ["python", NUMBA_SCRIPT, scene, OUT/"out_numba.png", W, H, DEPTH, N],
                    env_extra={"NUMBA_NUM_THREADS": str(p)}
                )
                # numba_pt.py debería imprimir "Tiempo: X.XXX s" en stderr/stdout
                # Si no lo hace, usamos el wall time del subprocess (menos preciso)
                t = parse_time(out) or wall
                results[scene][cfg_name][str(p)] = t

    save("exp4_numba_scaling", results)

    # ── Speedup y eficiencia ──────────────────────────────────────────────────
    try:
        baseline = load("exp1_baseline")
    except FileNotFoundError:
        print("\nAviso: exp1_baseline.json no encontrado, no se calcula speedup")
        return

    print("\n\n" + "="*60)
    print("  SPEEDUP Numba S(p) = Ts / T(p)")
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
