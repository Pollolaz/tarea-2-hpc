"""
EXP-2 — OpenMP: estrategias de scheduling (parte b)
=====================================================
Corre ./omp_pt_sched con static/dynamic/guided, distintos chunksizes,
p in {1,2,4,8} threads, en config MEDIUM y ambas escenas.

El binario omp_pt_sched debe aceptar args:
    ./omp_pt_sched <scene> <out.ppm> <W> <H> <D> <N> <threads> <schedule> [chunksize]

  schedule: "static" | "dynamic" | "guided"
  chunksize: entero (ignorado si guided)

En C++ usar omp_set_schedule() antes del pragma:
    omp_sched_t sched = omp_sched_static;  // según argv[8]
    omp_set_schedule(sched, chunk);
    #pragma omp parallel for schedule(runtime) num_threads(threads)

Tiempo estimado: 2–4 horas.

Uso:
    python exp2_scheduling.py
"""

from common import *

# ── Configuración del experimento ─────────────────────────────────────────────
W, H, S, N, _ = CONFIGS[1]   # siempre config MEDIUM (800x600, N=128)

# (schedule_name, lista_de_chunksizes)
# None en chunksize → no se pasa el arg (guided lo ignora igual)
STRATEGIES = [
    ("static",  [1, 8, 32, 64, 128]),
    ("dynamic", [1, 8, 32, 64, 128]),
    ("guided",  [None]),
]

BIN = bin("omp_pt_sched")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not BIN.exists():
        print(f"ERROR: no se encontró {BIN}  →  compilar con: make omp_pt_sched")
        return

    results = {}   # results[scene][schedule][str(chunk)][str(p)] = segundos

    for scene in SCENES:
        results[scene] = {}
        print(f"\n{'='*60}")
        print(f"  Escena: {scene}  |  Config MEDIUM {W}x{H} N={N}")
        print(f"{'='*60}")

        for sched_name, chunks in STRATEGIES:
            results[scene][sched_name] = {}
            print(f"\n  Schedule: {sched_name}")

            for chunk in chunks:
                chunk_key = str(chunk) if chunk is not None else "default"
                results[scene][sched_name][chunk_key] = {}

                for p in THREADS:
                    label = f"    {sched_name:<8} chunk={chunk_key:<8} p={p}"
                    print(label, end="  ", flush=True)

                    cmd = [BIN, scene, OUT/"out_sched.ppm",
                           W, H, DEPTH, N, p, sched_name]
                    if chunk is not None:
                        cmd.append(chunk)

                    out, wall = run(cmd)
                    t = parse_time(out) or wall
                    results[scene][sched_name][chunk_key][str(p)] = t

    save("exp2_scheduling", results)

    # ── Resumen: mejor estrategia por escena ──────────────────────────────────
    print("\n\n" + "="*60)
    print("  RESUMEN — Mejor tiempo con p=8 (config MEDIUM)")
    print("="*60)
    for scene in SCENES:
        print(f"\n  {scene}:")
        best_t = float("inf")
        best_label = ""
        for sched_name, chunks in STRATEGIES:
            for chunk in chunks:
                chunk_key = str(chunk) if chunk is not None else "default"
                t = results[scene][sched_name][chunk_key].get("8", float("inf"))
                flag = " ← MEJOR" if t < best_t else ""
                print(f"    {sched_name:<8} chunk={chunk_key:<8}  p=8 → {t:.2f}s{flag}")
                if t < best_t:
                    best_t = t
                    best_label = f"{sched_name} chunk={chunk_key}"
        print(f"  → Ganador: {best_label}  ({best_t:.2f}s)")

    print("\nUsar estos valores en exp3_omp_scaling.py → BEST_SCHED / BEST_CHUNK")

if __name__ == "__main__":
    main()
