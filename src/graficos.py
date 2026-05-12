"""
graficos.py — Genera todas las figuras del informe
====================================================
Lee los JSON producidos por exp1–exp4 y guarda PDFs en resultados/.

Figuras generadas:
  fig_<escena>_<config>.pdf  →  T(p), S(p), E(p) para OpenMP y Numba
  fig_scheduling_<escena>.pdf →  heatmap de tiempos por estrategia (EXP-2)

Uso:
    python graficos.py
"""

import json, sys
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

SRC = Path(__file__).parent
OUT = SRC / "resultados"

SCENES   = ["scene.txt", "scene_many.txt"]
SCENE_LABELS = {"scene.txt": "4 esferas", "scene_many.txt": "40 esferas"}
CONFIGS  = [("small", "400×300  N=32"), ("medium", "800×600  N=128"), ("large", "1600×1200  N=256")]
THREADS  = [1, 2, 4, 8]

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "legend.fontsize": 9})

# ── Helpers ───────────────────────────────────────────────────────────────────
def load(name):
    path = OUT / f"{name}.json"
    if not path.exists():
        print(f"  ⚠ no encontrado: {path}")
        return None
    with open(path) as f:
        return json.load(f)

def savefig(fig, name):
    path = OUT / f"{name}.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  → {path}")
    plt.close(fig)

# ── Fig 1: T(p), S(p), E(p) por config y escena ──────────────────────────────
def fig_scaling():
    baseline = load("exp1_baseline")
    omp      = load("exp3_omp_scaling")
    numba    = load("exp4_numba_scaling")

    if not all([baseline, omp, numba]):
        print("  Saltando fig_scaling (faltan JSONs)")
        return

    for scene in SCENES:
        scene_tag = scene.replace(".txt", "").replace("_", "")
        for cfg_key, cfg_label in CONFIGS:
            Ts = baseline[scene][cfg_key]["serial_pt"]

            T_omp   = [omp[scene][cfg_key][str(p)]   for p in THREADS]
            T_numba = [numba[scene][cfg_key][str(p)]  for p in THREADS]

            S_omp   = [Ts / T for T in T_omp]
            S_numba = [Ts / T for T in T_numba]
            E_omp   = [s / p  for s, p in zip(S_omp,   THREADS)]
            E_numba = [s / p  for s, p in zip(S_numba, THREADS)]

            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
            fig.suptitle(f"{cfg_label}  —  {SCENE_LABELS[scene]}", fontsize=13, fontweight="bold")

            # T(p)
            ax = axes[0]
            ax.plot(THREADS, T_omp,   "o-",  color="steelblue", label="OpenMP")
            ax.plot(THREADS, T_numba, "s--", color="darkorange", label="Numba")
            ax.axhline(Ts, color="gray", linestyle=":", linewidth=1.5, label=f"Serial ({Ts:.3f}s)")
            ax.plot(THREADS, [Ts/p for p in THREADS], "k--", linewidth=1, label="Ideal")
            ax.set(xlabel="Threads p", ylabel="Tiempo (s)", title="T(p)")
            ax.set_xticks(THREADS)
            ax.legend(); ax.grid(True, alpha=0.4)

            # S(p)
            ax = axes[1]
            ax.plot(THREADS, S_omp,   "o-",  color="steelblue", label="OpenMP")
            ax.plot(THREADS, S_numba, "s--", color="darkorange", label="Numba")
            ax.plot(THREADS, THREADS, "k--", linewidth=1, label="Ideal")
            ax.set(xlabel="p", ylabel="Speedup  S(p) = Ts / T(p)", title="Speedup")
            ax.set_xticks(THREADS)
            ax.legend(); ax.grid(True, alpha=0.4)

            # E(p)
            ax = axes[2]
            ax.plot(THREADS, E_omp,   "o-",  color="steelblue", label="OpenMP")
            ax.plot(THREADS, E_numba, "s--", color="darkorange", label="Numba")
            ax.axhline(1.0, color="k", linestyle="--", linewidth=1, label="Ideal")
            ax.set(xlabel="p", ylabel="Eficiencia  E(p) = S(p) / p", title="Eficiencia")
            ax.set_xticks(THREADS)
            ax.set_ylim(0, 1.15)
            ax.legend(); ax.grid(True, alpha=0.4)

            plt.tight_layout()
            savefig(fig, f"fig_{scene_tag}_{cfg_key}")

# ── Fig 2: heatmap scheduling (EXP-2) ────────────────────────────────────────
def fig_scheduling():
    data = load("exp2_scheduling")
    if not data:
        print("  Saltando fig_scheduling (falta exp2_scheduling.json)")
        return

    for scene in SCENES:
        scene_tag = scene.replace(".txt", "").replace("_", "")
        strategies = data[scene]

        # Construir matriz: filas = (sched, chunk), columnas = threads
        labels, matrix = [], []
        for sched_name, chunks in strategies.items():
            for chunk_key, by_p in chunks.items():
                labels.append(f"{sched_name}\nchunk={chunk_key}")
                matrix.append([by_p.get(str(p), float("nan")) for p in THREADS])

        mat = np.array(matrix)

        fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.55)))
        im = ax.imshow(mat, aspect="auto", cmap="YlGnBu_r")
        plt.colorbar(im, ax=ax, label="Tiempo (s)")

        ax.set_xticks(range(len(THREADS)))
        ax.set_xticklabels([f"p={p}" for p in THREADS])
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(f"Scheduling — {SCENE_LABELS[scene]}  (config MEDIUM 800×600 N=128)",
                     fontweight="bold")

        # Anotar valores
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                            color="black" if v > mat.max() * 0.5 else "white")

        plt.tight_layout()
        savefig(fig, f"fig_scheduling_{scene_tag}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generando figuras...\n")
    fig_scaling()
    fig_scheduling()
    print("\nListo.")
