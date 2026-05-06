# common.py — helpers compartidos por todos los scripts de experimento
import subprocess, os, time, json, re
from pathlib import Path

# ── Rutas ────────────────────────────────────────────────────────────────────
SRC = Path(__file__).parent          # directorio src/
OUT = SRC / "resultados"
OUT.mkdir(exist_ok=True)

# ── Parámetros globales ───────────────────────────────────────────────────────
SCENES  = ["scene.txt", "scene_many.txt"]
THREADS = [1, 2, 4, 8]
DEPTH   = 8

# (W, H, S_supersampling, N_samples, nombre)
CONFIGS = [
    (400,  300,  2,  32,  "small"),
    (800,  600,  4, 128,  "medium"),
    (1600, 1200, 4, 256,  "large"),
]

# ── Runner ────────────────────────────────────────────────────────────────────
def run(cmd, env_extra=None):
    """
    Ejecuta cmd (lista), devuelve (stdout+stderr, elapsed_seconds).
    Imprime el tiempo en vivo para saber que el proceso no se colgó.
    """
    env = os.environ.copy()
    if env_extra:
        env.update({k: str(v) for k, v in env_extra.items()})

    cmd = [str(c) for c in cmd]
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.perf_counter() - t0

    output = result.stdout + result.stderr
    print(f"    [{elapsed:7.2f}s]  {' '.join(cmd[-6:])}")   # últimos 6 tokens
    if result.returncode != 0:
        print(f"    ⚠ returncode={result.returncode}")
        print(result.stderr[-300:])

    return output, elapsed

def parse_time(output):
    """Extrae el tiempo del stderr del binario: 'Tiempo      : X.XXX s'"""
    m = re.search(r"Tiempo\s*:\s*([\d.]+)", output)
    return float(m.group(1)) if m else None

# ── Persistencia ─────────────────────────────────────────────────────────────
def save(name, data):
    path = OUT / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → {path}")

def load(name):
    with open(OUT / f"{name}.json") as f:
        return json.load(f)
