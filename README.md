# Tarea 2 — Path Tracing con OpenMP y Numba
**IIC3533 — Computación de Alto Rendimiento 2026-1**  
**Entrega:** Viernes 15 de Mayo de 2026, 23:59  
**Grupos de 3 personas | Dos computadores distintos**

---

## Resumen de lo que hay que entregar

- `omp_pt.cpp` — path tracer paralelo con OpenMP
- `numba_pt.py` — path tracer paralelo con Numba (`@njit(parallel=True)`)
- Informe PDF con tablas, gráficos y análisis de los experimentos

---

## Fase 0 — Setup del entorno (hacer UNA VEZ por máquina)

> **En Windows: correr TODO desde WSL.** Los binarios C++ son Linux ELF y no funcionan en Windows nativo.
> Abrir una terminal WSL antes de cualquier paso.

### 0a. WSL: navegar al proyecto y verificar compilador

```bash
# El proyecto vive en el filesystem de Windows, accesible desde WSL en /mnt/c/
cd /mnt/c/Users/nicol/OneDrive/Documents/GitHub/tarea-2-hpc

# Verificar que g++ y OpenMP están disponibles
g++ --version
echo "#include <omp.h>" | g++ -fopenmp -x c++ - -o /dev/null && echo "OpenMP OK"

# Si no están instalados:
sudo apt update && sudo apt install -y g++ libomp-dev
```

### 0b. Conda dentro de WSL

```bash
# Si no tienes conda en WSL, instalarlo (Miniconda):
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash
source ~/.bashrc

# Crear entorno y activarlo
conda create -n tarea2-hpc python=3.12 -y
conda activate tarea2-hpc
conda install numpy numba matplotlib -y
```

### 0c. Compilar binarios C++

```bash
conda activate tarea2-hpc
cd /mnt/c/Users/nicol/OneDrive/Documents/GitHub/tarea-2-hpc/src
make all
```

> **Verificar** que `./serial` y `./serial_pt` corren sin errores antes de continuar.

```bash
./serial_pt scene.txt test.ppm 400 300 8 32
# Debe imprimir: Tiempo : X.XX s
```

```bash
# Instalar ImageMagick
sudo apt install imagemagick

# Convertir en imagen
convert test.ppm test.png
```

```bash
# Correr una instancia de cada algoritmo
./serial scene.txt out_serial.ppm 800 600 8 4
./serial_pt scene.txt out_serial_pt.ppm 800 600 8 64
./omp_pt scene.txt out_omp_pt.ppm 800 600 8 64 4
./omp_pt_sched scene.txt out_omp_sched.ppm 800 600 8 64 4
# No implementado todavía
python3 numba_pt.py scene.txt out_numba_pt.ppm 800 600 8 64 

# Convertir en imagen
convert out_serial.ppm out_serial.png 
convert out_serial_pt.ppm out_serial_pt.png
convert out_omp_pt.ppm out_omp_pt.png
convert out_omp_sched.ppm out_omp_sched.png 
# No implementado todavía
convert out_numba_pt.ppm out_numba_pt.png
```

---

## Fase 1 — Implementar el código (antes de correr experimentos)

Hacer esto primero para que los experimentos no se interrumpan.

### 1a. `omp_pt.cpp`

Copiar `serial_pt.cpp` → `omp_pt.cpp` y agregar sobre el loop externo de píxeles:

```cpp
#pragma omp parallel for schedule(???) num_threads(p)
for (int i = 0; i < height; i++) {
    // ...loop interno de columnas y muestras...
}
```

> El RNG (`rand()`) **no es thread-safe**. Usar `rand_r(&seed)` con semilla por hilo:  
> `unsigned int seed = omp_get_thread_num();`

Compilar con: `make omp_pt`

### 1b. `numba_pt.py`

Estructura base:

```python
from numba import njit, prange
import numpy as np

@njit(parallel=True)
def render(scene, width, height, depth, samples):
    img = np.zeros((height, width, 3), dtype=np.float64)
    for i in prange(height):          # loop paralelo
        seed = np.uint64(i * 1234567) # semilla independiente por hilo
        for j in range(width):
            # trace_path(...)
    return img
```

> **Importante:** Numba requiere semillas independientes por hilo. No usar `np.random` global;  
> implementar un LCG inline o usar `numba.typed` structures para el estado del RNG.

---

## Fase 2 — Tabla de configuraciones de experimento

Todas las corridas usan **ambas escenas** (`scene.txt` = 4 esferas, `scene_many.txt` = 40 esferas).

| Config | Resolución   | S (supersampling) | N (samples/px) | Uso                        |
|--------|--------------|-------------------|----------------|----------------------------|
| small  | 400 × 300    | S=2               | N=32           | Warmup / verificación      |
| medium | 800 × 600    | S=4               | N=128          | **Principal** (scheduling) |
| large  | 1600 × 1200  | S=4               | N=256          | Escalabilidad final        |

---

## Fase 3 — Experimentos (ordenados para minimizar tiempo de cómputo)

> **Estrategia:** todos los experimentos se lanzan desde Python con `subprocess`.  
> El flujo es: correr binario → capturar tiempo → guardar resultado → seguir.  
> Al final del script los datos ya están listos para graficar.

Crear la carpeta de resultados y un archivo base `experimentos.py` en `src/`:

```
src/
  experimentos.py   ← script maestro que corre TODO
  graficos.py       ← genera las figuras del informe
```

---

### Bloque base — helpers compartidos por todos los EXPs

```python
# experimentos.py
import subprocess, os, time, json, re
from pathlib import Path

SRC = Path(__file__).parent          # directorio src/
OUT = SRC / "resultados"
OUT.mkdir(exist_ok=True)

SCENES = ["scene.txt", "scene_many.txt"]
THREADS = [1, 2, 4, 8]
DEPTH = 8

# Las 3 configuraciones: (W, H, S_supersampling, N_samples)
CONFIGS = [
    (400,  300,  2,  32,  "small"),
    (800,  600,  4, 128,  "medium"),
    (1600, 1200, 4, 256,  "large"),
]

def run(cmd, env=None):
    """Ejecuta cmd, devuelve (stdout, elapsed_seconds)."""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, env=full_env)
    elapsed = time.perf_counter() - t0
    print(f"  [{elapsed:.1f}s] {' '.join(str(c) for c in cmd)}")
    return result.stdout + result.stderr, elapsed

def save(name, data):
    path = OUT / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → guardado en {path}")
```

---

### EXP-1 · Baseline serial (parte a) — ~15–60 min total

**Qué medir:** tiempo Ts para `./serial` (ray tracing) y `./serial_pt` (path tracing) en las 3 configs × 2 escenas.

```python
# --- EXP-1: baseline serial ---
def exp1_serial():
    results = {}   # results[scene][cfg_name][programa] = segundos

    for scene in SCENES:
        results[scene] = {}
        for W, H, S, N, cfg_name in CONFIGS:
            results[scene][cfg_name] = {}
            print(f"\n[EXP-1] {scene} | {W}x{H}")

            # Ray tracer serial
            _, t = run([SRC/"serial", scene, "out.ppm", W, H, DEPTH, S])
            results[scene][cfg_name]["serial_rt"] = t

            # Path tracer serial
            _, t = run([SRC/"serial_pt", scene, "out_pt.ppm", W, H, DEPTH, N])
            results[scene][cfg_name]["serial_pt"] = t

    save("exp1_serial", results)
    return results

ts = exp1_serial()
```

**Preguntas a responder después:**
- ¿Cuánto crece el tiempo al duplicar resolución? (esperado: ×4, porque W×H se cuadruplica)
- ¿Cuánto al duplicar N? (esperado: ×2, lineal en samples)

---

### EXP-2 · OpenMP — estrategias de scheduling (parte b) — ~2–4 horas

Usar **config medium (800×600, N=128)**, ambas escenas, p ∈ {1, 2, 4, 8}.

> `omp_pt_sched` debe aceptar `schedule` y `chunksize` como argumentos de línea de comandos,  
> o compilar variantes separadas. Ajustar el `run()` según la interfaz que implementes.

```python
# --- EXP-2: scheduling strategies ---
def exp2_scheduling():
    W, H, S, N, _ = CONFIGS[1]   # config medium
    results = {}

    strategies = [
        ("static",  [1, 4, 16, 64]),   # (nombre, chunksizes a probar)
        ("dynamic", [1, 4, 16, 64]),
        ("guided",  [None]),            # guided no necesita chunksize
    ]

    for scene in SCENES:
        results[scene] = {}
        for sched, chunks in strategies:
            results[scene][sched] = {}
            for chunk in chunks:
                results[scene][sched][str(chunk)] = {}
                for p in THREADS:
                    print(f"\n[EXP-2] {scene} | {sched} chunk={chunk} p={p}")
                    cmd = [SRC/"omp_pt_sched", scene, "out.ppm", W, H, DEPTH, N, sched]
                    if chunk is not None:
                        cmd.append(chunk)
                    _, t = run(cmd, env={"OMP_NUM_THREADS": str(p)})
                    results[scene][sched][str(chunk)][str(p)] = t

    save("exp2_scheduling", results)
    return results


```

**Preguntas a responder:**
- ¿Qué estrategia es más eficiente?
- ¿Cambia la estrategia óptima entre `scene.txt` (4 esferas) y `scene_many.txt` (40 esferas)?
- ¿Por qué el trabajo por píxel es irregular? (pista: profundidad de rebotes varía por píxel)

---

### EXP-3 · OpenMP — escalabilidad completa (partes d/e) — ~2–3 horas

Con la **mejor estrategia** encontrada en EXP-2, correr las 3 configs × 2 escenas × p ∈ {1,2,4,8}.

```python
# --- EXP-3: escalabilidad OpenMP ---
BEST_SCHED = "guided"   # ← actualizar según resultado de EXP-2
BEST_CHUNK = None       # ← idem (None si guided, número si static/dynamic)

def exp3_omp():
    results = {}

    for scene in SCENES:
        results[scene] = {}
        for W, H, S, N, cfg_name in CONFIGS:
            results[scene][cfg_name] = {}
            for p in THREADS:
                print(f"\n[EXP-3] {scene} | {W}x{H} N={N} p={p}")
                cmd = [SRC/"omp_pt", scene, "out_omp.ppm", W, H, DEPTH, N]
                if BEST_CHUNK is not None:
                    cmd.append(BEST_CHUNK)
                _, t = run(cmd, env={"OMP_NUM_THREADS": str(p)})
                results[scene][cfg_name][str(p)] = t

    save("exp3_omp", results)
    return results

omp_results = exp3_omp()
```

---

### EXP-4 · Numba — escalabilidad completa (partes c/d/e) — ~1–2 horas

> **Primera corrida de Numba siempre es lenta** (~30s de compilación JIT).  
> El script hace un warmup con config small antes de medir las configs reales.

```python
# --- EXP-4: escalabilidad Numba ---
def exp4_numba():
    results = {}

    # Warmup: forzar compilación JIT antes de medir
    print("\n[EXP-4] Warmup Numba (compilación JIT)...")
    W0, H0, S0, N0, _ = CONFIGS[0]
    run(["python", SRC/"numba_pt.py", "scene.txt", "warmup.png", W0, H0, DEPTH, N0],
        env={"NUMBA_NUM_THREADS": "1"})

    for scene in SCENES:
        results[scene] = {}
        for W, H, S, N, cfg_name in CONFIGS:
            results[scene][cfg_name] = {}
            for p in THREADS:
                print(f"\n[EXP-4] {scene} | {W}x{H} N={N} p={p}")
                _, t = run(
                    ["python", SRC/"numba_pt.py", scene, "out_numba.png", W, H, DEPTH, N],
                    env={"NUMBA_NUM_THREADS": str(p)}
                )
                results[scene][cfg_name][str(p)] = t

    save("exp4_numba", results)
    return results

numba_results = exp4_numba()
```

---

### EXP-5 · Segundo computador (parte f)

Correr el mismo `experimentos.py` en la segunda máquina. Antes de ejecutarlo, registrar el hardware:

```python
# al inicio de experimentos.py, agregar:
import platform, subprocess as sp

def hw_info():
    info = {
        "hostname": platform.node(),
        "cpu": platform.processor(),
        "cores_logicos": os.cpu_count(),
    }
    try:
        lscpu = sp.check_output("lscpu", text=True)
        for line in lscpu.splitlines():
            if "Core(s) per socket" in line:
                info["cores_fisicos"] = line.split(":")[1].strip()
            if "CPU MHz" in line or "CPU max MHz" in line:
                info["mhz"] = line.split(":")[1].strip()
    except Exception:
        pass
    save("hw_info", info)
    print(json.dumps(info, indent=2))

hw_info()
```

Copiar `resultados/` de la segunda máquina a `resultados/maquina2/` para comparar.

---

## Fase 4 — Análisis y gráficos (`graficos.py`)

Lee los JSON generados por `experimentos.py` y produce las figuras del informe.  
Para cada configuración generar una figura con 3 subplots: T(p), S(p), E(p).

```python
# graficos.py
import json, matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("resultados")

def load(name):
    with open(OUT / f"{name}.json") as f:
        return json.load(f)

serial  = load("exp1_serial")
omp     = load("exp3_omp")
numba   = load("exp4_numba")

SCENES   = ["scene.txt", "scene_many.txt"]
CONFIGS  = [("small", "400×300 N=32"), ("medium", "800×600 N=128"), ("large", "1600×1200 N=256")]
THREADS  = [1, 2, 4, 8]

for scene in SCENES:
    scene_tag = "4esf" if scene == "scene.txt" else "40esf"
    for cfg_key, cfg_label in CONFIGS:

        Ts  = serial[scene][cfg_key]["serial_pt"]
        T_omp   = [omp[scene][cfg_key][str(p)]   for p in THREADS]
        T_numba = [numba[scene][cfg_key][str(p)]  for p in THREADS]

        S_omp   = [Ts / T for T in T_omp]
        S_numba = [Ts / T for T in T_numba]
        E_omp   = [s / p  for s, p in zip(S_omp,   THREADS)]
        E_numba = [s / p  for s, p in zip(S_numba, THREADS)]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"{cfg_label} — {scene_tag}", fontsize=13)

        # T(p)
        axes[0].plot(THREADS, T_omp,   'o-', label='OpenMP')
        axes[0].plot(THREADS, T_numba, 's-', label='Numba')
        axes[0].axhline(Ts, color='gray', linestyle='--', label='Serial')
        axes[0].plot(THREADS, [Ts/p for p in THREADS], 'k--', label='Ideal')
        axes[0].set(xlabel='Threads p', ylabel='Tiempo (s)', title='T(p)')
        axes[0].legend(fontsize=9); axes[0].grid(True)

        # S(p)
        axes[1].plot(THREADS, S_omp,   'o-', label='OpenMP')
        axes[1].plot(THREADS, S_numba, 's-', label='Numba')
        axes[1].plot(THREADS, THREADS, 'k--', label='Ideal')
        axes[1].set(xlabel='p', ylabel='Speedup S(p)', title='Speedup S(p) = Ts/T(p)')
        axes[1].legend(fontsize=9); axes[1].grid(True)

        # E(p)
        axes[2].plot(THREADS, E_omp,   'o-', label='OpenMP')
        axes[2].plot(THREADS, E_numba, 's-', label='Numba')
        axes[2].axhline(1.0, color='k', linestyle='--', label='Ideal')
        axes[2].set(xlabel='p', ylabel='Eficiencia E(p)', title='Eficiencia E(p) = S(p)/p')
        axes[2].legend(fontsize=9); axes[2].grid(True)

        plt.tight_layout()
        fname = OUT / f"fig_{scene_tag}_{cfg_key}.pdf"
        plt.savefig(fname, dpi=150)
        print(f"Guardado: {fname}")
        plt.close()
```

---

## Fase 5 — Informe PDF

Secciones del informe:

1. **Descripción del entorno** — hardware de ambas máquinas (cores, reloj, OS, compilador)
2. **Parte a** — tabla Ts + análisis de escala con resolución y N
3. **Parte b** — tabla de tiempos por estrategia de scheduling, análisis de irregularidad
4. **Parte c** — explicación de Numba JIT vs OpenMP
5. **Partes d/e** — gráficos T(p), S(p), E(p) por config; análisis de qué escala mejor
6. **Parte f** — comparación entre máquinas
7. **Declaración de uso de IA** (obligatorio si se usa)

> **Checklist de gráficos:** título, labels en ejes, leyenda, tamaño de fuente ≥ 10pt.

---

## Orden de ejecución recomendado (para minimizar bloqueos)

```
Día 1 AM  → Setup entorno ambas máquinas + compilar seriales
Día 1 PM  → Implementar omp_pt.cpp + numba_pt.py
Día 2 AM  → Lanzar EXP-1 (serial baseline, rápido) en ambas máquinas
Día 2 PM  → Lanzar EXP-2 (scheduling, lento) — puede dejarse corriendo de noche
Día 3 AM  → Lanzar EXP-3 + EXP-4 con mejor estrategia + EXP-5 en máquina 2
Día 4     → Gráficos + análisis + redacción del informe
Día 5     → Revisión final + entrega Canvas
```

---

## Flujo completo de ejecución

```bash
cd src/

# 1. Compilar todo en WSL
make all

# 2. Correr todos los experimentos en Anaconda (puede tardar varias horas)
conda activate tarea2-hpc
python exp1_baseline.py
python exp2_scheduling.py
python exp3_omp_scaling.py
python exp4_numba_scaling.py

# 3. Generar gráficos
python graficos.py

# Los resultados quedan en src/resultados/
#   exp1_serial.json, exp2_scheduling.json,
#   exp3_omp.json, exp4_numba.json
#   fig_4esf_small.pdf, fig_4esf_medium.pdf, ...
```

Para correr un solo experimento durante desarrollo, ejecutar el script correspondiente:

```bash
python exp1_baseline.py       # baseline serial  (~15-60 min)
python exp2_scheduling.py     # scheduling OMP   (~2-4 horas)
python exp3_omp_scaling.py    # escalabilidad OMP (~2-3 horas)
python exp4_numba_scaling.py  # escalabilidad Numba (~1-2 horas)
```

---

## Comandos de testing manual

Todos los comandos se ejecutan desde `src/`. Activar el entorno conda antes de correr Numba:

```bash
conda activate tarea2-hpc
cd src/
```

### Compilar todo

```bash
make all
# Genera: ./serial  ./serial_pt  ./omp_pt  ./omp_pt_sched
```

---

### Serial

```bash
# Ray tracer serial
./serial scene.txt output_rt.ppm 400 300 8 2

# Path tracer serial (config small)
./serial_pt scene.txt output_pt.ppm 400 300 8 32

# Path tracer serial (config medium)
./serial_pt scene.txt output_pt.ppm 800 600 8 128

# Con scene_many (40 esferas)
./serial_pt scene_many.txt output_pt_many.ppm 400 300 8 32
```

Formato de argumentos: `<scene> <output.ppm> <W> <H> <D> <N>`
- `D` = profundidad máxima de rebotes
- `N` = muestras por pixel (path tracing) / S de supersampling (ray tracing)

---

### OpenMP (`omp_pt`)

```bash
# 4 threads, config small
./omp_pt scene.txt output_omp.ppm 400 300 8 32 4

# 8 threads, config medium
./omp_pt scene.txt output_omp.ppm 800 600 8 128 8

# Alternativa: controlar threads por variable de entorno (sin pasar P)
OMP_NUM_THREADS=4 ./omp_pt scene.txt output_omp.ppm 400 300 8 32
```

Formato: `<scene> <output.ppm> <W> <H> <D> <N> [P]`
- `P` = número de threads (opcional, default: `OMP_NUM_THREADS` o todos los disponibles)

---

### OpenMP con scheduling (`omp_pt_sched`)

```bash
# static chunk=1, 4 threads
./omp_pt_sched scene.txt output_sched.ppm 800 600 8 128 4 static 1

# dynamic chunk=16, 4 threads
./omp_pt_sched scene.txt output_sched.ppm 800 600 8 128 4 dynamic 16

# guided (sin chunksize), 8 threads
./omp_pt_sched scene.txt output_sched.ppm 800 600 8 128 8 guided

# Con scene_many para comparar
./omp_pt_sched scene_many.txt output_sched.ppm 800 600 8 128 4 dynamic 4
```

Formato: `<scene> <output.ppm> <W> <H> <D> <N> <P> <schedule> [chunksize]`
- `schedule` = `static` | `dynamic` | `guided`
- `chunksize` = opcional (si se omite, el runtime usa su default)

---

### Numba (`numba_pt.py`)

```bash
# 1 thread, config small (primer run incluye compilación JIT ~30s)
NUMBA_NUM_THREADS=1 python numba_pt.py scene.txt output_numba.png 400 300 8 32

# 4 threads, config medium
NUMBA_NUM_THREADS=4 python numba_pt.py scene.txt output_numba.png 800 600 8 128

# 8 threads, config large
NUMBA_NUM_THREADS=8 python numba_pt.py scene.txt output_numba.png 1600 1200 8 256

# Con scene_many
NUMBA_NUM_THREADS=4 python numba_pt.py scene_many.txt output_numba.png 400 300 8 32
```

Formato: `<scene> <output.png> <W> <H> <D> <N>`
- Threads se controla **solo** con `NUMBA_NUM_THREADS` (no es argumento posicional)
- La primera ejecución siempre es lenta (~30s de JIT); el tiempo reportado excluye eso
- La salida es `.png` (no `.ppm` como los binarios C++)

> **Tip:** Para ver la imagen generada en cualquier formato:
> ```bash
> # Ver PPM (instalar imagemagick si no está)
> display output_pt.ppm
> # o convertir a PNG
> convert output_pt.ppm output_pt.png
> ```

---

### Verificar que las imágenes son razonablemente similares

Una sanidad check rápida: correr serial y omp_pt con los mismos parámetros y comparar visualmente los PPM. No serán idénticos (RNG distinto por semilla), pero deben verse similares.

```bash
./serial_pt scene.txt ref_serial.ppm 400 300 8 32
./omp_pt    scene.txt ref_omp.ppm    400 300 8 32 4
NUMBA_NUM_THREADS=4 python numba_pt.py scene.txt ref_numba.png 400 300 8 32
```
