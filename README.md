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
make clean && make
```

> **Verificar** que `./serial` y `./serial_pt` corren sin errores antes de continuar.

```bash
./serial_pt scene.txt test.ppm 400 300 8 32
# Debe imprimir: Tiempo : X.XX s
```

Para poder visualizar archivos ppm, usamos ImageMagick para convertir a una imagen png.

```bash
# Instalar ImageMagick
sudo apt install imagemagick

# Convertir en imagen
convert test.ppm test.png
```

Código para verificar funcionamiento de código e integridad de las imagenes generadas.

```bash
# Correr una instancia de cada algoritmo
./serial scene.txt out_serial.ppm 800 600 8 4
./serial_pt scene.txt out_serial_pt.ppm 800 600 8 64
./omp_pt scene.txt out_omp_pt.ppm 800 600 8 64 4
./omp_pt_sched scene.txt out_omp_sched.ppm 800 600 8 64 4
python3 numba_pt.py scene.txt out_numba_pt.ppm 800 600 8 64 

# Convertir en imagen
convert out_serial.ppm out_serial.png 
convert out_serial_pt.ppm out_serial_pt.png
convert out_omp_pt.ppm out_omp_pt.png
convert out_omp_sched.ppm out_omp_sched.png 
convert out_numba_pt.ppm out_numba_pt.png
```

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

```
src/
  experimentos.py   ← script maestro que corre TODO
  graficos.py       ← genera las figuras del informe
```

---

### EXP-1 · Baseline serial (parte a) — ~15–60 min total

**Qué medir:** tiempo Ts para `./serial` (ray tracing) y `./serial_pt` (path tracing) en las 3 configs × 2 escenas.

**Preguntas a responder después:**
- ¿Cuánto crece el tiempo al duplicar resolución? (esperado: ×4, porque W×H se cuadruplica)
- ¿Cuánto al duplicar N? (esperado: ×2, lineal en samples)

---

### EXP-2 · OpenMP — estrategias de scheduling (parte b) — ~2–4 horas

Usar **config medium (800×600, N=128)**, ambas escenas, p ∈ {1, 2, 4, 8}.

Estrategias a probar: `static` [chunks: 1, 4, 16, 64], `dynamic` [chunks: 1, 4, 16, 64], `guided`.

**Preguntas a responder:**
- ¿Qué estrategia es más eficiente?
- ¿Cambia la estrategia óptima entre `scene.txt` (4 esferas) y `scene_many.txt` (40 esferas)?
- ¿Por qué el trabajo por píxel es irregular? (pista: profundidad de rebotes varía por píxel)

---

### EXP-3 · OpenMP — escalabilidad completa (partes d/e) — ~2–3 horas

Con la **mejor estrategia** encontrada en EXP-2, correr las 3 configs × 2 escenas × p ∈ {1,2,4,8}.

```python
BEST_SCHED = "guided"   # ← actualizar según resultado de EXP-2
BEST_CHUNK = None       # ← idem (None si guided, número si static/dynamic)
```

---

### EXP-4 · Numba — escalabilidad completa (partes c/d/e) — ~1–2 horas

> **Primera corrida de Numba siempre es lenta** (~30s de compilación JIT).  
> El script hace un warmup con config small antes de medir las configs reales.

---

### EXP-5 · Segundo computador (parte f)

Correr el mismo `experimentos.py` en la segunda máquina. Copiar `resultados/` de la segunda máquina a `resultados/maquina2/` para comparar.

---

## Fase 4 — Análisis y gráficos (`graficos.py`)

Lee los JSON generados por los scripts de experimentos y produce las figuras del informe.  
Para cada configuración genera una figura con 3 subplots: T(p), S(p), E(p).

Resultados guardados en `src/resultados/`:
```
exp1_serial.json
exp2_scheduling.json
exp3_omp.json
exp4_numba.json
fig_4esf_small.pdf
fig_4esf_medium.pdf
...
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
Día 1 PM  → Verificar que omp_pt.cpp y numba_pt.py funcionan correctamente
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
make clean && make

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

---

## Comandos de testing manual

Todos los comandos se ejecutan desde `src/`. Activar el entorno conda antes de correr Numba:

```bash
conda activate tarea2-hpc
cd src/
```

### Compilar todo

```bash
make clean && make
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

---

### Ver imágenes

```bash
# Convertir PPM a PNG y abrir en Windows
convert out_serial.ppm out_serial.png && explorer.exe out_serial.png
convert out_serial_pt.ppm out_serial_pt.png && explorer.exe out_serial_pt.png
convert out_omp_pt.ppm out_omp_pt.png && explorer.exe out_omp_pt.png
convert out_omp_sched.ppm out_omp_sched.png && explorer.exe out_omp_sched.png
convert out_numba_pt.ppm out_numba_pt.png && explorer.exe out_numba_pt.png

# Comparar todas de un vistazo
convert \
  \( out_serial.png -gravity North -background white -splice 0x20 -annotate +0+2 "serial" \) \
  \( out_serial_pt.png -gravity North -background white -splice 0x20 -annotate +0+2 "serial_pt" \) \
  \( out_omp_pt.png -gravity North -background white -splice 0x20 -annotate +0+2 "omp_pt" \) \
  \( out_omp_sched.png -gravity North -background white -splice 0x20 -annotate +0+2 "omp_pt_sched" \) \
  +append comparison.png && explorer.exe comparison.png
```

### Verificar que las imágenes son razonablemente similares

```bash
./serial_pt scene.txt ref_serial.ppm 400 300 8 32
./omp_pt    scene.txt ref_omp.ppm    400 300 8 32 4
NUMBA_NUM_THREADS=4 python numba_pt.py scene.txt ref_numba.png 400 300 8 32
```
