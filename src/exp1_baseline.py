"""
EXP-1 — Baseline serial (parte a)
==================================
Corre ./serial (ray tracing) y ./serial_pt (path tracing) en las 3 configuraciones
y las 2 escenas. Guarda los tiempos en resultados/exp1_baseline.json.

Tiempo estimado: 15–60 min según la máquina.

Uso:
    python exp1_baseline.py
"""

from common import *

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    results = {}   # results[scene][cfg_name][programa] = segundos

    for scene in SCENES:
        results[scene] = {}
        print(f"\n{'='*60}")
        print(f"  Escena: {scene}")
        print(f"{'='*60}")

        for W, H, S, N, cfg_name in CONFIGS:
            results[scene][cfg_name] = {}
            print(f"\n  Config {cfg_name}: {W}x{H}  S={S}  N={N}")

            # ── Ray tracer serial ──────────────────────────────────────────
            print("    serial (ray tracing)...")
            out, wall = run([bin("serial"), scene, OUT/"out_serial.ppm", W, H, DEPTH, S])
            t = parse_time(out) or wall
            results[scene][cfg_name]["serial_rt"] = t
            print(f"    Tiempo medido: {t:.3f}s")

            # ── Path tracer serial ─────────────────────────────────────────
            print("    serial_pt (path tracing)...")
            out, wall = run([bin("serial_pt"), scene, OUT/"out_serial_pt.ppm", W, H, DEPTH, N])
            t = parse_time(out) or wall
            results[scene][cfg_name]["serial_pt"] = t
            print(f"    Tiempo medido: {t:.3f}s")

    save("exp1_baseline", results)

    # ── Resumen en pantalla ───────────────────────────────────────────────────
    print("\n\n" + "="*60)
    print("  RESUMEN — Tiempos serial_pt (s)")
    print("="*60)
    header = f"{'Config':<10}" + "".join(f"{s:<20}" for s in SCENES)
    print(header)
    for _, _, _, _, cfg_name in CONFIGS:
        row = f"  {cfg_name:<8}"
        for scene in SCENES:
            t = results[scene][cfg_name]["serial_pt"]
            row += f"  {t:>8.2f}s          "
        print(row)

    print("\n¿Cuánto crece al duplicar resolución?")
    for scene in SCENES:
        t_s = results[scene]["small"]["serial_pt"]
        t_m = results[scene]["medium"]["serial_pt"]
        t_l = results[scene]["large"]["serial_pt"]
        print(f"  {scene}: small→medium ×{t_m/t_s:.2f}  medium→large ×{t_l/t_m:.2f}  (esperado ≈×4)")

    print("\n¿Cuánto crece al duplicar N?")
    print("  (comparar small N=32 vs medium N=128: ×4 en N, ×4 en res → diferencia debería venir de N)")

if __name__ == "__main__":
    main()
