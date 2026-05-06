#!/usr/bin/env python3
"""
numba_pt.py - Path tracer paralelo con Numba.

Uso: python numba_pt.py <scene.txt> <output.png> [W H D N]
  W, H : resolucion (default 800x600)
  D    : profundidad maxima de rebotes (default 8)
  N    : muestras por pixel (default 64)

El numero de threads se controla con la variable de entorno NUMBA_NUM_THREADS.
El tiempo reportado excluye la compilacion JIT de Numba.
"""

import sys, time, math
import numpy as np
from numba import njit, prange

# =============================================================================
# Algebra lineal 3D
# =============================================================================

@njit(inline='always')
def dot(ax, ay, az, bx, by, bz):
    return ax*bx + ay*by + az*bz

@njit(inline='always')
def length2(ax, ay, az):
    return ax*ax + ay*ay + az*az

@njit(inline='always')
def vlength(ax, ay, az):
    return np.sqrt(ax*ax + ay*ay + az*az)

# =============================================================================
# Intersecciones
# =============================================================================

@njit(inline='always')
def intersect_sphere(rox, roy, roz, rdx, rdy, rdz, cx, cy, cz, radius):
    """Retorna t > 0 o -1.0 si no hay interseccion."""
    ox = rox-cx; oy = roy-cy; oz = roz-cz
    a  = rdx*rdx + rdy*rdy + rdz*rdz
    hb = ox*rdx + oy*rdy + oz*rdz
    c  = ox*ox + oy*oy + oz*oz - radius*radius
    disc = hb*hb - a*c
    if disc < 0.0:
        return -1.0
    sq = np.sqrt(disc)
    t  = (-hb - sq) / a
    if t < 1e-4:
        t = (-hb + sq) / a
    if t < 1e-4:
        return -1.0
    return t


@njit(inline='always')
def intersect_plane(rox, roy, roz, rdx, rdy, rdz, nx, ny, nz, d):
    """Retorna t > 0 o -1.0 si no hay interseccion."""
    denom = nx*rdx + ny*rdy + nz*rdz
    if abs(denom) < 1e-8:
        return -1.0
    t = (d - (nx*rox + ny*roy + nz*roz)) / denom
    if t < 1e-4:
        return -1.0
    return t

# =============================================================================
# closest_hit
# Retorna: (t, hit, px,py,pz, nx,ny,nz, ar,ag,ab, reflectivity, shininess)
# Si no hay hit: t=inf, hit=False.
# =============================================================================

@njit
def closest_hit(rox, roy, roz, rdx, rdy, rdz,
                sph_cx, sph_cy, sph_cz, sph_r,
                sph_ar, sph_ag, sph_ab, sph_refl, sph_shin,
                pln_nx, pln_ny, pln_nz, pln_d,
                pln_ar, pln_ag, pln_ab, pln_refl, pln_shin):

    best_t    = np.inf
    best_idx  = -1
    best_type = 0  # 1=sphere, 2=plane

    for i in range(sph_cx.shape[0]):
        t = intersect_sphere(rox,roy,roz, rdx,rdy,rdz,
                             sph_cx[i], sph_cy[i], sph_cz[i], sph_r[i])
        if t > 0.0 and t < best_t:
            best_t = t; best_idx = i; best_type = 1

    for i in range(pln_nx.shape[0]):
        t = intersect_plane(rox,roy,roz, rdx,rdy,rdz,
                            pln_nx[i], pln_ny[i], pln_nz[i], pln_d[i])
        if t > 0.0 and t < best_t:
            best_t = t; best_idx = i; best_type = 2

    if best_type == 0:
        return np.inf, False, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0, 0.0

    px = rox + rdx*best_t
    py = roy + rdy*best_t
    pz = roz + rdz*best_t

    if best_type == 1:
        i   = best_idx
        nl  = vlength(px-sph_cx[i], py-sph_cy[i], pz-sph_cz[i])
        nnx = (px-sph_cx[i])/nl
        nny = (py-sph_cy[i])/nl
        nnz = (pz-sph_cz[i])/nl
        ar = sph_ar[i]; ag = sph_ag[i]; ab = sph_ab[i]
        refl = sph_refl[i]; shin = sph_shin[i]
    else:
        i   = best_idx
        nnx = pln_nx[i]; nny = pln_ny[i]; nnz = pln_nz[i]
        if nnx*rdx + nny*rdy + nnz*rdz > 0.0:
            nnx=-nnx; nny=-nny; nnz=-nnz
        ar = pln_ar[i]; ag = pln_ag[i]; ab = pln_ab[i]
        refl = pln_refl[i]; shin = pln_shin[i]

    return best_t, True, px,py,pz, nnx,nny,nnz, ar,ag,ab, refl, shin

# =============================================================================
# Direccion aleatoria en hemisferio (metodo de rechazo, igual que C++)
# Usa np.random.uniform() — estado interno de Numba, thread-safe en prange.
# =============================================================================

@njit
def random_hemisphere(nnx, nny, nnz):
    """Retorna (dx, dy, dz) en el hemisferio orientado por la normal."""
    for _ in range(256):
        px = np.random.uniform(-1.0, 1.0)
        py = np.random.uniform(-1.0, 1.0)
        pz = np.random.uniform(-1.0, 1.0)
        l2 = px*px + py*py + pz*pz
        if 1e-10 < l2 <= 1.0:
            il  = 1.0/np.sqrt(l2)
            px *= il; py *= il; pz *= il
            dx = nnx+px; dy = nny+py; dz = nnz+pz
            ld2 = dx*dx + dy*dy + dz*dz
            if ld2 > 1e-10:
                il2 = 1.0/np.sqrt(ld2)
                return dx*il2, dy*il2, dz*il2
    return nnx, nny, nnz

# =============================================================================
# trace_path — logica identica a pt.hpp
# =============================================================================

@njit
def trace_path(rox, roy, roz, rdx, rdy, rdz, max_depth,
               sph_cx, sph_cy, sph_cz, sph_r,
               sph_ar, sph_ag, sph_ab, sph_refl, sph_shin,
               pln_nx, pln_ny, pln_nz, pln_d,
               pln_ar, pln_ag, pln_ab, pln_refl, pln_shin,
               lights_x, lights_y, lights_z, lights_int,
               amb_r, amb_g, amb_b):

    cr = 0.0; cg = 0.0; cb = 0.0   # color acumulado
    tr = 1.0; tg = 1.0; tb = 1.0   # throughput

    for depth in range(max_depth):
        ht, hit, px,py,pz, nx,ny,nz, ar,ag,ab, refl, shin = closest_hit(
            rox,roy,roz, rdx,rdy,rdz,
            sph_cx, sph_cy, sph_cz, sph_r,
            sph_ar, sph_ag, sph_ab, sph_refl, sph_shin,
            pln_nx, pln_ny, pln_nz, pln_d,
            pln_ar, pln_ag, pln_ab, pln_refl, pln_shin)

        if not hit:
            # Fondo: gradiente celeste como fuente de luz ambiente
            t_sky = 0.5*(rdy+1.0)
            cr += tr * ((1.0-t_sky)*1.0 + t_sky*0.5) * 0.6
            cg += tg * ((1.0-t_sky)*1.0 + t_sky*0.7) * 0.6
            cb += tb * ((1.0-t_sky)*1.0 + t_sky*1.0) * 0.6
            break

        # Iluminacion directa
        dr = amb_r*ar; dg = amb_g*ag; db = amb_b*ab

        for li in range(lights_x.shape[0]):
            lvx = lights_x[li]-px; lvy = lights_y[li]-py; lvz = lights_z[li]-pz
            ldist = np.sqrt(lvx*lvx + lvy*lvy + lvz*lvz)
            ldx = lvx/ldist; ldy = lvy/ldist; ldz = lvz/ldist

            # Rayo de sombra
            sh_t, sh_hit, _,_,_, _,_,_, _,_,_, _, _ = closest_hit(
                px+nx*1e-4, py+ny*1e-4, pz+nz*1e-4, ldx, ldy, ldz,
                sph_cx, sph_cy, sph_cz, sph_r,
                sph_ar, sph_ag, sph_ab, sph_refl, sph_shin,
                pln_nx, pln_ny, pln_nz, pln_d,
                pln_ar, pln_ag, pln_ab, pln_refl, pln_shin)
            if sh_hit and sh_t < ldist:
                continue  # en sombra

            lint = lights_int[li]
            ndl  = nx*ldx + ny*ldy + nz*ldz
            diff = max(0.0, ndl) * lint
            dr += ar*diff; dg += ag*diff; db += ab*diff

            # Especular
            rlx = 2.0*ndl*nx - ldx
            rly = 2.0*ndl*ny - ldy
            rlz = 2.0*ndl*nz - ldz
            spec_cos = -(rlx*rdx + rly*rdy + rlz*rdz)
            if spec_cos > 0.0:
                spec = (spec_cos**shin) * lint * 0.5
                dr += spec; dg += spec; db += spec

        f = 1.0 - refl
        cr += tr*f*dr; cg += tg*f*dg; cb += tb*f*db

        # Siguiente rebote
        if refl > 1e-4:
            rdotn = rdx*nx + rdy*ny + rdz*nz
            rdx2 = rdx - 2.0*rdotn*nx
            rdy2 = rdy - 2.0*rdotn*ny
            rdz2 = rdz - 2.0*rdotn*nz
            rl   = 1.0/np.sqrt(rdx2*rdx2 + rdy2*rdy2 + rdz2*rdz2)
            rdx = rdx2*rl; rdy = rdy2*rl; rdz = rdz2*rl
            rox = px+nx*1e-4; roy = py+ny*1e-4; roz = pz+nz*1e-4
            tr *= refl; tg *= refl; tb *= refl
        else:
            rdx, rdy, rdz = random_hemisphere(nx, ny, nz)
            rox = px+nx*1e-4; roy = py+ny*1e-4; roz = pz+nz*1e-4
            tr *= ar; tg *= ag; tb *= ab

        # Terminacion anticipada
        if max(tr, max(tg, tb)) < 0.01:
            break

    return min(1.0,max(0.0,cr)), min(1.0,max(0.0,cg)), min(1.0,max(0.0,cb))

# =============================================================================
# render — loop principal paralelo con prange
# np.random.seed() dentro de @njit siembra el estado interno de Numba,
# que es independiente por thread -> sin condiciones de carrera.
# =============================================================================

@njit(parallel=True)
def render(W, H, max_depth, samples,
           cam_ox, cam_oy, cam_oz,
           cam_llx, cam_lly, cam_llz,
           cam_hx,  cam_hy,  cam_hz,
           cam_vx,  cam_vy,  cam_vz,
           sph_cx, sph_cy, sph_cz, sph_r,
           sph_ar, sph_ag, sph_ab, sph_refl, sph_shin,
           pln_nx, pln_ny, pln_nz, pln_d,
           pln_ar, pln_ag, pln_ab, pln_refl, pln_shin,
           lights_x, lights_y, lights_z, lights_int,
           amb_r, amb_g, amb_b):

    img = np.zeros((H, W, 3), dtype=np.float64)

    for j in prange(H):
        # Semilla independiente por fila: siembra el estado interno de Numba
        # para el thread que procesa esta fila.
        np.random.seed(j * 1234567 + 1)

        for i in range(W):
            cr = 0.0; cg = 0.0; cb = 0.0

            for s in range(samples):
                du = np.random.random()
                dv = np.random.random()

                u = (np.float64(i) + du) / np.float64(W)
                v = (np.float64(j) + dv) / np.float64(H)

                rdx = cam_llx + u*cam_hx + v*cam_vx - cam_ox
                rdy = cam_lly + u*cam_hy + v*cam_vy - cam_oy
                rdz = cam_llz + u*cam_hz + v*cam_vz - cam_oz
                rl  = 1.0/np.sqrt(rdx*rdx + rdy*rdy + rdz*rdz)
                rdx *= rl; rdy *= rl; rdz *= rl

                c_r, c_g, c_b = trace_path(
                    cam_ox, cam_oy, cam_oz, rdx, rdy, rdz, max_depth,
                    sph_cx, sph_cy, sph_cz, sph_r,
                    sph_ar, sph_ag, sph_ab, sph_refl, sph_shin,
                    pln_nx, pln_ny, pln_nz, pln_d,
                    pln_ar, pln_ag, pln_ab, pln_refl, pln_shin,
                    lights_x, lights_y, lights_z, lights_int,
                    amb_r, amb_g, amb_b)
                cr += c_r; cg += c_g; cb += c_b

            inv_n = 1.0 / np.float64(samples)
            img[j, i, 0] = cr * inv_n
            img[j, i, 1] = cg * inv_n
            img[j, i, 2] = cb * inv_n

    return img

# =============================================================================
# Carga de escena y camara (Python puro, fuera de Numba)
# =============================================================================

def load_scene(path):
    planes = [
        ([0,1,0],  -1.5, [0.75,0.75,0.75], 0.05, 8.0),
        ([0,1,0],   2.5, [0.90,0.90,0.90], 0.00, 1.0),
        ([0,0,1],  -4.0, [0.80,0.80,0.80], 0.00, 1.0),
        ([1,0,0],  -3.0, [0.75,0.15,0.15], 0.00, 1.0),
        ([-1,0,0], -3.0, [0.15,0.75,0.15], 0.00, 1.0),
    ]
    lights  = [([0.5, 2.2, 0.5], 1.00)]
    ambient = [0.02, 0.02, 0.02]

    spheres = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if parts[0] != 'sphere':
                continue
            cx,cy,cz = float(parts[1]),float(parts[2]),float(parts[3])
            r        = float(parts[4])
            red,g,b  = float(parts[5]),float(parts[6]),float(parts[7])
            refl     = float(parts[8])
            shin     = float(parts[9])
            spheres.append((cx,cy,cz,r,red,g,b,refl,shin))

    def farr(lst): return np.array(lst, dtype=np.float64)

    sph_cx   = farr([s[0] for s in spheres])
    sph_cy   = farr([s[1] for s in spheres])
    sph_cz   = farr([s[2] for s in spheres])
    sph_r    = farr([s[3] for s in spheres])
    sph_ar   = farr([s[4] for s in spheres])
    sph_ag   = farr([s[5] for s in spheres])
    sph_ab   = farr([s[6] for s in spheres])
    sph_refl = farr([s[7] for s in spheres])
    sph_shin = farr([s[8] for s in spheres])

    pln_nx   = farr([p[0][0] for p in planes])
    pln_ny   = farr([p[0][1] for p in planes])
    pln_nz   = farr([p[0][2] for p in planes])
    pln_d    = farr([p[1]    for p in planes])
    pln_ar   = farr([p[2][0] for p in planes])
    pln_ag   = farr([p[2][1] for p in planes])
    pln_ab   = farr([p[2][2] for p in planes])
    pln_refl = farr([p[3]    for p in planes])
    pln_shin = farr([p[4]    for p in planes])

    lights_x   = farr([l[0][0] for l in lights])
    lights_y   = farr([l[0][1] for l in lights])
    lights_z   = farr([l[0][2] for l in lights])
    lights_int = farr([l[1]    for l in lights])
    amb        = farr(ambient)

    return (sph_cx, sph_cy, sph_cz, sph_r,
            sph_ar, sph_ag, sph_ab, sph_refl, sph_shin,
            pln_nx, pln_ny, pln_nz, pln_d,
            pln_ar, pln_ag, pln_ab, pln_refl, pln_shin,
            lights_x, lights_y, lights_z, lights_int, amb)


def make_camera(W, H):
    eye    = np.array([0.0, 0.5, 3.8])
    lookat = np.array([0.0, 0.0, 0.0])
    up     = np.array([0.0, 1.0, 0.0])
    vfov   = 55.0
    theta  = vfov * math.pi / 180.0
    half_h = math.tan(theta / 2.0)
    half_w = half_h * W / H
    ww = eye - lookat;  ww /= np.linalg.norm(ww)
    uu = np.cross(up, ww); uu /= np.linalg.norm(uu)
    vv = np.cross(ww, uu)
    lower_left = eye - uu*half_w - vv*half_h - ww
    horiz      = uu * (2.0 * half_w)
    vert       = vv * (2.0 * half_h)
    return eye, lower_left, horiz, vert

# =============================================================================
# Main
# =============================================================================

def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print("Uso: python numba_pt.py <scene.txt> <output.png> [W H D N]",
              file=sys.stderr)
        sys.exit(1)

    scene_file  = args[0]
    output_file = args[1]
    W         = int(args[2]) if len(args) > 2 else 800
    H         = int(args[3]) if len(args) > 3 else 600
    MAX_DEPTH = int(args[4]) if len(args) > 4 else 8
    SAMPLES   = int(args[5]) if len(args) > 5 else 64

    scene = load_scene(scene_file)
    (sph_cx, sph_cy, sph_cz, sph_r,
     sph_ar, sph_ag, sph_ab, sph_refl, sph_shin,
     pln_nx, pln_ny, pln_nz, pln_d,
     pln_ar, pln_ag, pln_ab, pln_refl, pln_shin,
     lights_x, lights_y, lights_z, lights_int, amb) = scene

    cam_origin, cam_ll, cam_h, cam_v = make_camera(W, H)

    # Warmup JIT: compilar antes de medir (imagen minima 4x2)
    _ = render(4, 2, 2, 1,
               cam_origin[0], cam_origin[1], cam_origin[2],
               cam_ll[0], cam_ll[1], cam_ll[2],
               cam_h[0],  cam_h[1],  cam_h[2],
               cam_v[0],  cam_v[1],  cam_v[2],
               sph_cx, sph_cy, sph_cz, sph_r,
               sph_ar, sph_ag, sph_ab, sph_refl, sph_shin,
               pln_nx, pln_ny, pln_nz, pln_d,
               pln_ar, pln_ag, pln_ab, pln_refl, pln_shin,
               lights_x, lights_y, lights_z, lights_int,
               amb[0], amb[1], amb[2])

    # Render real — cronometro empieza aqui, despues del JIT
    t0 = time.perf_counter()

    img = render(W, H, MAX_DEPTH, SAMPLES,
                 cam_origin[0], cam_origin[1], cam_origin[2],
                 cam_ll[0], cam_ll[1], cam_ll[2],
                 cam_h[0],  cam_h[1],  cam_h[2],
                 cam_v[0],  cam_v[1],  cam_v[2],
                 sph_cx, sph_cy, sph_cz, sph_r,
                 sph_ar, sph_ag, sph_ab, sph_refl, sph_shin,
                 pln_nx, pln_ny, pln_nz, pln_d,
                 pln_ar, pln_ag, pln_ab, pln_refl, pln_shin,
                 lights_x, lights_y, lights_z, lights_int,
                 amb[0], amb[1], amb[2])

    elapsed = time.perf_counter() - t0

    # Guardar PNG (flip vertical: j=0 es la fila inferior, igual que PPM)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.imsave(output_file, img[::-1], vmin=0.0, vmax=1.0)

    print(f"Resolucion  : {W}x{H}",              file=sys.stderr)
    print(f"Prof. maxima: {MAX_DEPTH} rebotes",   file=sys.stderr)
    print(f"Muestras/px : {SAMPLES}",             file=sys.stderr)
    print(f"Rayos total : {W*H*SAMPLES}",         file=sys.stderr)
    print(f"Esferas     : {sph_cx.shape[0]}",     file=sys.stderr)
    print(f"Tiempo      : {elapsed:.6f} s",       file=sys.stderr)
    print(f"Imagen      : {output_file}",         file=sys.stderr)


if __name__ == "__main__":
    main()
