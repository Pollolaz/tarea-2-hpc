// pt.hpp - Algoritmo de path tracing (usado por serial_pt).
#pragma once

#include "scene.hpp"

// En cada rebote: (1) iluminacion directa con rayo de sombra, igual que en RT;
// (2) rayo difuso aleatorio cuya contribucion pondera `throughput` (producto
// acumulado de colores a lo largo del camino). Promediar N muestras converge
// a la iluminacion global (color bleeding, sombras suaves, luz indirecta).
Vec3 trace_path(Ray ray, const Scene& sc, int max_depth, RNG& rng) {
    Vec3 color(0, 0, 0), throughput(1, 1, 1);

    for (int depth = 0; depth < max_depth; ++depth) {
        Hit h = closest_hit(ray, sc);

        if (!h.valid) {
            // Fondo: gradiente celeste como fuente de luz ambiente
            double t   = 0.5 * (ray.dir.y + 1.0);
            Vec3   sky = (1.0-t)*Vec3(1,1,1) + t*Vec3(0.5,0.7,1.0);
            color += throughput * sky * 0.6;
            break;
        }

        // Iluminacion directa: rayo de sombra hacia cada luz
        Vec3 direct = sc.ambient * h.mat.albedo;
        for (const auto& light : sc.lights) {
            Vec3   ldir  = (light.pos - h.point).normalized();
            double ldist = (light.pos - h.point).length();
            Hit sh = closest_hit({h.point + h.normal*1e-4, ldir}, sc);
            if (sh.valid && sh.t < ldist) continue;  // en sombra
            double diff = std::max(0.0, h.normal.dot(ldir)) * light.intensity;
            direct += h.mat.albedo * diff;
            Vec3   rl   = (2.0*h.normal.dot(ldir))*h.normal - ldir;
            double spec = std::pow(std::max(0.0, rl.dot(-ray.dir)), h.mat.shininess) * light.intensity;
            direct += Vec3(1,1,1) * spec * 0.5;
        }
        color += throughput * (1.0 - h.mat.reflectivity) * direct;

        // Siguiente rebote: especular (determinista) o difuso (aleatorio)
        if (h.mat.reflectivity > 1e-4) {
            Vec3 rdir  = ray.dir - (2.0*h.normal.dot(ray.dir))*h.normal;
            ray        = {h.point + h.normal*1e-4, rdir.normalized()};
            throughput = throughput * h.mat.reflectivity;
        } else {
            ray        = {h.point + h.normal*1e-4, random_hemisphere(h.normal, rng)};
            throughput = throughput * h.mat.albedo;
        }

        // Terminacion anticipada: throughput casi nulo -> contribucion despreciable
        if (std::max(throughput.x, std::max(throughput.y, throughput.z)) < 0.01) break;
    }
    return {clampd(color.x), clampd(color.y), clampd(color.z)};
}
