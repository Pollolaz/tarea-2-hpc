// serial.cpp - Ray tracer serial de referencia.
//
// Iluminacion directa unicamente: sombra, difuso (Lambert) y especular (Phong).
// La luz no rebota entre superficies; las zonas en sombra solo reciben luz
// ambiental. El bucle sobre pixeles es el candidato natural a paralelizar.
//
// Uso: ./serial <scene.txt> <output.ppm> [W H D SPP]
//   SPP: muestras por eje (total SPP*SPP rayos/px). Default: 1.

#include "scene.hpp"

// Implementacion iterativa: acumula iluminacion directa en cada rebote especular
// hasta que el rayo llega al fondo o agota la profundidad maxima.
Vec3 trace(Ray ray, const Scene& sc, int max_depth) {
    Vec3 color(0, 0, 0);
    Vec3 mask(1, 1, 1);  // atenuacion acumulada por reflectividad

    for (int depth = 0; depth < max_depth; ++depth) {
        Hit h = closest_hit(ray, sc);

        if (!h.valid) {
            double t   = 0.5 * (ray.dir.y + 1.0);
            Vec3   sky = (1.0-t)*Vec3(1,1,1) + t*Vec3(0.5,0.7,1.0);
            color += mask * sky * 0.25;
            break;
        }

        Vec3 local = sc.ambient * h.mat.albedo;
        for (const auto& light : sc.lights) {
            Vec3   ldir  = (light.pos - h.point).normalized();
            double ldist = (light.pos - h.point).length();
            Hit shadow = closest_hit({h.point + h.normal*1e-4, ldir}, sc);
            if (shadow.valid && shadow.t < ldist) continue;  // en sombra

            double diff = std::max(0.0, h.normal.dot(ldir)) * light.intensity;
            local += h.mat.albedo * diff;

            Vec3   rl   = (2.0*h.normal.dot(ldir))*h.normal - ldir;
            double spec = std::pow(std::max(0.0, rl.dot(-ray.dir)),
                                   h.mat.shininess) * light.intensity;
            local += Vec3(1,1,1) * spec * 0.5;
        }
        color += mask * (1.0 - h.mat.reflectivity) * local;

        if (h.mat.reflectivity < 1e-4) break;

        // Rayo reflejado: r = d - 2*(d·n)*n
        mask = mask * h.mat.reflectivity;
        Vec3 rdir = ray.dir - (2.0*h.normal.dot(ray.dir))*h.normal;
        ray = {h.point + h.normal*1e-4, rdir.normalized()};
    }

    return {clampd(color.x), clampd(color.y), clampd(color.z)};
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Uso: " << argv[0] << " <scene.txt> <output.ppm> [W H D SPP]\n";
        return 1;
    }

    const std::string scene_file  = argv[1];
    const std::string output_file = argv[2];
    const int W         = (argc > 3) ? std::stoi(argv[3]) : 800;
    const int H         = (argc > 4) ? std::stoi(argv[4]) : 600;
    const int MAX_DEPTH = (argc > 5) ? std::stoi(argv[5]) : 8;
    const int SPP       = (argc > 6) ? std::stoi(argv[6]) : 1;

    Scene  sc  = load_scene(scene_file);
    sc.ambient = {0.08, 0.08, 0.08};  // sin rebotes indirectos se necesita mas ambiente
    Camera cam({0, 0.5, 3.8}, {0, 0, 0}, {0, 1, 0}, 55.0, W, H);

    std::vector<Vec3> pixels(W * H);
    auto t0 = std::chrono::high_resolution_clock::now();

    // Cada pixel es independiente: candidato directo a paralelizar.
    for (int j = 0; j < H; ++j)
        for (int i = 0; i < W; ++i) {
            Vec3 color(0, 0, 0);
            for (int sy = 0; sy < SPP; ++sy)
                for (int sx = 0; sx < SPP; ++sx) {
                    double u = (i + (sx + 0.5) / SPP) / W;
                    double v = (j + (sy + 0.5) / SPP) / H;
                    color += trace(cam.get_ray(u, v), sc, MAX_DEPTH);
                }
            pixels[j*W + i] = color / (double)(SPP * SPP);
        }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    save_ppm(output_file, pixels, W, H);

    std::cerr << "Resolucion  : " << W << "x" << H << "\n"
              << "Prof. maxima: " << MAX_DEPTH << " rebotes\n"
              << "Muestras/px : " << SPP << "x" << SPP << " (" << SPP*SPP << " rayos/px)\n"
              << "Esferas     : " << sc.spheres.size() << "\n"
              << "Tiempo      : " << elapsed << " s\n"
              << "Imagen      : " << output_file << "\n";

    return 0;
}
