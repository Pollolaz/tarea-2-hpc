// serial_pt.cpp - Path tracer serial de referencia.
//
// A diferencia de serial.cpp, la luz rebota entre superficies difusas: en cada
// interseccion se elige una direccion aleatoria del hemisferio y se acumula la
// luz que llega por ella (Monte Carlo). El promedio sobre N muestras converge a
// la iluminacion global (color bleeding, sombras suaves, luz indirecta).
//
// Uso: ./serial_pt <scene.txt> <output.ppm> [W H D N]
//   N: caminos por pixel. Default: 64.

#include "pt.hpp"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Uso: " << argv[0] << " <scene.txt> <output.ppm> [W H D N]\n";
        return 1;
    }

    const std::string scene_file  = argv[1];
    const std::string output_file = argv[2];
    const int W         = (argc > 3) ? std::stoi(argv[3]) : 800;
    const int H         = (argc > 4) ? std::stoi(argv[4]) : 600;
    const int MAX_DEPTH = (argc > 5) ? std::stoi(argv[5]) : 8;
    const int SAMPLES   = (argc > 6) ? std::stoi(argv[6]) : 64;

    Scene  sc  = load_scene(scene_file);
    Camera cam({0, 0.5, 3.8}, {0, 0, 0}, {0, 1, 0}, 55.0, W, H);

    std::vector<Vec3> pixels(W * H);
    auto t0 = std::chrono::high_resolution_clock::now();

    // Cada pixel lanza SAMPLES caminos y promedia. La semilla del RNG es funcion
    // del indice del pixel: resultados deterministas y sin condiciones de carrera
    // al paralelizar (cada thread tiene su propio estado de RNG).
    for (int j = 0; j < H; ++j)
        for (int i = 0; i < W; ++i) {
            RNG  rng(static_cast<uint64_t>(j) * W + i + 1);
            Vec3 color(0, 0, 0);
            for (int s = 0; s < SAMPLES; ++s) {
                double u = (i + rng.uniform()) / W;
                double v = (j + rng.uniform()) / H;
                color += trace_path(cam.get_ray(u, v), sc, MAX_DEPTH, rng);
            }
            pixels[j*W + i] = color / static_cast<double>(SAMPLES);
        }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    save_ppm(output_file, pixels, W, H);

    std::cerr << "Resolucion  : " << W << "x" << H << "\n"
              << "Prof. maxima: " << MAX_DEPTH << " rebotes\n"
              << "Muestras/px : " << SAMPLES << "\n"
              << "Rayos total : " << (long long)W*H*SAMPLES << "\n"
              << "Esferas     : " << sc.spheres.size() << "\n"
              << "Tiempo      : " << elapsed << " s\n"
              << "Imagen      : " << output_file << "\n";

    return 0;
}
