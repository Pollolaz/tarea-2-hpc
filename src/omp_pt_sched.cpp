// omp_pt_sched.cpp - Path tracer paralelo con OpenMP y scheduling configurable.
//
// Permite experimentar con distintas estrategias de planificacion (static,
// dynamic, guided) y valores de chunksize desde la linea de comandos.
//
// Uso: ./omp_pt_sched <scene.txt> <output.ppm> [W H D N] <threads> <schedule> [chunksize]
//   threads:   numero de threads OpenMP
//   schedule:  static | dynamic | guided
//   chunksize: tamano de chunk (opcional; 0 = default del runtime)

#include "pt.hpp"
#include <omp.h>
#include <cstring>

static omp_sched_t parse_schedule(const char* s) {
    if (std::strcmp(s, "static")  == 0) return omp_sched_static;
    if (std::strcmp(s, "dynamic") == 0) return omp_sched_dynamic;
    if (std::strcmp(s, "guided")  == 0) return omp_sched_guided;
    std::cerr << "ERROR: schedule desconocido \"" << s
              << "\". Usar: static | dynamic | guided\n";
    std::exit(1);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Uso: " << argv[0]
                  << " <scene.txt> <output.ppm> [W H D N] <threads> <schedule> [chunksize]\n";
        return 1;
    }

    const std::string scene_file  = argv[1];
    const std::string output_file = argv[2];
    const int W         = (argc > 3) ? std::stoi(argv[3]) : 800;
    const int H         = (argc > 4) ? std::stoi(argv[4]) : 600;
    const int MAX_DEPTH = (argc > 5) ? std::stoi(argv[5]) : 8;
    const int SAMPLES   = (argc > 6) ? std::stoi(argv[6]) : 64;
    const int THREADS   = (argc > 7) ? std::stoi(argv[7]) : omp_get_max_threads();

    // argv[8] = "static"|"dynamic"|"guided", argv[9] = chunksize (opcional)
    omp_sched_t SCHED = (argc > 8) ? parse_schedule(argv[8]) : omp_sched_static;
    const int CHUNK    = (argc > 9) ? std::atoi(argv[9]) : 0;

    // Fija schedule en tiempo de ejecucion
    omp_set_schedule(SCHED, CHUNK);

    Scene  sc  = load_scene(scene_file);
    Camera cam({0, 0.5, 3.8}, {0, 0, 0}, {0, 1, 0}, 55.0, W, H);

    std::vector<Vec3> pixels(W * H);
    auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(runtime) num_threads(THREADS)
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

    std::string sched_name = (argc > 8) ? std::string(argv[8]) : std::string("static");
    std::cerr << "Resolucion  : " << W << "x" << H << "\n"
              << "Prof. maxima: " << MAX_DEPTH << " rebotes\n"
              << "Muestras/px : " << SAMPLES << "\n"
              << "Rayos total : " << (long long)W*H*SAMPLES << "\n"
              << "Esferas     : " << sc.spheres.size() << "\n"
              << "Threads     : " << THREADS << "\n"
              << "Schedule    : " << sched_name << " chunk=" << CHUNK << "\n"
              << "Tiempo      : " << elapsed << " s\n"
              << "Imagen      : " << output_file << "\n";

    return 0;
}
