// scene.hpp - Tipos, geometria y utilidades compartidos por todos los renderizadores.
//
// Cada binario se compila desde un unico .cpp que incluye este header,
// por lo que no hay violaciones de ODR aunque las funciones no sean inline.
#pragma once

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- Vec3 -------------------------------------------------------------------
struct Vec3 {
    double x = 0, y = 0, z = 0;

    Vec3() = default;
    Vec3(double x, double y, double z) : x(x), y(y), z(z) {}

    Vec3  operator+(const Vec3& v) const { return {x+v.x, y+v.y, z+v.z}; }
    Vec3  operator-(const Vec3& v) const { return {x-v.x, y-v.y, z-v.z}; }
    Vec3  operator*(double  t)     const { return {x*t,   y*t,   z*t  }; }
    Vec3  operator*(const Vec3& v) const { return {x*v.x, y*v.y, z*v.z}; }
    Vec3  operator/(double  t)     const { return {x/t,   y/t,   z/t  }; }
    Vec3  operator-()              const { return {-x, -y, -z}; }
    Vec3& operator+=(const Vec3& v) { x+=v.x; y+=v.y; z+=v.z; return *this; }

    double dot(const Vec3& v)   const { return x*v.x + y*v.y + z*v.z; }
    Vec3   cross(const Vec3& v) const {
        return {y*v.z - z*v.y, z*v.x - x*v.z, x*v.y - y*v.x};
    }
    double length2()    const { return dot(*this); }
    double length()     const { return std::sqrt(length2()); }
    Vec3   normalized() const { return *this / length(); }
};

inline Vec3 operator*(double t, const Vec3& v) { return v * t; }

inline double clampd(double x, double lo = 0.0, double hi = 1.0) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// --- RNG --------------------------------------------------------------------
// Semilla funcion del indice del pixel: resultados deterministas y sin
// condiciones de carrera al paralelizar (cada thread opera sobre pixeles distintos).
struct RNG {
    std::mt19937_64 gen;
    std::uniform_real_distribution<double> dist{0.0, 1.0};

    explicit RNG(uint64_t seed) : gen(seed) {}
    double uniform() { return dist(gen); }
};

// Direccion uniforme en la esfera unitaria (metodo de rechazo).
Vec3 random_unit_sphere(RNG& rng) {
    while (true) {
        Vec3 p = {rng.uniform()*2-1, rng.uniform()*2-1, rng.uniform()*2-1};
        double l2 = p.length2();
        if (l2 <= 1.0 && l2 > 1e-10) return p / std::sqrt(l2);
    }
}

// Direccion aleatoria en el hemisferio orientado por `normal`, con distribucion
// proporcional al coseno del angulo. Sumar un vector unitario aleatorio a la
// punta de `normal` produce esa distribucion; el estimador Monte Carlo
// simplifica a: contribucion = color * L_incidente  (sin factor explicito)
Vec3 random_hemisphere(const Vec3& normal, RNG& rng) {
    Vec3 d = normal + random_unit_sphere(rng);
    if (d.length2() < 1e-10) return normal;
    return d.normalized();
}

// --- Tipos de la escena -----------------------------------------------------
struct Material { Vec3 albedo; double reflectivity, shininess; };
struct Ray      { Vec3 origin, dir; Vec3 at(double t) const { return origin + dir*t; } };

struct Hit {
    double   t     = std::numeric_limits<double>::infinity();
    Vec3     point, normal;
    Material mat;
    bool     valid = false;
};

struct Sphere {
    Vec3 center; double radius; Material mat;

    // Interseccion rayo-esfera: cuadratica a*t^2 + 2*hb*t + c = 0.
    Hit intersect(const Ray& r) const {
        Vec3   oc = r.origin - center;
        double a  = r.dir.dot(r.dir), hb = oc.dot(r.dir);
        double c  = oc.dot(oc) - radius*radius, disc = hb*hb - a*c;
        if (disc < 0) return {};
        double sq = std::sqrt(disc), t = (-hb - sq) / a;
        if (t < 1e-4) t = (-hb + sq) / a;
        if (t < 1e-4) return {};
        Vec3 p = r.at(t);
        return {t, p, (p - center).normalized(), mat, true};
    }
};

struct Plane {
    Vec3 normal; double d; Material mat;

    // Interseccion rayo-plano: t = (d - n·o) / n·dir.
    // La normal apunta siempre hacia el lado del rayo.
    Hit intersect(const Ray& r) const {
        double denom = normal.dot(r.dir);
        if (std::abs(denom) < 1e-8) return {};
        double t = (d - normal.dot(r.origin)) / denom;
        if (t < 1e-4) return {};
        Vec3 n = (denom < 0) ? normal : -normal;
        return {t, r.at(t), n, mat, true};
    }
};

struct Light { Vec3 pos; double intensity; };

struct Scene {
    std::vector<Sphere> spheres;
    std::vector<Plane>  planes;
    std::vector<Light>  lights;
    Vec3 ambient = {0.02, 0.02, 0.02};
};

// --- Interseccion mas cercana ------------------------------------------------
Hit closest_hit(const Ray& ray, const Scene& sc) {
    Hit best;
    for (const auto& s : sc.spheres) { Hit h = s.intersect(ray); if (h.valid && h.t < best.t) best = h; }
    for (const auto& p : sc.planes)  { Hit h = p.intersect(ray); if (h.valid && h.t < best.t) best = h; }
    return best;
}

// --- Camara -----------------------------------------------------------------
struct Camera {
    Vec3 origin, lower_left, horiz, vert;

    Camera(Vec3 eye, Vec3 lookat, Vec3 up, double vfov_deg, int w, int h) : origin(eye) {
        double theta  = vfov_deg * M_PI / 180.0;
        double half_h = std::tan(theta/2.0), half_w = half_h*(double)w/h;
        Vec3 ww = (eye-lookat).normalized(), uu = up.cross(ww).normalized(), vv = ww.cross(uu);
        lower_left = eye - uu*half_w - vv*half_h - ww;
        horiz = uu*(2.0*half_w); vert = vv*(2.0*half_h);
    }

    // (s,t) en [0,1]^2: s horizontal (izq->der), t vertical (abajo->arriba)
    Ray get_ray(double s, double t) const {
        Vec3 dir = lower_left + s*horiz + t*vert - origin;
        return {origin, dir.normalized()};
    }
};

// --- Carga de escena --------------------------------------------------------
// Habitacion estilo Cornell box: 5 planos fijos + luz cenital + esferas desde archivo.
Scene load_scene(const std::string& path) {
    Scene sc;
    sc.planes = {
        {{ 0, 1, 0}, -1.5, {{0.75,0.75,0.75}, 0.05, 8}},  // piso
        {{ 0, 1, 0},  2.5, {{0.90,0.90,0.90}, 0.00, 1}},  // techo
        {{ 0, 0, 1}, -4.0, {{0.80,0.80,0.80}, 0.00, 1}},  // fondo
        {{ 1, 0, 0}, -3.0, {{0.75,0.15,0.15}, 0.00, 1}},  // izquierda (roja)
        {{-1, 0, 0}, -3.0, {{0.15,0.75,0.15}, 0.00, 1}},  // derecha (verde)
    };
    sc.lights.push_back({{ 0.5, 2.2, 0.5}, 1.00});

    std::ifstream f(path);
    if (!f) { std::cerr << "Error: no se puede abrir " << path << "\n"; std::exit(1); }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line); std::string tok; ss >> tok;
        if (tok != "sphere") continue;
        Sphere s; double r, g, b;
        ss >> s.center.x >> s.center.y >> s.center.z >> s.radius
           >> r >> g >> b >> s.mat.reflectivity >> s.mat.shininess;
        s.mat.albedo = {r, g, b};
        sc.spheres.push_back(s);
    }
    return sc;
}

// --- Guardado PPM -----------------------------------------------------------
void save_ppm(const std::string& path, const std::vector<Vec3>& pixels, int w, int h) {
    std::ofstream f(path);
    f << "P3\n" << w << " " << h << "\n255\n";
    for (int j = h-1; j >= 0; --j)
        for (int i = 0; i < w; ++i) {
            const Vec3& p = pixels[j*w+i];
            f << (int)(255.99*p.x) << ' ' << (int)(255.99*p.y) << ' ' << (int)(255.99*p.z) << '\n';
        }
}
