CXX      = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra

W       = 800
H       = 600
D       = 8    # profundidad máxima de rebotes
SPP     = 4    # muestras por eje en serial (total: SPP^2 rayos/px)
SAMPLES = 64   # caminos por píxel en serial_pt

.PHONY: all clean render render_pt

THREADS = 4
all: serial serial_pt omp_pt omp_pt_sched

serial: serial.cpp scene.hpp
	$(CXX) $(CXXFLAGS) -o serial serial.cpp

serial_pt: serial_pt.cpp pt.hpp scene.hpp
	$(CXX) $(CXXFLAGS) -o serial_pt serial_pt.cpp

omp_pt: omp_pt.cpp pt.hpp scene.hpp
	$(CXX) $(CXXFLAGS) -fopenmp -o omp_pt omp_pt.cpp

omp_pt_sched: omp_pt_sched.cpp pt.hpp scene.hpp
	$(CXX) $(CXXFLAGS) -fopenmp -o omp_pt_sched omp_pt_sched.cpp

render: serial
	./serial scene.txt output.ppm $(W) $(H) $(D) $(SPP)

render_pt: serial_pt
	./serial_pt scene.txt output_pt.ppm $(W) $(H) $(D) $(SAMPLES)

render_omp: omp_pt
	./omp_pt scene.txt output_omp.ppm $(W) $(H) $(D) $(SAMPLES) $(THREADS)

clean:
	rm -f serial serial_pt omp_pt output.ppm output_pt.ppm output_omp.ppm
