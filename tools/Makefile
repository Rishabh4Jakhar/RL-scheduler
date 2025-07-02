CC=clang
CCFLAGS=-O3
CXX=clang++
CXXFLAGS=-std=c++11 -O3 
DEBUG=
INCLUDE=-I/home/$(USER)/profiler/tools/inc 
FLAGS_OPENMP=-fopenmp=libomp
LIBRARY=-L/home/$(USER)/profiler/tools -Wl,--rpath,/home/$(USER)/profiler/tools

TARGETS := libhijack.so libpmu.so 
OBJ_FILES := hijack.o pmu.o 

all: clean $(TARGETS) clean-obj

libhijack.so: hijack.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(FLAGS_OPENMP) -shared -o $@ -fPIC $< -ldl -lpthread -lrt -lm

libpmu.so: pmu.c
	$(CC) $(CCFLAGS) $(INCLUDE) $(FLAGS_OPENMP) -shared -o $@ -fPIC $< -lpfm -lpthread -lrt -lm

clean-obj:
	rm -rf *.o

clean:
	rm -rf *.o $(TARGETS)