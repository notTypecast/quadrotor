# Compiler and paths
CXX = g++
INCLUDES_CEM = -I/usr/include/eigen3/ -I. -I./simple_nn/src -I./algevo/src
INCLUDES_NUM = -I/usr/include/eigen3/ -I.
CXXFLAGS = -O3

# Paths for TBB
USE_TBB=true
TBB_HEADER=/usr/include/tbb
TBB_LIB=/usr/lib/x86_64-linux-gnu
TBB_FLAGS = -ltbb -I$(TBB_HEADER) -L$(TBB_LIB)

# MPI
USE_MPI=true
N_PROCESSES=16

# Paths for CasADi
CASADI_HEADER=/usr/local/include/casadi
CASADI_LIB=/usr/local/lib
CASADI_FLAGS = -lcasadi -I$(CASADI_HEADER) -L$(CASADI_LIB)

# Defined variable
DEFINED =

# Build target
build-2d-cem: main_2d_cem.cpp
ifeq ($(USE_TBB), true)
	$(CXX) $(INCLUDES_CEM) main_2d_cem.cpp -o build/main_2d_cem $(CXXFLAGS) $(TBB_FLAGS) -DUSE_TBB=true -DUSE_TBB_ONEAPI=true
else
	$(CXX) $(INCLUDES_CEM) main_2d_cem.cpp -o build/main_2d_cem $(CXXFLAGS)
endif

run-2d-cem: build/main_2d_cem
	./build/main_2d_cem

build-2d: main_2d.cpp
ifeq ($(USE_MPI), true)
	mpicxx $(INCLUDES_NUM) main_2d.cpp -o build/main_2d $(CXXFLAGS) $(CASADI_FLAGS) -DQUAD_WITH_MPI $(DEFINED)
else
	$(CXX) $(INCLUDES_NUM) main_2d.cpp -o build/main_2d $(CXXFLAGS) $(CASADI_FLAGS) $(DEFINED)
endif

run-2d: build/main_2d
ifeq ($(USE_MPI), true)
	mpiexec -n $(N_PROCESSES) ./build/main_2d
else
	./build/main_2d
endif

build-3d: main_3d.cpp
ifeq ($(USE_MPI), true)
	LD_LIBRARY_PATH=/usr/local/lib mpicxx $(INCLUDES_NUM) main_3d.cpp -o build/main_3d $(CXXFLAGS) $(CASADI_FLAGS) -DQUAD_WITH_MPI $(DEFINED)
else
	$(CXX) $(INCLUDES_NUM) main_3d.cpp -o build/main_3d $(CXXFLAGS) $(CASADI_FLAGS) $(DEFINED)
endif

run-3d: build/main_3d
ifeq ($(USE_MPI), true)
	LD_LIBRARY_PATH=/usr/local/lib mpiexec -n $(N_PROCESSES) ./build/main_3d
else
	./build/main_3d
endif

build-3d-pre: main_3d_pretrained.cpp
	$(CXX) $(INCLUDES_NUM) main_3d_pretrained.cpp -o build/main_3d_pretrained $(CXXFLAGS) $(CASADI_FLAGS)

run-3d-pre: build/main_3d_pretrained
	./build/main_3d_pretrained

clean:
	rm -f build/*