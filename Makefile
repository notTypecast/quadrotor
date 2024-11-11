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

# Paths for CasADi
CASADI_HEADER=/usr/local/include/casadi
CASADI_LIB=/usr/local/lib
CASADI_FLAGS = -lcasadi -I$(CASADI_HEADER) -L$(CASADI_LIB)

# Build target
build-cem: main_cem.cpp
ifeq ($(USE_TBB), true)
	$(CXX) $(INCLUDES_CEM) main_cem.cpp -o build/main_cem $(CXXFLAGS) $(TBB_FLAGS) -DUSE_TBB=true -DUSE_TBB_ONEAPI=true
else
	$(CXX) $(INCLUDES_CEM) main_cem.cpp -o build/main_cem $(CXXFLAGS)
endif

run-cem: build/main_cem
	./build/main_cem

build-num: main_num.cpp
	$(CXX) $(INCLUDES_NUM) main_num.cpp -o build/main_num $(CXXFLAGS) $(CASADI_FLAGS)

run-num: build/main_num
	./build/main_num

build-quad: main_quad.cpp
	$(CXX) $(INCLUDES_NUM) main_quad.cpp -o build/main_quad $(CXXFLAGS) $(CASADI_FLAGS)

run-quad: build/main_quad
	./build/main_quad

build-quad-pre: main_quad_pretrained.cpp
	$(CXX) $(INCLUDES_NUM) main_quad_pretrained.cpp -o build/main_quad_pretrained $(CXXFLAGS) $(CASADI_FLAGS)

run-quad-pre: build/main_quad_pretrained
	./build/main_quad_pretrained

clean:
	rm -f build/*