
CXX=clang++ --gcc-toolchain=/nfs/software/x86_64/gcc/7.4.0
CFLAGS=-std=c++11 -O3 -fsycl
LIBS=-lOpenCL

heat: heat_sycl.cpp Makefile
	$(CXX) $(CFLAGS) heat_sycl.cpp $(LIBS) -o $@


