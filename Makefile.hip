
HIP_CC=/opt/rocm/hip/bin/hipcc

heat_hip: heat_hip.hip.cpp Makefile.hip
	$(HIP_CC) -std=c++11 -O3 heat_hip.hip.cpp -o $@

