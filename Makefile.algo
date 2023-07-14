
# std::execution port: https://github.com/nvidia/stdexec
# std::mdspan port: https://github.com/kokkos/mdspan


CC=g++
CFLAGS= -std=c++20 -Ofast -march=native -DNDEBUG
CLIBS= -ltbb

NC=nvc++
NFLAGS= -std=c++20 -O4 -fast -march=native -DNDEBUG -Mllvm-fast -DNDEBUG

all: heat_algo

heat_algo: heat_algo.cpp
	$(CC) $(CFLAGS) -o $@ $^ $(CLIBS)

heat_algo_span: heat_algo_span.cpp
	$(CC) $(CFLAGS) -o $@ $^ $(CLIBS)

nvc_cpu: heat_algo_nvc.cpp
	$(NC) -stdpar=multicore $(NFLAGS) -o heat_algo_nvc_cpu $^

nvc_gpu: heat_algo_nvc.cpp
	$(NC) -stdpar=gpu $(NFLAGS) -o heat_algo_nvc_gpu $^

clean:
	rm -f heat_algo heat_algo_span heat_algo_nvc_cpu heat_algo_nvc_gpu

.PHONY: all clean
