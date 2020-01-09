
/*
** PROGRAM: heat equation solve
**
** PURPOSE: This program will explore use of an explicit
**          finite difference method to solve the heat
**          equation under a method of manufactured solution (MMS)
**          scheme. The solution has been set to be a simple 
**          function based on exponentials and trig functions.
**
**          A finite difference scheme is used on a 1000x1000 cube.
**          A total of 0.5 units of time are simulated.
**
**          The MMS solution has been adapted from
**          G.W. Recktenwald (2011). Finite difference approximations
**          to the Heat Equation. Portland State University.
**
**
** USAGE:   Run with two arguments:
**          First is the number of cells.
**          Second is the number of timesteps.
**
**          For example, with 100x100 cells and 10 steps:
**
**          ./heat 100 10
**
**
** HISTORY: Written by Tom Deakin, Oct 2018
**          Ported to SYCL by Tom Deakin, Nov 2019
**          Ported to OpenCL by Tom Deakin, Jan 2020
**
*/

#include <iostream>
#include <chrono>
#include <cmath>
#include <fstream>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <CL/cl2.hpp>

// Key constants used in this program
#define PI std::acos(-1.0) // Pi
#define LINE "--------------------" // A line for fancy output

// Function definitions
double solution(const double t, const double x, const double y, const double alpha, const double length);
double l2norm(const unsigned int n, const double * u, const int nsteps, const double dt, const double alpha, const double dx, const double length);


// Main function
int main(int argc, char *argv[]) {

  // Start the total program runtime timer
  auto start = std::chrono::high_resolution_clock::now();

  // Problem size, forms an nxn grid
  unsigned int n = 1000;

  // Number of timesteps
  int nsteps = 10;


  // Check for the correct number of arguments
  // Print usage and exits if not correct
  if (argc == 3) {

    // Set problem size from first argument
    n = atoi(argv[1]);
    if (n < 0) {
      std::cerr << "Error: n must be positive" << std::endl;
      exit(EXIT_FAILURE);
    }

    // Set number of timesteps from second argument
    nsteps = atoi(argv[2]);
    if (nsteps < 0) {
      std::cerr << "Error: nsteps must be positive" << std::endl;
      exit(EXIT_FAILURE);
    }
  }


  //
  // Set problem definition
  //
  double alpha = 0.1;          // heat equation coefficient
  double length = 1000.0;      // physical size of domain: length x length square
  double dx = length / (n+1);  // physical size of each cell (+1 as don't simulate boundaries as they are given)
  double dt = 0.5 / nsteps;    // time interval (total time of 0.5s)


  // Stability requires that dt/(dx^2) <= 0.5,
  double r = alpha * dt / (dx * dx);

  // Initalise an OpenCL queue on a GPU device
  // Get list of platforms
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  if (platforms.size() < 1) {
    std::cerr << "Error: no OpenCL platforms" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::vector<cl::Device> device_list;
  platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &device_list);
  if (device_list.size() < 1) {
    std::cerr << "Error: no OpenCL devices" << std::endl;
    exit(EXIT_FAILURE);
  }
  cl::Device device = device_list[0];
  std::string device_name;
  device.getInfo(CL_DEVICE_NAME, &device_name);

  cl::Context context(device);
  cl::CommandQueue queue(context);

  // Create kernels
  std::ifstream stream("kernels.cl");
  if (!stream.is_open()) {
    std::cerr << "Error: Cannot open kernels.cl file" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::string kernel_source(
    std::istreambuf_iterator<char>(stream),
    (std::istreambuf_iterator<char>()));

  cl::Program program(context, kernel_source, true);
  cl::KernelFunctor<cl_uint, cl_double, cl_double, cl::Buffer> initial_value(program, "initial_value");
  cl::KernelFunctor<cl_uint, cl::Buffer> zero(program, "zero");
  cl::KernelFunctor<cl_uint, cl_double, cl_double, cl_double, cl::Buffer, cl::Buffer> solve(program, "solve");

  // Print message detailing runtime configuration
  std::cout
    << std::endl
    << " MMS heat equation" << std::endl << std::endl
    << LINE << std::endl
    << "Problem input" << std::endl << std::endl
    << " Grid size: " << n << " x " << n << std::endl
    << " Cell width: " << dx << std::endl
    << " Grid length: " << length << "x" << length << std::endl
    << std::endl
    << " Alpha: " << alpha << std::endl
    << std::endl
    << " Steps: " <<  nsteps << std::endl
    << " Total time: " << dt*(double)nsteps << std::endl
    << " Time step: " << dt << std::endl
    << " SYCL device: " << device_name << std::endl
    << LINE << std::endl;




  // Stability check
  std::cout << "Stability" << std::endl << std::endl;
  std::cout << " r value: " << r << std::endl;
  if (r > 0.5)
    std::cout << " Warning: unstable" << std::endl;
  std::cout << LINE << std::endl;


  // Allocate two nxn grids
  cl::Buffer u{context, CL_MEM_READ_WRITE, sizeof(double)*n*n};
  cl::Buffer u_tmp{context, CL_MEM_READ_WRITE, sizeof(double)*n*n};

  // Set the initial value of the grid under the MMS scheme
  initial_value(cl::EnqueueArgs(queue, cl::NDRange(n,n)), n, dx, length, u);
  zero(cl::EnqueueArgs(queue, cl::NDRange(n*n)), n, u_tmp);

  // Ensure everything is initalised on the device
  queue.finish();

  //
  // Run through timesteps under the explicit scheme
  //

  // Start the solve timer
  auto tic = std::chrono::high_resolution_clock::now();
  for (int t = 0; t < nsteps; ++t) {

    // Call the solve kernel
    // Computes u_tmp at the next timestep
    // given the value of u at the current timestep
    if (t % 2 == 0)
      solve(cl::EnqueueArgs(queue, cl::NDRange(n,n)), n, alpha, dx, dt, u, u_tmp);
    else
      solve(cl::EnqueueArgs(queue, cl::NDRange(n,n)), n, alpha, dx, dt, u_tmp, u);

  }
  // Stop solve timer
  queue.finish();
  auto toc = std::chrono::high_resolution_clock::now();

  // Get access to u on the host
  double *u_host = new double[n*n];
  queue.enqueueReadBuffer(u, CL_TRUE, 0, sizeof(double)*n*n, u_host);

  //
  // Check the L2-norm of the computed solution
  // against the *known* solution from the MMS scheme
  //
  double norm = l2norm(n, u_host, nsteps, dt, alpha, dx, length);

  // Stop total timer
  auto stop = std::chrono::high_resolution_clock::now();

  // Print results
  std::cout
    << "Results" << std::endl << std::endl
    << "Error (L2norm): " << norm << std::endl
    << "Solve time (s): " << std::chrono::duration_cast<std::chrono::duration<double>>(toc-tic).count() << std::endl
    << "Total time (s): " << std::chrono::duration_cast<std::chrono::duration<double>>(stop-start).count() << std::endl
    << "Bandwidth (GB/s): " << 1.0E-9*2.0*n*n*nsteps*sizeof(double)/std::chrono::duration_cast<std::chrono::duration<double>>(toc-tic).count() << std::endl
    << LINE << std::endl;

  delete[] u_host;

}

// True answer given by the manufactured solution
double solution(const double t, const double x, const double y, const double alpha, const double length) {

  return exp(-2.0*alpha*PI*PI*t/(length*length)) * sin(PI*x/length) * sin(PI*y/length);

}


// Computes the L2-norm of the computed grid and the MMS known solution
// The known solution is the same as the boundary function.
double l2norm(const unsigned int n, const double * u, const int nsteps, const double dt, const double alpha, const double dx, const double length) {

  // Final (real) time simulated
  double time = dt * (double)nsteps;

  // L2-norm error
  double l2norm = 0.0;

  // Loop over the grid and compute difference of computed and known solutions as an L2-norm
  double y = dx;
  for (int j = 0; j < n; ++j) {
    double x = dx;
    for (int i = 0; i < n; ++i) {
      double answer = solution(time, x, y, alpha, length);
      l2norm += (u[i+j*n] - answer) * (u[i+j*n] - answer);

      x += dx;
    }
    y += dx;
  }

  return sqrt(l2norm);

}

