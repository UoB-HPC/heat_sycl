#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <numeric>
#include <execution>
#include <algorithm>

// Key constants used in this program
#define PI acos(-1.0)                 // Pi
#define LINE "--------------------\n" // A line for fancy output

// Function definitions
void initial_value(const unsigned int n, const double dx, const double length, std::vector<double> &u);
void zero(const unsigned int n, std::vector<double> &u);
void solve(const unsigned int n, const double alpha, const double dx, const double dt, const std::vector<double> &u, std::vector<double> &u_tmp);
double solution(const double t, const double x, const double y, const double alpha, const double length);
double l2norm(const unsigned int n, const std::vector<double> &u, const unsigned int nsteps, const double dt, const double alpha, const double dx, const double length);

// Main function
int main(int argc, char *argv[])
{
  // Start the total program runtime timer
  auto start = std::chrono::high_resolution_clock::now();

  // Problem size, forms an nxn grid
  unsigned int n = 1000;

  // Number of timesteps
  unsigned int nsteps = 10;

  // Check for the correct number of arguments
  // Print usage and exits if not correct
  if (argc == 3)
  {
    // Set problem size from first argument
    n = atoi(argv[1]);
    if (n < 0)
    {
      std::cerr << "Error: n must be positive\n";
      exit(EXIT_FAILURE);
    }

    // Set number of timesteps from second argument
    nsteps = atoi(argv[2]);
    if (nsteps < 0)
    {
      std::cerr << "Error: nsteps must be positive\n";
      exit(EXIT_FAILURE);
    }
  }

  //
  // Set problem definition
  //
  double alpha = 0.1;           // heat equation coefficient
  double length = 1000.0;       // physical size of domain: length x length square
  double dx = length / (n + 1); // physical size of each cell (+1 as don't simulate boundaries as they are given)
  double dt = 0.5 / nsteps;     // time interval (total time of 0.5s)

  // Stability requires that dt/(dx^2) <= 0.5,
  double r = alpha * dt / (dx * dx);

  // Print message detailing runtime configuration
  std::cout << std::endl
            << " MMS heat equation" << std::endl
            << std::endl
            << LINE << std::endl
            << "Problem input" << std::endl
            << std::endl
            << " Grid size: " << n << " x " << n << std::endl
            << " Cell width: " << dx << std::endl
            << " Grid length: " << length << "x" << length << std::endl
            << std::endl
            << " Alpha: " << alpha << std::endl
            << std::endl
            << " Steps: " << nsteps << std::endl
            << " Total time: " << dt * (double)nsteps << std::endl
            << " Time step: " << dt << std::endl
            << LINE << std::endl;

  // Stability check
  std::cout << "Stability" << std::endl << std::endl;
  std::cout << " r value: " << r << std::endl;
  if (r > 0.5)
    std::cout << " Warning: unstable" << std::endl;
  std::cout << LINE << std::endl;

  // Allocate memory for temperature grid (u) and a temporary grid (u_tmp)
  std::vector<double> u(n * n);
  std::vector<double> u_tmp(n * n);

  // Set initial values of temperature grid
  initial_value(n, dx, length, u);
  // Initialize temporary grid to zero
  zero(n,u_tmp);

  //
  // Run through timesteps under the explicit scheme
  //

  // Start simulation timer
  auto tic = std::chrono::high_resolution_clock::now();
  for (int t = 0; t < nsteps; ++t)
  {
    // Call the solve kernel
    // Computes u_tmp at the next timestep
    // given the value of u at the current timestep
    solve(n, alpha, dx, dt, u, u_tmp);

    // Swap the grids (u and u_tmp)
    std::swap(u, u_tmp);
  }

  // Stop simulation timer
  auto toc = std::chrono::high_resolution_clock::now();

  //
  // Check the L2-norm of the computed solution
  // against the *known* solution from the MMS scheme
  //
  double norm = l2norm(n, u, nsteps, dt, alpha, dx, length);

  // Stop total timer
  auto stop = std::chrono::high_resolution_clock::now();

  // Print results
  std::cout
      << "Results" << std::endl
      << std::endl
      << "Error (L2norm): " << norm << std::endl
      << "Solve time (s): " << std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic).count() << std::endl
      << "Total time (s): " << std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count() << std::endl
      << "Bandwidth (GB/s): " << 1.0E-9 * 2.0 * n * n * nsteps * sizeof(double) / std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic).count() << std::endl
      << LINE << std::endl;

}

// Sets the mesh to an initial value, determined by the MMS scheme
void initial_value(const unsigned int n, const double dx, const double length, std::vector<double> &u)
{
  // Loop over all grid points (excluding boundaries)
  std::for_each_n(std::execution::par_unseq, u.begin(), n*n, [u = u.data(), n, dx, length](int index) {
    const int j = int(index / n); // Column index
    const int i = index -(j*n); // Row index

    const double x = (i + 1) * dx; // x-coordinate
    const double y = (j + 1) * dx; // y-coordinate

    // Compute the known solution using MMS scheme
    u[index] = sin(PI * x / length) * sin(PI * y / length); });
}

// Zero the array u
void zero(const unsigned int n, std::vector<double> &u)
{
  std::fill(std::execution::par_unseq, u.begin(), u.end(), 0.0);
}

// Function to solve the heat equation
void solve(const unsigned int n, const double alpha, const double dx, const double dt, const std::vector<double> &u, std::vector<double> &u_tmp)
{
  // Finite difference constant multiplier
  const double r = alpha * dt / (dx * dx);
  const double r2 = 1.0 - 4.0 * r;


  // Loop over all grid points (excluding boundaries)
  std::for_each_n(std::execution::par_unseq, u_tmp.begin(), n*n, [u_tmp=u_tmp.data(), u = u.data(), n, r, r2](int index) {
    const int j = int(index / n); // Column index
    const int i = index -(j*n); // Row index

    // Update the 5-point stencil, using boundary conditions on the edges of the domain.
    // Boundaries are zero because the MMS solution is zero there.
    u_tmp[index] = r2 * u[index] +
                       r * ((i < n - 1) ? u[index+1] : 0.0) +
                       r * ((i > 0) ? u[index-1] : 0.0) +
                       r * ((j < n - 1) ? u[index + n] : 0.0) +
                       r * ((j > 0) ? u[index-n] : 0.0);
  });
}

// Function to compute the exact solution at a given time and position
double solution(const double t, const double x, const double y, const double alpha, const double length)
{
  return exp(-t * alpha * PI * PI / length / length) * (sin(PI * x / length) * sin(PI * y / length));
}

// Computes the L2-norm of the computed grid and the MMS known solution
// The known solution is the same as the boundary function.
double l2norm(const unsigned int n, const std::vector<double> &u, const unsigned int nsteps, const double dt, const double alpha, const double dx, const double length)
{
  // Final (real) time simulated
  double time = dt * (double)nsteps;

  return sqrt(std::transform_reduce(std::execution::par_unseq, u.begin(), u.end(), 0.0, std::plus<double>(), [u=u.data(), n, nsteps, dt, dx, alpha, length](int idx) {
    int j = int(idx / n); // Column index
    int i = idx -(j*n); // Row index

    double y = (j + 1) * dx; // y-coordinate
    double x = (i + 1) * dx; // x-coordinate

    double answer = solution(nsteps * dt, x, y, alpha, length);
    return (u[idx] - answer) * (u[idx] - answer);

  }));
}
