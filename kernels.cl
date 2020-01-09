
#define PI acos(-1.0) // Pi

// Sets the mesh to an initial value, determined by the MMS scheme
kernel void initial_value(const unsigned int n, const double dx, const double length, global double * u) {
  int i = get_global_id(0);
  int j = get_global_id(1);
  unsigned int idx = i+j*n;
  double y = dx * (j+1); // Physical y position
  double x = dx * (i+1); // Physical x position
  u[idx] = sin(PI * x / length) * sin(PI * y / length);
}


// Zero the array u
kernel void zero(const unsigned int n, global double * u) {

  u[get_global_id(0)] = 0.0;

}


// Compute the next timestep, given the current timestep
// Loop over the nxn grid
kernel void solve(const unsigned int n, const double alpha, const double dx, const double dt, global double * restrict u, global double * restrict u_tmp) {

  // Finite difference constant multiplier
  const double r = alpha * dt / (dx * dx);
  const double r2 = 1.0 - 4.0*r;

  size_t j = get_global_id(1);
  size_t i = get_global_id(0);

  // Update the 5-point stencil, using boundary conditions on the edges of the domain.
  // Boundaries are zero because the MMS solution is zero there.
  u_tmp[i+j*n] =  r2 * u[i+j*n] +
  r * ((i < n-1) ? u[i+1+j*n] : 0.0) +
  r * ((i > 0)   ? u[i-1+j*n] : 0.0) +
  r * ((j < n-1) ? u[i+(j+1)*n] : 0.0) +
  r * ((j > 0)   ? u[i+(j-1)*n] : 0.0);
}

