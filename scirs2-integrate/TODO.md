# scirs2-integrate TODO

This module provides numerical integration functionality similar to SciPy's integrate module.

## Direct SciPy Integration Method Parity Status

| SciPy Function | Status | Notes |
|----------------|--------|-------|
| `quad` | ✅ Implemented | Basic adaptive quadrature |
| `dblquad`/`tplquad` | ✅ Implemented | Multi-dimensional quadrature |
| `nquad` | ✅ Implemented | General multi-dimensional |
| `fixed_quad` | ✅ Implemented | As Gaussian quadrature |
| `trapezoid` | ✅ Implemented | Basic non-adaptive |
| `simpson` | ✅ Implemented | Basic non-adaptive |
| `romb` | ✅ Implemented | As Romberg integration |
| `solve_ivp` | ✅ Implemented | With multiple methods |
| `RK23` | ✅ Implemented | Fixed |
| `RK45` | ✅ Implemented | Fixed |
| `BDF` | ✅ Implemented | Enhanced with improved Jacobian handling and error estimation |
| `solve_bvp` | ✅ Implemented | Boundary Value Problem solver |
| `DOP853` | ✅ Implemented | Higher-order Runge-Kutta |
| `Radau` | ⚠️ Partially implemented | Implicit Runge-Kutta (Newton iteration fails with mass matrices) |
| `LSODA` | ✅ Implemented | Adaptive switching method with enhanced stiffness detection |
| `qmc_quad` | ✅ Implemented | Quasi-Monte Carlo |
| `tanhsinh` | ✅ Implemented | Tanh-sinh quadrature |
| `lebedev_rule` | ✅ Implemented | Spherical quadrature |
| `newton_cotes` | ✅ Implemented | Rule coefficients |
| `nsum` | ✅ Implemented | Series summation |
| `quad_vec` | ✅ Implemented | Vectorized integration |
| `cubature` | ✅ Implemented | Adaptive multidimensional integration |

## Current Status

- [x] Set up module structure
- [x] Error handling
- [x] Numerical quadrature
  - [x] Basic methods (trapezoid rule, Simpson's rule)
  - [x] Adaptive quadrature for improved accuracy
  - [x] Gaussian quadrature
  - [x] Romberg integration
  - [x] Monte Carlo integration
  - [x] Multiple integration (for higher dimensions)
- [x] Ordinary differential equations (ODE)
  - [x] Euler method
  - [x] Runge-Kutta methods (RK4)
  - [x] Support for first-order ODE systems
- [x] Example code for all integration methods
- [x] Fix implementation issues
  - [x] Fix Gaussian quadrature node/weight calculation (current implementation had scaling issues)
  - [x] Improve Monte Carlo importance sampling stability
  - [x] Handle deep recursion issues better in Romberg integration
- [x] Enhance ODE solvers
  - [x] Variable step-size methods foundations (RK45, RK23)
  - [x] Explicit and implicit methods foundation (BDF implementation)
  - [x] Stiff equation solvers foundation (BDF)
  - [x] Boundary value problems
  - [x] Fix critical implementation issues with variable step and implicit methods:
    - [x] Fix coefficient calculations for RK23 
    - [x] Fix error estimation and step acceptance logic for RK45
    - [x] Revise BDF implementation with more stable numerical method for Newton iterations
    - [x] Add auto-differentiation or better numerical Jacobian calculation for BDF
    - [x] Enable currently ignored tests once fixed
  - [ ] Fix Radau method with mass matrices:
    - [ ] Newton iteration fails to converge when mass matrices are used
    - [ ] Issue appears in both standalone use and with event detection
    - [ ] Debug shows step size repeatedly decreasing due to Newton failures
- [x] Add utilities for numerical methods
  - [x] Numerical Jacobian calculation
  - [x] Linear system solver
  - [x] Newton method implementation

## SciPy Parity Methods Implementation

- [ ] Complete implementation of high-priority SciPy methods
  - [x] Additional ODE solvers
    - [x] DOP853 (8th order Runge-Kutta with 5th order error estimator)
    - [⚠️] Radau (implicit Runge-Kutta, L-stable) - fix mass matrix implementation
    - [x] LSODA (automatic Adams/BDF switching for non-stiff/stiff problems) - with enhanced stiffness detection
    - [x] Enhanced BDF with improved Jacobian handling and error estimation
  - [x] Quadrature rules generator
    - [x] Implement `newton_cotes` for generating quadrature weights and coefficients
    - [x] Support for open and closed forms of Newton-Cotes formulas
    - [x] Higher-order formulas with stability analysis
  - [x] Advanced quadrature methods
    - [x] tanh-sinh quadrature for improper integrals and singularities
    - [x] Lebedev quadrature for spherical integration
    - [x] Implement `cubature` with complete feature set for high-dimensional integration
    - [x] Add a dedicated one-dimensional interface for cubature
  - [x] Vectorized integration
    - [x] Implement `quad_vec` for array-valued integrands
    - [x] Efficient batch processing of multiple integrals
    - [x] Support for multiple threads during vector integration
  - [x] Series acceleration and summation
    - [x] Implement `nsum` for convergent series summation
    - [x] Sequence transformations (Euler, Richardson, etc.)
    - [x] Acceleration methods for slowly convergent series

- [x] Event detection in ODE solvers
  - [x] Root-finding for event functions during integration
  - [x] Integration termination based on events
  - [x] Precise event location with interpolation
  - [x] Support for state discontinuities at events
  - [x] Callback system for event handling

## Differential Algebraic Equation (DAE) Solvers

- [✅] Implement differential algebraic equation (DAE) solvers
  - [x] Index-1 DAE systems
    - [x] Semi-explicit DAE solver (form: x' = f(x,y,t), 0 = g(x,y,t))
    - [x] Support for mass matrices in ODE systems
      - [x] Basic framework for mass matrices (identity, constant, time-dependent)
      - [x] Transformation to standard form for explicit solvers
      - [x] Direct handling in implicit methods
      - [x] State-dependent mass matrices
    - [x] Implicit DAE solver for general index-1 systems (form: F(t,y,y') = 0)
  - [✅] Higher-index DAE systems with index reduction
    - [x] Pantelides algorithm for automatic index reduction
    - [x] Dummy derivative method for high-index DAEs
    - [x] Projection methods for constraint satisfaction
  - [✅] Implement BDF methods specifically for DAE systems
    - [x] Variable-order BDF methods for DAEs
    - [x] BDF methods for semi-explicit DAEs
    - [x] BDF methods for fully implicit DAEs
    - [x] Index reduction techniques for BDF methods
    - [x] Krylov-enhanced implicit solvers
      - [x] Matrix-free GMRES implementation
      - [x] Krylov methods for semi-explicit DAEs
      - [x] Krylov methods for fully implicit DAEs
      - [x] Simple diagonal preconditioners
    - [x] Block-structured preconditioners for large DAE systems
  - [✅] Comprehensive DAE examples
    - [x] Pendulum system (semi-explicit DAE)
    - [x] RLC circuit (fully implicit DAE)
    - [x] Slider-crank mechanism (higher-index DAE)
    - [x] Method comparison examples
  - [x] Documentation for DAE solvers
    - [x] Theory and implementation guide
    - [x] Event detection with DAEs
    - [x] Mass matrix handling with DAEs

## Partial Differential Equations (PDE) Support

- [⚠️] Add support for partial differential equations (PDE)
  - [⚠️] Finite difference methods
    - [x] 1D Cartesian grid implementations
    - [x] 2D Cartesian grid implementations
    - [x] 3D Cartesian grid implementations
    - [x] Explicit time-stepping schemes
    - [x] Implicit time-stepping schemes
      - [x] Crank-Nicolson method for parabolic PDEs
      - [x] Backward Euler method for stiff problems
      - [x] Alternating Direction Implicit (ADI) method for 2D problems
    - [ ] Support for irregular domains with ghost points
    - [ ] Adaptive mesh refinement (AMR) capabilities
  - [⚠️] Finite element methods
    - [x] Linear elements
    - [ ] Quadratic and cubic elements
    - [x] Basic Galerkin formulations for elliptic PDEs
    - [ ] Petrov-Galerkin formulations
    - [x] Support for triangular meshes
    - [x] Support for irregular domains
    - [ ] Automatic mesh generation interfaces
  - [⚠️] Method of lines for time-dependent PDEs
    - [x] Basic spatial discretization with central differences
    - [x] Higher-order finite difference schemes
    - [x] Integration with existing ODE solvers
    - [x] Support for parabolic PDEs (diffusion, advection-diffusion)
    - [x] Support for 1D parabolic PDEs
    - [x] Support for 2D parabolic PDEs
    - [x] Support for hyperbolic PDEs (wave equation)
    - [x] Support for elliptic PDEs (Poisson, Laplace)
    - [x] Support for stiff spatial operators
      - [x] Implicit handling of stiff advection-diffusion-reaction equations
      - [x] Efficient solvers for tridiagonal and block-tridiagonal systems
  - [⚠️] Spectral methods
    - [x] Fourier spectral methods for periodic domains
    - [x] Chebyshev methods for non-periodic domains
    - [x] Legendre methods for non-periodic domains
    - [x] Spectral element methods for complex geometries

## Error Handling and Convergence Improvements

- [ ] Improve error handling and convergence criteria
  - [⚠️] Better adaptive error control for ODE solvers
    - [x] PI step size controllers for smoother adaptation in enhanced methods
    - [x] Embedded error estimators for more methods, including BDF
    - [ ] Local extrapolation for higher accuracy
    - [x] Continuous output capability (dense output)
  - [⚠️] Smarter step size selection for stiff problems
    - [x] Automatic stiffness detection for LSODA and enhanced methods
    - [x] Method switching for problems with changing stiffness (LSODA)
    - [x] Error-based Jacobian update strategies in enhanced BDF
    - [ ] Analytical Jacobian support with symbolic differentiation
  - [⚠️] Enhanced convergence acceleration
    - [x] Improved nonlinear solvers for implicit methods
    - [ ] Anderson acceleration for fixed-point iterations
    - [ ] Multirate methods for systems with multiple timescales

## Documentation and Examples

- [ ] Add more examples and documentation
  - [ ] Tutorial for common integration problems
    - [ ] Step-by-step walkthrough for ODE, DAE, and BVP problems
    - [ ] Best practices for parameter selection
    - [ ] Error handling and troubleshooting guide
  - [ ] Examples for physical system modeling
    - [ ] Mechanical systems (pendulums, springs, etc.)
    - [ ] Electrical circuits with nonlinear components
    - [ ] Chemical reaction networks
    - [ ] Population dynamics and epidemiological models
  - [ ] Comparison with SciPy solutions
    - [ ] Direct comparison of algorithms and parameters
    - [ ] Performance benchmarks against SciPy
    - [ ] Adaptation guide for users transitioning from SciPy
  - [ ] Advanced usage patterns
    - [x] Custom event detection and handling
    - [ ] Complex boundary conditions
    - [x] Discontinuous systems and hybrid methods

## Boundary Value Problem Enhancements

- [ ] Additional boundary value problem features
  - [ ] Support for multipoint boundary value problems
    - [ ] Shooting methods for multipoint problems
    - [ ] Collocation with interior matching conditions
    - [ ] Automatic initial guess generation
  - [ ] More sophisticated mesh adaptation
    - [ ] Error-based mesh refinement
    - [ ] Adaptive order selection for collocation methods
    - [ ] Moving mesh methods for problems with sharp transitions
  - [ ] Support for Robin and mixed boundary conditions
    - [ ] General format: a*u + b*u' = c at boundaries
    - [ ] Interface for periodic boundary conditions
    - [ ] Special handling for singular boundary conditions

## Specialized Integration Methods

- [ ] Specialized integration methods for specific problem domains
  - [x] Hamiltonian and symplectic integrators
    - [x] Symplectic Euler and leapfrog methods
    - [x] Higher-order symplectic methods (e.g., Gauss collocation)
    - [x] Symmetric methods for reversible systems
  - [ ] Geometric integrators that preserve structure
    - [x] Methods that preserve energy, momentum, or other invariants
    - [ ] Lie group integrators for systems on manifolds
    - [ ] Volume-preserving methods for incompressible flows
  - [ ] Method-of-manufactured-solutions toolkit
    - [ ] Tools for verifying order of accuracy
    - [ ] Reference problem generation
  - [ ] Physics-informed numerical methods
    - [ ] Integration methods with physics constraints
    - [ ] Structure-preserving approximations

## Performance Optimizations

- [⚠️] Performance comparable to or better than SciPy's integrate
  - [⚠️] Optimize critical numerical routines
    - [x] Optimized linear solvers for implicit methods
      - [x] Specialized banded matrix solvers
      - [x] LU decomposition reuse for repeated solves
      - [x] Automatic solver selection based on matrix structure
    - [x] Cache-friendly algorithm implementations
    - [ ] Memory access pattern optimization
    - [ ] Expression template optimization for scalar operations
    - [ ] Custom allocators for numerical workspaces
    - [ ] Auto-tuning for different hardware configurations
  - [ ] Implement SIMD operations for key algorithms
    - [ ] Vectorized ODE function evaluation
    - [ ] SIMD-optimized linear algebra operations
    - [ ] Batch processing for multiple similar integration problems
    - [ ] AVX/AVX2/AVX-512 optimizations for x86 platforms
    - [ ] ARM NEON optimizations for mobile/embedded platforms
  - [ ] Profile-guided optimization for critical code paths
    - [ ] Identify and optimize hotspots in numerical routines
    - [ ] Specialized code paths for common integration scenarios
    - [ ] Loop nest optimization for numerical kernels

## Parallel and Distributed Computation

- [⚠️] Support for parallel and distributed computation
  - [⚠️] Parallel evaluation of function values
    - [x] Parallel Jacobian evaluation for large ODE systems
    - [x] Graph coloring for parallel sparse Jacobian computation
    - [ ] Thread-pool based parallelism for Monte Carlo integration
    - [ ] Work-stealing schedulers for adaptive algorithms
    - [ ] Concurrent evaluation of independent function calls
    - [ ] Add standard `workers` parameter to parallelizable functions
  - [ ] Domain decomposition for large-scale problems
    - [ ] Spatial domain partitioning for PDEs
    - [ ] Waveform relaxation methods for ODEs/DAEs
    - [ ] Parallel-in-time integration methods
  - [ ] Distributed memory parallelism
    - [ ] MPI integration for cluster computing
    - [ ] Hybrid MPI+OpenMP implementation
    - [ ] GPU offloading for compute-intensive operations
  - [ ] Asynchronous integration methods
    - [ ] Event-based progression for loosely coupled systems
    - [ ] Communication-avoiding algorithms

## Automatic Differentiation Integration

- [ ] Integration with automatic differentiation for gradient-based methods
  - [ ] True automatic differentiation for Jacobian calculation
    - [ ] Forward-mode AD for small number of parameters
    - [ ] Reverse-mode AD for large number of parameters
    - [ ] Mixed-mode AD for complex problems
    - [ ] Sparse AD for structured Jacobians
  - [ ] Better convergence in implicit methods
    - [ ] Exact Jacobian calculation for BDF and implicit Runge-Kutta methods
    - [ ] Automatic differentiation for sensitivity analysis
    - [ ] Integration with machine learning frameworks for neural ODEs
  - [ ] Adjoint methods for large-scale sensitivity analysis
    - [ ] Continuous adjoint methods for ODEs and DAEs
    - [ ] Checkpointing strategies for memory-efficient adjoints
    - [ ] Sensitivity calculation for optimal control problems

## Symbolic Integration Support

- [ ] Support for symbolic manipulation and simplification
  - [ ] Automatic conversion of higher-order ODEs to first-order systems
    - [ ] Symbolic differentiation and substitution
    - [ ] Structure preservation in conversion process
    - [ ] Special handling for common higher-order equations
  - [ ] Symbolic preprocessing of equations
    - [ ] Automatic detection of conservation laws
    - [ ] Equation simplification and canonical forms
    - [ ] Detection of symmetries and invariants
  - [ ] Code generation from symbolic representations
    - [ ] Fast specialized integrators for specific problem forms
    - [ ] Symbolic derivation of analytical Jacobians
    - [ ] Generation of problem-specific SIMD code

## Visualization and Analysis Tools

- [ ] Advanced visualization tools for solutions
  - [ ] Phase space plots for ODE systems
    - [ ] 2D and 3D phase portraits
    - [ ] Stability analysis visualization
    - [ ] Bifurcation diagrams and parameter sensitivity
  - [ ] Interactive solution exploration
    - [ ] Real-time parameter adjustment
    - [ ] Interactive continuation of solutions
    - [ ] Web-based visualization interfaces
  - [ ] Specialized visualization for specific problem types
    - [ ] Vector field visualization for dynamical systems
    - [ ] Solution manifold exploration for BVPs
    - [ ] Error and convergence visualization tools
  - [ ] Integration with common plotting libraries
    - [ ] Plotters, ndarray-stats, and plotly integration
    - [ ] Export to standard visualization formats

## Domain-Specific Integration Methods

- [ ] Domain-specific solvers for physics, engineering, and finance
  - [ ] Mechanical systems
    - [ ] Rigid body dynamics
    - [ ] Multibody systems with constraints
    - [ ] Deformable solids and structural dynamics
    - [ ] Contact and collision integration
  - [ ] Chemical kinetics
    - [ ] Stiff reaction systems
    - [ ] Chemical equilibrium calculations
    - [ ] Reaction-diffusion systems
    - [ ] Biochemical network simulation
  - [ ] Circuit simulation
    - [ ] Nonlinear circuit element models
    - [ ] Modified nodal analysis integration
    - [ ] Specialized methods for oscillators
    - [ ] Power system transient analysis
  - [ ] Option pricing and financial modeling
    - [ ] Stochastic differential equation solvers
    - [ ] American option early exercise boundary
    - [ ] Fixed-income derivative models
    - [ ] Risk modeling and uncertainty propagation
  - [ ] Fluid dynamics
    - [ ] Navier-Stokes solvers
    - [ ] Compressible and incompressible flows
    - [ ] Free-surface and multiphase flows
  - [ ] Quantum mechanics
    - [ ] Time-dependent Schrödinger equation
    - [ ] Density matrix evolution
    - [ ] Path integral methods

## Quasi-Monte Carlo Integration

- [ ] Implement Quasi-Monte Carlo integration methods
  - [ ] Implement `qmc_quad` function
  - [ ] Low-discrepancy sequence generators
    - [ ] Sobol sequences
    - [ ] Halton sequences
    - [ ] Faure sequences
  - [ ] Scrambling techniques for error estimation
  - [ ] Randomized quasi-Monte Carlo methods
  - [ ] Dimension reduction techniques for high-dimensional integrals