# scirs2-special TODO

This module provides special functions similar to SciPy's special module.

## Current Status

- [x] Set up module structure
- [x] Error handling
- [x] Bessel functions
  - [x] J₀, J₁, Jₙ (first kind)
  - [x] Y₀, Y₁, Yₙ (second kind)
  - [x] I₀, I₁, Iᵥ (modified, first kind)
  - [x] K₀, K₁, Kᵥ (modified, second kind)
- [x] Gamma and related functions
  - [x] Gamma function
  - [x] Log gamma function
  - [x] Digamma function
  - [x] Beta function
  - [x] Incomplete beta function
- [x] Error functions
  - [x] Error function (erf)
  - [x] Complementary error function (erfc)
  - [x] Inverse error function (erfinv)
  - [x] Inverse complementary error function (erfcinv)
- [x] Orthogonal polynomials
  - [x] Legendre polynomials
  - [x] Associated Legendre polynomials
  - [x] Laguerre polynomials
  - [x] Generalized Laguerre polynomials
  - [x] Hermite polynomials
  - [x] Chebyshev polynomials
  - [x] Gegenbauer polynomials
  - [x] Jacobi polynomials
- [x] Example for getting function values

## Future Tasks

- [x] Fix Clippy warning for needless_range_loop in orthogonal.rs
- [x] Add more special functions
  - [x] Airy functions
  - [x] Elliptic integrals and functions
  - [x] Hypergeometric functions
  - [x] Spherical harmonics
  - [x] Mathieu functions
  - [x] Zeta functions
  - [x] Kelvin functions
  - [x] Parabolic cylinder functions
  - [x] Lambert W function
  - [x] Struve functions
  - [x] Fresnel integrals
  - [x] Spheroidal wave functions
  - [x] Wright Omega function
  - [x] Coulomb functions
  - [x] Wright Bessel functions
  - [x] Logarithmic integral
- [ ] Enhance numerical precision
  - [ ] Improved algorithms for edge cases
  - [ ] Better handling of overflow and underflow
  - [ ] Extended precision options
  - [ ] Specialized routines for extreme parameter values
  - [ ] Use precomputed data for high-precision constants
- [ ] Optimize performance
  - [ ] Use more efficient algorithms for function evaluation
  - [ ] Precomputed coefficients and lookup tables where appropriate
  - [ ] Parallelization of array operations
  - [ ] SIMD optimizations for vector operations
  - [ ] Function-specific optimizations similar to SciPy's specialized implementations
- [x] Fix build issues
  - [x] Fix parameter mismatches in gamma function calls
  - [x] Remove unused variables and imports
  - [x] Fix function interface consistency issues
  - [x] Ensure proper typing throughout codebase
  - [x] Fix Clippy warnings in fresnel.rs (assign-op-pattern, needless-return)
- [ ] Add comprehensive testing infrastructure
  - [ ] Test data for validation against known values
  - [ ] Property-based testing for mathematical identities
  - [ ] Edge case testing with extreme parameter values
  - [ ] Regression tests for fixed numerical issues
  - [ ] Roundtrip testing where applicable

## Documentation and Examples

- [ ] Add more examples and documentation
  - [ ] Tutorial for common special function applications
  - [x] Logarithmic integral example
  - [x] Wright bessel functions example
  - [x] Spheroidal wave functions example
  - [x] Coulomb functions example 
  - [ ] Visual examples showing function behavior
  - [ ] Mathematical background for each function category
  - [ ] Numerical behavior documentation for edge cases
  - [ ] Performance characteristics and limitations
- [ ] Fix ignored doctests

## Array Support and Interoperability

- [ ] Enhance array operations
  - [ ] Support for multidimensional arrays
  - [ ] Vectorized operations for all functions
  - [ ] Lazy evaluation for large arrays
  - [ ] GPU acceleration for array operations
  - [ ] Support for array-like objects
- [ ] Implement alternative backends similar to SciPy's array API
  - [ ] Generalized interface for custom array types
  - [ ] Support for generic array operations
  - [ ] Feature flags for different array implementations

## Combinatorial Functions

- [ ] Add combinatorial functions
  - [ ] Binomial coefficients
  - [ ] Factorial and double factorial
  - [ ] Permutations and combinations
  - [ ] Stirling numbers
  - [ ] Bell numbers
  - [ ] Bernoulli numbers
  - [ ] Euler numbers

## Statistical Functions

- [ ] Add statistical convenience functions
  - [ ] Logistic function and its derivatives
  - [ ] Softmax and log-softmax functions
  - [ ] Log1p, expm1 (already in std but with array support)
  - [ ] LogSumExp for numerical stability
  - [ ] Normalized sinc function
  - [ ] Statistical distributions related functions

## Long-term Goals

- [ ] Performance comparable to or better than SciPy's special
- [ ] Integration with statistical and physics modules
- [ ] Support for arbitrary precision computation
- [ ] Comprehensive coverage of all SciPy special functions
- [ ] Advanced visualization tools for special functions
- [ ] Domain-specific packages for physics, engineering, and statistics
- [ ] Support for complex arguments in all functions
- [ ] Consistent API design for all function families
- [ ] Feature parity with SciPy's special module