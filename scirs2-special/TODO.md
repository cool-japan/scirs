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
- [x] Enhance numerical precision
  - [x] Improved algorithms for edge cases
  - [x] Better handling of overflow and underflow
  - [x] Extended precision options
  - [x] Specialized routines for extreme parameter values
  - [x] Use precomputed data for high-precision constants
- [x] Optimize performance
  - [x] Use more efficient algorithms for function evaluation
  - [x] Precomputed coefficients and lookup tables where appropriate
  - [x] Parallelization of array operations
  - [x] SIMD optimizations for vector operations
  - [x] Function-specific optimizations similar to SciPy's specialized implementations
- [x] Fix build issues
  - [x] Fix parameter mismatches in gamma function calls
  - [x] Remove unused variables and imports
  - [x] Fix function interface consistency issues
  - [x] Ensure proper typing throughout codebase
  - [x] Fix Clippy warnings in fresnel.rs (assign-op-pattern, needless-return)
- [x] Add comprehensive testing infrastructure
  - [x] Test data for validation against known values
  - [x] Property-based testing for mathematical identities
  - [x] Edge case testing with extreme parameter values
  - [x] Regression tests for fixed numerical issues
  - [x] Roundtrip testing where applicable

## Documentation and Examples

- [x] Add more examples and documentation
  - [x] Tutorial for common special function applications
  - [x] Logarithmic integral example
  - [x] Wright bessel functions example
  - [x] Spheroidal wave functions example
  - [x] Coulomb functions example 
  - [x] Comprehensive usage examples
  - [x] Advanced usage patterns and optimization examples
  - [x] Mathematical properties demonstration
  - [x] Performance characteristics examples
- [x] Fix ignored doctests

## Array Support and Interoperability

- [x] Enhance array operations
  - [x] Support for multidimensional arrays
  - [x] Vectorized operations for all functions
  - [x] Lazy evaluation for large arrays
  - [x] GPU acceleration for array operations
  - [x] Support for array-like objects
- [x] Implement alternative backends similar to SciPy's array API
  - [x] Generalized interface for custom array types
  - [x] Support for generic array operations
  - [x] Feature flags for different array implementations

## Combinatorial Functions

- [x] Add combinatorial functions
  - [x] Binomial coefficients
  - [x] Factorial and double factorial
  - [x] Permutations and combinations
  - [x] Stirling numbers
  - [x] Bell numbers
  - [x] Bernoulli numbers
  - [x] Euler numbers

## Statistical Functions

- [x] Add statistical convenience functions
  - [x] Logistic function and its derivatives
  - [x] Softmax and log-softmax functions
  - [x] Log1p, expm1 (already in std but with array support)
  - [x] LogSumExp for numerical stability
  - [x] Normalized sinc function
  - [x] Statistical distributions related functions

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