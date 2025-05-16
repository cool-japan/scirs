# Integration Examples for scirs2-integrate

This directory contains examples demonstrating the various integration methods available in the `scirs2-integrate` crate. These examples range from basic usage to advanced techniques for handling challenging integration problems.

## Basic Examples

- **[basic_integration.rs](basic_integration.rs)**: Demonstrates simple numerical integration using the basic quadrature methods.
- **[gaussian_quadrature.rs](gaussian_quadrature.rs)**: Shows how to use Gauss-Legendre quadrature for polynomial-like functions.
- **[romberg_integration.rs](romberg_integration.rs)**: Illustrates Romberg integration, which uses Richardson extrapolation to improve accuracy.

## ODE Solvers

- **[ode_solver.rs](ode_solver.rs)**: Basic example of solving ordinary differential equations.
- **[dop853_example.rs](dop853_example.rs)**: Demonstrates the Dormand-Prince 8(5,3) method for high-accuracy ODE integration.
- **[radau_example.rs](radau_example.rs)**: Shows the Radau method for stiff ODEs.
- **[lsoda_example.rs](lsoda_example.rs)**: Illustrates LSODA, which automatically switches between methods based on problem stiffness.
- **[lsoda_method_switching.rs](lsoda_method_switching.rs)**: Focuses on LSODA's method switching capabilities.
- **[ode_improved_bdf.rs](ode_improved_bdf.rs)**: Demonstrates improved backward differentiation formulas for stiff problems.

## Multi-dimensional Integration

- **[multidimensional_integration.rs](multidimensional_integration.rs)**: Shows basic techniques for integrating functions of multiple variables.
- **[cubature_example.rs](cubature_example.rs)**: Demonstrates cubature methods for multidimensional integration.
- **[newton_cotes_example.rs](newton_cotes_example.rs)**: Shows Newton-Cotes formulas for numerical integration.
- **[lebedev_example.rs](lebedev_example.rs)**: Demonstrates Lebedev quadrature for functions on the sphere.

## Advanced Techniques

- **[adaptive_monte_carlo.rs](adaptive_monte_carlo.rs)**: Demonstrates adaptive Monte Carlo techniques for challenging integrands.
  - [Documentation](adaptive_monte_carlo.md): Detailed explanation of the adaptive Monte Carlo approach.

- **[adaptive_hybrid_monte_carlo.rs](adaptive_hybrid_monte_carlo.rs)**: Shows a hybrid approach combining QMC with importance sampling.
  - [Documentation](adaptive_hybrid_monte_carlo.md): Explanation of the hybrid approach advantages and limitations.

- **[adaptive_cubature.rs](adaptive_cubature.rs)**: Implements adaptive domain subdivision for multi-dimensional integration.
  - [Documentation](adaptive_cubature.md): Details of the adaptive cubature algorithm and its applications.

- **[monte_carlo_integration.rs](monte_carlo_integration.rs)**: Basic Monte Carlo integration with variance reduction techniques.
- **[qmc_example.rs](qmc_example.rs)**: Quasi-Monte Carlo integration with low-discrepancy sequences.
- **[tanhsinh_example.rs](tanhsinh_example.rs)**: Double exponential quadrature for handling endpoint singularities.

## Method Comparisons

- **[method_comparison.rs](method_comparison.rs)**: Compares different integration methods on the same test functions.
- **[combined_methods.rs](combined_methods.rs)**: Shows how to combine different integration techniques.
- **[lsoda_comparison.rs](lsoda_comparison.rs)**: Compares LSODA with other ODE solvers.

## Specialized Applications

- **[bvp_example.rs](bvp_example.rs)**: Boundary value problems for ordinary differential equations.

## Performance Considerations

The examples illustrate various performance characteristics:

1. **Accuracy vs. Efficiency**: Different methods provide tradeoffs between accuracy and computational cost. Higher-order methods generally provide better accuracy per function evaluation but have higher overhead.

2. **Problem-Specific Methods**: Some methods are specialized for particular types of problems:
   - Tanh-sinh for endpoint singularities
   - Adaptive cubature for functions with localized features
   - Monte Carlo for high-dimensional integrals
   - LSODA for mixed stiff/non-stiff ODEs

3. **Adaptive Strategies**: Many examples demonstrate adaptive techniques that concentrate computational effort where it's most needed:
   - Error-based subdivision in adaptive cubature
   - Importance sampling in Monte Carlo
   - Adaptive step size control in ODE solvers

## Running the Examples

To run an example, use:

```bash
cargo run --example <example_name>
```

For instance:

```bash
cargo run --example adaptive_monte_carlo
```

## Error Handling

Most examples demonstrate proper error handling using the `IntegrateResult` type. This provides a consistent way to handle potential errors during integration:

- Out-of-bounds errors
- Convergence failures
- Excessive function evaluations
- Invalid input parameters

## Extension Points

These examples can serve as starting points for your own integration tasks. Consider extending them by:

1. Implementing custom integrand functions
2. Combining multiple integration techniques
3. Adding visualization of results
4. Developing problem-specific error estimation
5. Creating adaptive strategies for specific domains