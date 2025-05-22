# Adaptive Cubature Integration Example

This example demonstrates adaptive cubature integration using domain subdivision for multi-dimensional integration problems. This approach is particularly valuable for integrands with challenging features like peaks, ridges, or discontinuities.

## Key Techniques Demonstrated

### 1. Adaptive Domain Subdivision

The algorithm recursively divides the integration domain into subregions based on error estimates, focusing computational effort where the integrand is most challenging:

- Adaptive refinement in regions with high error estimates
- Dimension-based splitting strategy (longest dimension first)
- Hierarchical error estimation and convergence testing
- Automatic termination based on error tolerance or resource limits

### 2. Error Estimation Using Nested Quadrature Rules

The implementation uses two different order Gauss-Legendre quadrature rules to estimate the error:

- Low order rule (3 points) provides a baseline estimate
- High order rule (5 points) provides a more accurate estimate
- The difference between the two estimates serves as an error indicator
- This provides automatic error control without additional function evaluations

### 3. Tensor Product Extension to Multiple Dimensions

The example demonstrates how to extend one-dimensional quadrature rules to handle multiple dimensions:

- Recursive application of 1D rules to each dimension
- Tensor product construction
- Coordinate transformation to map from reference domain to integration region
- Scaling to account for domain volumes

## Test Functions

The example includes several challenging test functions to demonstrate the adaptivity:

1. **Sharp Peak Function**: A Gaussian peak in the center of the domain that tests the ability to concentrate points around regions of high variation.

2. **Multiple Peaks Function**: A function with two separated peaks that tests the ability to identify and focus on multiple important regions.

3. **Ridge Function**: A function with a sharp ridge along a diagonal, which is challenging for adaptive methods that use axis-aligned subdivision.

4. **Discontinuous Function**: A function with a discontinuity, testing the algorithm's ability to detect and handle discontinuities.

## Implementation Details

The implementation consists of several key components:

1. **SubRegion Management**: 
   - Regions are represented by their boundaries and error estimates
   - A priority queue ensures regions with highest estimated error are subdivided first
   - Depth tracking prevents excessive recursion

2. **Integration Algorithm**:
   - Tensor product Gauss-Legendre quadrature for each subregion
   - Recursive implementation to handle arbitrary dimensions
   - Error estimation using difference between high and low order rules

3. **Adaptive Refinement Strategy**:
   - Regions are split along their longest dimension
   - Error-based subdivision prioritizes regions with larger error
   - Both absolute and relative error criteria supported

## Results and Performance

The example provides detailed output for each test function, including:

- Number of function evaluations
- Number of subregions created
- Maximum recursion depth
- Estimated integral value
- Estimated error
- Actual error (when reference value is available)
- Execution time

The results demonstrate that adaptive cubature effectively:
- Concentrates computation in regions with challenging features
- Handles a variety of difficult integrands
- Provides reliable error estimates
- Uses computational resources efficiently

## Comparison to Other Methods

The adaptive cubature approach has several advantages over other integration methods:

- More efficient than uniform grid methods for functions with local features
- More robust than Monte Carlo for functions with sharp peaks or discontinuities
- Generally more reliable error estimates than fixed-rule quadrature
- Better scaling with dimensionality than fixed grid methods

However, it also has limitations:
- Performance degrades in higher dimensions (curse of dimensionality)
- Axis-aligned subdivision can be inefficient for diagonal ridges
- Computational overhead increases with number of subregions

## When to Use Adaptive Cubature

Adaptive cubature is most appropriate for:
1. Low to moderate dimensional problems (typically 2-6 dimensions)
2. Functions with localized features (peaks, discontinuities)
3. When accurate error estimates are required
4. When the function evaluation is expensive

For very high-dimensional problems, Monte Carlo or Quasi-Monte Carlo methods are generally more appropriate.