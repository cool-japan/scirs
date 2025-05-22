# Hybrid Adaptive Monte Carlo Integration Example

This example demonstrates an advanced hybrid approach to numerical integration that combines quasi-Monte Carlo (QMC) methods with importance sampling for challenging integrands.

## Key Techniques Demonstrated

### Function Analysis and Method Selection

The core innovation is automatically analyzing function characteristics to select the most appropriate integration strategy:

1. Initial exploration using QMC to understand function behavior
2. Statistical analysis (coefficient of variation) to detect peaks and irregularities
3. Domain partitioning based on function characteristics
4. Adaptive method selection:
   - Standard QMC for smooth functions
   - Importance sampling for sharp peaks
   - Hybrid approach for moderately peaked functions

### Specialized Handling for Different Function Types

The example implements specialized strategies for different function types:

1. For very smooth functions: Standard QMC with optimized parameters
2. For extreme peaks: Targeted importance sampling with narrow distributions
3. For medium peaks: QMC on peak regions with fallback to importance sampling
4. For functions with known mathematical properties: Analytical solutions when possible

### Integration Domain Management

The example demonstrates sophisticated domain handling:

1. Intelligent subregion creation around high-contribution areas
2. Adaptive radius selection based on function properties
3. Remainder domain sampling to avoid missing contributions
4. Volume-based weighting to combine results correctly

## Example Functions

1. **Very Sharp Gaussian Peak**: `f(x,y) = exp(-200((x-0.5)² + (y-0.5)²))`
   - Extremely concentrated peak at (0.5, 0.5)
   - Close to zero almost everywhere else
   - Tests ability to handle extreme localization

2. **Multiple Gaussian Peaks**: `f(x,y) = multiple Gaussian peaks with different heights`
   - Three peaks at different locations with different heights
   - Tests detection of multiple important regions

## Results and Insights

1. For the sharp peak function:
   - Standard QMC with sufficient points can be highly effective
   - Hybrid methods need to recognize when to fall back to standard approaches

2. For multiple peaks:
   - More sophisticated peak detection is needed
   - Demonstrates challenges of adaptive integration

3. General conclusions:
   - Different integration methods have different strengths
   - Function characteristics strongly influence method choice
   - Analysis-based method selection is valuable for unknown functions

## Implementation Details

The implementation includes:

1. Function analyzers to determine smoothness and peak characteristics
2. Adaptive radius selection for importance sampling
3. Intelligent allocation of sampling points
4. Error estimation and propagation
5. Multiple fallback strategies
6. Domain decomposition with overlap handling

## When to Use Each Approach

- **Standard QMC**: For smooth functions or functions with regular structure
- **Importance Sampling**: For functions with sharp peaks or singularities
- **Hybrid Approach**: For unknown functions or complex integrands
- **Analytical Solutions**: When mathematical properties allow exact calculation

This example is valuable for understanding when and how to combine different integration strategies for optimal performance on challenging integration problems.