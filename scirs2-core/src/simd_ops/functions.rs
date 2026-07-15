//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use ::ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1};
use num_traits::Zero;

/// Unified SIMD operations trait
pub trait SimdUnifiedOps: Sized + Copy + PartialOrd + Zero {
    /// Element-wise addition
    fn simd_add(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise subtraction
    fn simd_sub(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise multiplication
    fn simd_mul(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise division
    fn simd_div(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self>;
    /// Dot product
    fn simd_dot(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self;
    /// Matrix-vector multiplication (GEMV)
    fn simd_gemv(a: &ArrayView2<Self>, x: &ArrayView1<Self>, beta: Self, y: &mut Array1<Self>);
    /// Matrix-matrix multiplication (GEMM)
    fn simd_gemm(
        alpha: Self,
        a: &ArrayView2<Self>,
        b: &ArrayView2<Self>,
        beta: Self,
        c: &mut Array2<Self>,
    );
    /// Vector norm (L2)
    fn simd_norm(a: &ArrayView1<Self>) -> Self;
    /// Element-wise maximum
    fn simd_max(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise minimum
    fn simd_min(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self>;
    /// Scalar multiplication
    fn simd_scalar_mul(a: &ArrayView1<Self>, scalar: Self) -> Array1<Self>;
    /// Sum reduction
    fn simd_sum(a: &ArrayView1<Self>) -> Self;
    /// Mean reduction
    fn simd_mean(a: &ArrayView1<Self>) -> Self;
    /// Find maximum element
    fn simd_max_element(a: &ArrayView1<Self>) -> Self;
    /// Find minimum element
    fn simd_min_element(a: &ArrayView1<Self>) -> Self;
    /// Fused multiply-add: a * b + c
    fn simd_fma(a: &ArrayView1<Self>, b: &ArrayView1<Self>, c: &ArrayView1<Self>) -> Array1<Self>;
    /// Enhanced cache-optimized addition for large arrays
    fn simd_add_cache_optimized(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self>;
    /// Advanced-optimized fused multiply-add for maximum performance
    fn simd_fma_advanced_optimized(
        a: &ArrayView1<Self>,
        b: &ArrayView1<Self>,
        c: &ArrayView1<Self>,
    ) -> Array1<Self>;
    /// Adaptive SIMD operation that selects optimal implementation
    fn simd_add_adaptive(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self>;
    /// Matrix transpose
    fn simd_transpose(a: &ArrayView2<Self>) -> Array2<Self>;
    /// Cache-optimized blocked matrix transpose
    /// Uses L1 cache-friendly block sizes for improved memory access patterns.
    /// Expected 3-5x speedup for large matrices (>512x512).
    fn simd_transpose_blocked(a: &ArrayView2<Self>) -> Array2<Self>;
    /// Element-wise absolute value
    fn simd_abs(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise square root
    fn simd_sqrt(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise exponential (e^x)
    fn simd_exp(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise natural logarithm (ln(x))
    fn simd_ln(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise sine (sin(x))
    fn simd_sin(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise cosine (cos(x))
    fn simd_cos(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise tangent (tan(x))
    fn simd_tan(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise hyperbolic sine (sinh(x))
    fn simd_sinh(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise hyperbolic cosine (cosh(x))
    fn simd_cosh(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise hyperbolic tangent (tanh(x))
    fn simd_tanh(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise floor (largest integer <= x)
    fn simd_floor(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise ceiling (smallest integer >= x)
    fn simd_ceil(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise rounding to nearest integer
    fn simd_round(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise arctangent (atan(x))
    fn simd_atan(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise arcsine (asin(x))
    fn simd_asin(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise arccosine (acos(x))
    fn simd_acos(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise two-argument arctangent (atan2(y, x))
    fn simd_atan2(y: &ArrayView1<Self>, x: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise base-10 logarithm (log10(x))
    fn simd_log10(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise base-2 logarithm (log2(x))
    fn simd_log2(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise clamp (constrain values to [min, max])
    fn simd_clamp(a: &ArrayView1<Self>, min: Self, max: Self) -> Array1<Self>;
    /// Element-wise fractional part (x - trunc(x))
    fn simd_fract(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise trunc (round toward zero)
    fn simd_trunc(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise reciprocal (1/x)
    fn simd_recip(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise power with scalar exponent (base^exp)
    fn simd_powf(base: &ArrayView1<Self>, exp: Self) -> Array1<Self>;
    /// Element-wise power with array exponent (`base[i]^exp[i]`)
    fn simd_pow(base: &ArrayView1<Self>, exp: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise power with integer exponent (base^n)
    fn simd_powi(base: &ArrayView1<Self>, n: i32) -> Array1<Self>;
    /// Element-wise gamma function Γ(x)
    fn simd_gamma(x: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise 2^x (base-2 exponential)
    fn simd_exp2(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise cube root (cbrt(x) = x^(1/3))
    fn simd_cbrt(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise ln(1+x) (numerically stable for small x)
    fn simd_ln_1p(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise exp(x)-1 (numerically stable for small x)
    fn simd_exp_m1(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise conversion from degrees to radians (x * π / 180)
    fn simd_to_radians(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise conversion from radians to degrees (x * 180 / π)
    fn simd_to_degrees(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise digamma function ψ(x) = d/dx ln(Γ(x))
    fn simd_digamma(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise trigamma function ψ'(x) = d²/dx² ln(Γ(x))
    /// The second derivative of log-gamma, critical for Fisher information in Bayesian inference.
    fn simd_trigamma(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise log-gamma function ln(Γ(x))
    /// More numerically stable than computing gamma(x).ln() - used extensively in statistical distributions.
    fn simd_ln_gamma(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise error function erf(x) = (2/√π) ∫₀ˣ e^(-t²) dt
    /// Critical for normal distribution CDF: Φ(x) = 0.5 * (1 + erf(x/√2))
    /// Properties: erf(0)=0, erf(∞)=1, erf(-x)=-erf(x)
    fn simd_erf(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise complementary error function erfc(x) = 1 - erf(x)
    /// More numerically stable than 1 - erf(x) for large x
    fn simd_erfc(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise inverse error function erfinv(y) such that erf(erfinv(y)) = y
    /// Critical for inverse normal CDF (probit function): Φ⁻¹(p) = √2 * erfinv(2p - 1)
    /// Domain: (-1, 1), Range: (-∞, ∞)
    /// Properties: erfinv(0)=0, erfinv(-y)=-erfinv(y) (odd function)
    fn simd_erfinv(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise inverse complementary error function erfcinv(y) such that erfc(erfcinv(y)) = y
    /// More numerically stable than erfinv(1-y) for y close to 0
    /// Domain: (0, 2), Range: (-∞, ∞)
    fn simd_erfcinv(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise sigmoid (logistic) function: σ(x) = 1 / (1 + exp(-x))
    /// Critical for neural networks, logistic regression, and probability modeling
    /// Range: (0, 1), σ(0) = 0.5, σ(-∞) = 0, σ(+∞) = 1
    /// Properties: σ(-x) = 1 - σ(x), derivative σ'(x) = σ(x)(1 - σ(x))
    fn simd_sigmoid(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise GELU (Gaussian Error Linear Unit) activation function
    /// GELU(x) = x * Φ(x) = x * 0.5 * (1 + erf(x / √2))
    /// Where Φ(x) is the standard normal CDF
    /// Critical for Transformer models (BERT, GPT, etc.)
    /// Properties: GELU(0) = 0, smooth approximation of ReLU
    fn simd_gelu(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise Swish (SiLU - Sigmoid Linear Unit) activation function
    /// Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
    /// Self-gated activation discovered via neural architecture search
    /// Used in EfficientNet, GPT-NeoX, and many modern architectures
    /// Properties: smooth, non-monotonic, self-gating, unbounded above
    fn simd_swish(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise Softplus activation function
    /// Softplus(x) = ln(1 + exp(x))
    /// Smooth approximation of ReLU
    /// Used in probabilistic models, Bayesian deep learning, smooth counting
    /// Properties: softplus(0) = ln(2) ≈ 0.693, always positive, derivative = sigmoid(x)
    fn simd_softplus(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise Mish activation function
    /// Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    /// Self-regularized non-monotonic activation function
    /// Used in YOLOv4, modern object detection, and neural architectures
    /// Properties: smooth, non-monotonic, Mish(0) = 0, unbounded above
    fn simd_mish(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise ELU (Exponential Linear Unit) activation function
    /// ELU(x, α) = x if x >= 0, α * (exp(x) - 1) if x < 0
    /// Helps with vanishing gradients and faster learning
    /// Used in deep neural networks for smoother outputs
    /// Properties: smooth, continuous derivative, bounded below by -α
    fn simd_elu(a: &ArrayView1<Self>, alpha: Self) -> Array1<Self>;
    /// SELU activation function (Scaled Exponential Linear Unit)
    ///
    /// SELU(x) = λ * (x if x > 0, α * (exp(x) - 1) if x <= 0)
    /// where λ ≈ 1.0507 and α ≈ 1.6733 (fixed constants)
    /// Self-normalizing: preserves mean=0, variance=1 through layers
    /// Used in Self-Normalizing Neural Networks (SNNs)
    /// Eliminates need for BatchNorm when using LeCun Normal initialization
    fn simd_selu(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Hardsigmoid activation function
    ///
    /// Hardsigmoid(x) = clip((x + 3) / 6, 0, 1)
    /// Piecewise linear approximation of sigmoid
    /// Used in MobileNetV3 for efficient inference
    /// Avoids expensive exp() computation
    fn simd_hardsigmoid(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Hardswish activation function
    ///
    /// Hardswish(x) = x * hardsigmoid(x) = x * clip((x + 3) / 6, 0, 1)
    /// Piecewise linear approximation of Swish
    /// Used in MobileNetV3 for efficient inference
    /// Avoids expensive exp() computation while maintaining self-gating
    fn simd_hardswish(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Sinc function (normalized)
    ///
    /// sinc(x) = sin(πx) / (πx) for x ≠ 0, sinc(0) = 1
    /// Critical for signal processing, windowing, interpolation
    /// Properties: sinc(n) = 0 for all non-zero integers n
    fn simd_sinc(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Log-softmax function for numerically stable probability computation
    ///
    /// log_softmax(x_i) = x_i - log(Σ_j exp(x_j))
    /// Critical for neural networks, especially cross-entropy loss
    /// More numerically stable than computing log(softmax(x))
    /// Used in Transformers, LLMs, and classification networks
    fn simd_log_softmax(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Inverse hyperbolic sine: asinh(x) = ln(x + √(x² + 1))
    ///
    /// Domain: (-∞, +∞), Range: (-∞, +∞)
    /// Used in: hyperbolic geometry, conformal mapping, special relativity (rapidity)
    fn simd_asinh(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Inverse hyperbolic cosine: acosh(x) = ln(x + √(x² - 1))
    ///
    /// Domain: [1, +∞), Range: [0, +∞)
    /// Returns NaN for x < 1
    /// Used in: hyperbolic geometry, distance calculations, special relativity
    fn simd_acosh(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Inverse hyperbolic tangent: atanh(x) = 0.5 * ln((1+x)/(1-x))
    ///
    /// Domain: (-1, 1), Range: (-∞, +∞)
    /// Returns ±∞ at x = ±1, NaN for |x| > 1
    /// Used in: statistical transformations (Fisher's z), probability
    fn simd_atanh(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Beta function: B(a, b) = Γ(a)Γ(b)/Γ(a+b)
    ///
    /// The beta function is fundamental for:
    /// - Beta distribution (Bayesian priors)
    /// - Binomial coefficients: C(n,k) = 1/(n+1)/B(n-k+1, k+1)
    /// - Statistical hypothesis testing
    /// - Incomplete beta function (regularized)
    fn simd_beta(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self>;
    /// Log-Beta function: ln(B(a, b)) = ln(Γ(a)) + ln(Γ(b)) - ln(Γ(a+b))
    ///
    /// More numerically stable than computing B(a,b) for large arguments.
    /// Returns ln(B(a,b)) for each pair of inputs.
    fn simd_ln_beta(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self>;
    /// Linear interpolation: lerp(a, b, t) = a + t * (b - a) = a * (1 - t) + b * t
    ///
    /// Computes element-wise linear interpolation between arrays `a` and `b`
    /// using interpolation parameter `t`. When t=0, returns a; when t=1, returns b.
    ///
    /// Critical for:
    /// - Animation blending (skeletal animation, morph targets)
    /// - Quaternion SLERP approximation (for small angles)
    /// - Gradient computation in neural networks
    /// - Smooth parameter transitions
    /// - Color blending and image processing
    fn simd_lerp(a: &ArrayView1<Self>, b: &ArrayView1<Self>, t: Self) -> Array1<Self>;
    /// Smoothstep interpolation: smoothstep(edge0, edge1, x)
    ///
    /// Returns smooth Hermite interpolation between 0 and 1 when edge0 < x < edge1.
    /// - Returns 0 if x <= edge0
    /// - Returns 1 if x >= edge1
    /// - Returns smooth curve: 3t² - 2t³ where t = (x - edge0) / (edge1 - edge0)
    ///
    /// Critical for:
    /// - Shader programming (lighting, transitions)
    /// - Activation function variants
    /// - Smooth threshold functions
    /// - Anti-aliasing and blending
    fn simd_smoothstep(edge0: Self, edge1: Self, x: &ArrayView1<Self>) -> Array1<Self>;
    /// Hypotenuse: hypot(x, y) = sqrt(x² + y²)
    ///
    /// Computes element-wise hypotenuse without overflow/underflow issues.
    /// Uses the standard library implementation which handles extreme values.
    ///
    /// Critical for:
    /// - Distance calculations in 2D/3D
    /// - Computing vector magnitudes
    /// - Graphics and physics simulations
    /// - Complex number modulus: |a+bi| = hypot(a, b)
    fn simd_hypot(x: &ArrayView1<Self>, y: &ArrayView1<Self>) -> Array1<Self>;
    /// Copysign: copysign(x, y) returns x with the sign of y
    ///
    /// For each element, returns the magnitude of x with the sign of y.
    /// - copysign(1.0, -2.0) = -1.0
    /// - copysign(-3.0, 4.0) = 3.0
    ///
    /// Critical for:
    /// - Sign manipulation in numerical algorithms
    /// - Implementing special functions (e.g., reflection formula)
    /// - Gradient sign propagation
    fn simd_copysign(x: &ArrayView1<Self>, y: &ArrayView1<Self>) -> Array1<Self>;
    /// Smootherstep (Ken Perlin's improved smoothstep): 6t⁵ - 15t⁴ + 10t³
    ///
    /// An improved version of smoothstep with second-order continuous derivatives.
    /// The first AND second derivatives are zero at the boundaries.
    ///
    /// Critical for:
    /// - Perlin noise and procedural generation
    /// - High-quality animation easing
    /// - Shader programming (better lighting transitions)
    /// - Gradient-based optimization (smoother loss landscapes)
    fn simd_smootherstep(edge0: Self, edge1: Self, x: &ArrayView1<Self>) -> Array1<Self>;
    /// Logaddexp: log(exp(a) + exp(b)) computed in a numerically stable way
    ///
    /// Uses the identity: log(exp(a) + exp(b)) = max(a,b) + log(1 + exp(-|a-b|))
    /// This avoids overflow/underflow for large positive or negative values.
    ///
    /// Critical for:
    /// - Log-probability computations (Bayesian inference)
    /// - Log-likelihood calculations in ML
    /// - Hidden Markov Model forward/backward algorithms
    /// - Neural network loss functions (cross-entropy)
    fn simd_logaddexp(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self>;
    /// Logit function: log(p / (1-p)) - inverse of sigmoid
    ///
    /// Maps probabilities in (0, 1) to log-odds in (-∞, +∞).
    /// The logit function is the inverse of the sigmoid (logistic) function.
    ///
    /// Critical for:
    /// - Logistic regression (log-odds interpretation)
    /// - Probability calibration
    /// - Converting probabilities to unbounded space for optimization
    /// - Statistical modeling (link functions)
    fn simd_logit(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Element-wise square: x²
    ///
    /// More efficient than simd_pow(x, 2) or simd_mul(x, x) as it's a single multiplication.
    ///
    /// Critical for:
    /// - Variance computation: E\[X²\] - E\[X\]²
    /// - Distance calculations: ||a - b||² = (a - b)²
    /// - Neural network loss functions (MSE)
    /// - Physics simulations (kinetic energy: ½mv²)
    fn simd_square(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Inverse square root: 1/sqrt(x)
    ///
    /// More efficient than simd_div(1, simd_sqrt(x)) for normalization operations.
    ///
    /// Critical for:
    /// - Vector normalization: v * rsqrt(dot(v,v))
    /// - Graphics (lighting, physics simulations)
    /// - Layer normalization in neural networks
    /// - Quaternion normalization
    fn simd_rsqrt(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Simultaneous sin and cos: returns (sin(x), cos(x))
    ///
    /// More efficient than calling sin and cos separately when both are needed.
    /// Returns a tuple of two arrays.
    ///
    /// Critical for:
    /// - Rotation matrices (2D and 3D)
    /// - Fourier transforms
    /// - Wave simulations
    /// - Animation and physics
    fn simd_sincos(a: &ArrayView1<Self>) -> (Array1<Self>, Array1<Self>);
    /// Numerically stable exp(x) - 1
    ///
    /// Returns exp(x) - 1 accurately for small x values where exp(x) ≈ 1.
    /// For small x, the direct calculation exp(x) - 1 suffers from catastrophic cancellation.
    ///
    /// Critical for:
    /// - Financial calculations (compound interest for small rates)
    /// - Numerical integration of differential equations
    /// - Statistical distributions (Poisson, exponential)
    /// - Machine learning (softplus, log-sum-exp)
    fn simd_expm1(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Numerically stable ln(1 + x)
    ///
    /// Returns ln(1 + x) accurately for small x values where 1 + x ≈ 1.
    /// For small x, the direct calculation ln(1 + x) suffers from catastrophic cancellation.
    ///
    /// Critical for:
    /// - Log-probability calculations (log(1 - p) for small p)
    /// - Numerical integration
    /// - Statistical distributions
    /// - Machine learning (binary cross-entropy loss)
    fn simd_log1p(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Sum of squares
    fn simd_sum_squares(a: &ArrayView1<Self>) -> Self;
    /// Element-wise multiplication (alias for simd_mul)
    fn simd_multiply(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self>;
    /// Check if SIMD is available for this type
    fn simd_available() -> bool;
    /// Ultra-optimized sum reduction (alias for simd_sum for compatibility)
    fn simd_sum_f32_ultra(a: &ArrayView1<Self>) -> Self {
        Self::simd_sum(a)
    }
    /// Ultra-optimized subtraction (alias for simd_sub for compatibility)
    fn simd_sub_f32_ultra(
        a: &ArrayView1<Self>,
        b: &ArrayView1<Self>,
        result: &mut ArrayViewMut1<Self>,
    );
    /// Ultra-optimized multiplication (alias for simd_mul for compatibility)
    fn simd_mul_f32_ultra(
        a: &ArrayView1<Self>,
        b: &ArrayView1<Self>,
        result: &mut ArrayViewMut1<Self>,
    );
    /// Ultra-optimized cubes sum (power 3 sum)
    fn simd_sum_cubes(a: &ArrayView1<Self>) -> Self;
    /// Ultra-optimized division (alias for simd_div for compatibility)
    fn simd_div_f32_ultra(
        a: &ArrayView1<Self>,
        b: &ArrayView1<Self>,
        result: &mut ArrayViewMut1<Self>,
    );
    /// Ultra-optimized sine function
    fn simd_sin_f32_ultra(a: &ArrayView1<Self>, result: &mut ArrayViewMut1<Self>);
    /// Ultra-optimized addition (alias for simd_add for compatibility)
    fn simd_add_f32_ultra(
        a: &ArrayView1<Self>,
        b: &ArrayView1<Self>,
        result: &mut ArrayViewMut1<Self>,
    );
    /// Ultra-optimized fused multiply-add
    fn simd_fma_f32_ultra(
        a: &ArrayView1<Self>,
        b: &ArrayView1<Self>,
        c: &ArrayView1<Self>,
        result: &mut ArrayViewMut1<Self>,
    );
    /// Ultra-optimized power function
    fn simd_pow_f32_ultra(
        a: &ArrayView1<Self>,
        b: &ArrayView1<Self>,
        result: &mut ArrayViewMut1<Self>,
    );
    /// Ultra-optimized exponential function
    fn simd_exp_f32_ultra(a: &ArrayView1<Self>, result: &mut ArrayViewMut1<Self>);
    /// Ultra-optimized cosine function
    fn simd_cos_f32_ultra(a: &ArrayView1<Self>, result: &mut ArrayViewMut1<Self>);
    /// Ultra-optimized dot product
    fn simd_dot_f32_ultra(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self;
    /// Variance (population variance)
    fn simd_variance(a: &ArrayView1<Self>) -> Self;
    /// Standard deviation
    fn simd_std(a: &ArrayView1<Self>) -> Self;
    /// L1 norm (Manhattan norm)
    fn simd_norm_l1(a: &ArrayView1<Self>) -> Self;
    /// L∞ norm (Chebyshev norm / max absolute)
    fn simd_norm_linf(a: &ArrayView1<Self>) -> Self;
    /// Cosine similarity between two vectors
    fn simd_cosine_similarity(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self;
    /// Euclidean distance between two vectors
    fn simd_distance_euclidean(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self;
    /// Manhattan distance between two vectors
    fn simd_distance_manhattan(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self;
    /// Chebyshev distance between two vectors
    fn simd_distance_chebyshev(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self;
    /// Cosine distance (1 - cosine_similarity)
    fn simd_distance_cosine(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self;
    /// Squared Euclidean distance between two vectors (no sqrt)
    ///
    /// Computes Σ(a_i - b_i)² — faster than [`simd_distance_euclidean`](Self::simd_distance_euclidean)
    /// when only relative ordering matters (e.g. k-NN comparisons, KdTree/BallTree descent),
    /// since it avoids the trailing `sqrt`.
    fn simd_distance_squared_euclidean(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self;
    /// Weighted sum
    fn simd_weighted_sum(values: &ArrayView1<Self>, weights: &ArrayView1<Self>) -> Self;
    /// Weighted mean
    fn simd_weighted_mean(values: &ArrayView1<Self>, weights: &ArrayView1<Self>) -> Self;
    /// Find index of minimum element (argmin)
    fn simd_argmin(a: &ArrayView1<Self>) -> Option<usize>;
    /// Find index of maximum element (argmax)
    fn simd_argmax(a: &ArrayView1<Self>) -> Option<usize>;
    /// Clip values to [min_val, max_val] range
    fn simd_clip(a: &ArrayView1<Self>, min_val: Self, max_val: Self) -> Array1<Self>;
    /// Log-sum-exp for numerically stable softmax computation
    fn simd_log_sum_exp(a: &ArrayView1<Self>) -> Self;
    /// Softmax for probability distribution (softmax = exp(x - log_sum_exp(x)))
    fn simd_softmax(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Cumulative sum
    fn simd_cumsum(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Cumulative product
    fn simd_cumprod(a: &ArrayView1<Self>) -> Array1<Self>;
    /// First-order difference (`a[i+1] - a[i]`)
    fn simd_diff(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Sign function: returns -1 for negative, 0 for zero, +1 for positive
    fn simd_sign(a: &ArrayView1<Self>) -> Array1<Self>;
    /// ReLU activation: max(0, x)
    fn simd_relu(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Leaky ReLU: x if x > 0 else alpha * x
    fn simd_leaky_relu(a: &ArrayView1<Self>, alpha: Self) -> Array1<Self>;
    /// L2 normalization (unit vector)
    fn simd_normalize(a: &ArrayView1<Self>) -> Array1<Self>;
    /// Standardization: (x - mean) / std
    fn simd_standardize(a: &ArrayView1<Self>) -> Array1<Self>;

    // ============================================================================
    // ZERO-ALLOCATION SIMD OPERATIONS (Phase 1: ToRSh SIMD Performance Fix)
    // ============================================================================
    // These methods write directly to pre-allocated output buffers, eliminating
    // all intermediate allocations. Critical for achieving SIMD speedup in ToRSh.
    //
    // Design rationale:
    // - Use raw slices (&[Self], &mut [Self]) instead of ArrayView for maximum efficiency
    // - No intermediate Array1 allocation - writes directly to output
    // - Enables ToRSh to reduce from 4 allocations to 1 per operation
    // ============================================================================

    /// Zero-allocation element-wise addition: output = a + b
    ///
    /// Writes SIMD addition results directly to pre-allocated output buffer.
    /// This is the core operation for achieving SIMD speedup without allocation overhead.
    ///
    /// # Panics
    /// Panics if `a`, `b`, and `output` do not have the same length.
    ///
    /// # Performance
    /// - x86_64 AVX2: Processes 8 elements per cycle
    /// - ARM64 NEON: Processes 4 elements per cycle
    /// - Expected 2-4x speedup over scalar for large arrays (>1000 elements)
    fn simd_add_into(a: &[Self], b: &[Self], output: &mut [Self]);

    /// Zero-allocation element-wise subtraction: output = a - b
    fn simd_sub_into(a: &[Self], b: &[Self], output: &mut [Self]);

    /// Zero-allocation element-wise multiplication: output = a * b
    ///
    /// Writes SIMD multiplication results directly to pre-allocated output buffer.
    fn simd_mul_into(a: &[Self], b: &[Self], output: &mut [Self]);

    /// Zero-allocation element-wise division: output = a / b
    fn simd_div_into(a: &[Self], b: &[Self], output: &mut [Self]);

    /// In-place element-wise addition: a += b
    ///
    /// Modifies `a` in-place, adding corresponding elements from `b`.
    /// Zero allocations, zero copies - pure SIMD operation.
    fn simd_add_inplace(a: &mut [Self], b: &[Self]);

    /// In-place element-wise subtraction: a -= b
    fn simd_sub_inplace(a: &mut [Self], b: &[Self]);

    /// In-place element-wise multiplication: a *= b
    fn simd_mul_inplace(a: &mut [Self], b: &[Self]);

    /// In-place element-wise division: a /= b
    fn simd_div_inplace(a: &mut [Self], b: &[Self]);

    /// In-place scalar addition: a += scalar
    fn simd_add_scalar_inplace(a: &mut [Self], scalar: Self);

    /// In-place scalar multiplication: a *= scalar
    fn simd_mul_scalar_inplace(a: &mut [Self], scalar: Self);

    /// Zero-allocation fused multiply-add: output = a * b + c
    fn simd_fma_into(a: &[Self], b: &[Self], c: &[Self], output: &mut [Self]);
}
