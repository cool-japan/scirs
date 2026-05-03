//! WGSL compute-shader kernels for browser-based WASM deployment.
//!
//! Each constant holds the WGSL source for a `@compute` shader that
//! evaluates a batch of special-function values.  The shaders operate on
//! `array<f32>` — inputs must be cast from `f64` before upload and cast
//! back after download.
//!
//! The host-side dispatch function [`gamma_batch_wgpu`] is a stub; the
//! actual `wgpu` device/queue/pipeline setup is left as a future integration
//! point (it requires the `wasm_wgpu` feature that is not yet available
//! in `scirs2-core`).  Until that integration ships, the function always
//! returns [`WgslDispatchError::GpuNotAvailable`] so the caller can fall
//! back to CPU.
//!
//! # Feature gating
//!
//! The module is unconditionally compiled so that the shader sources are
//! always inspectable (useful for documentation and validation tooling).
//! The actual GPU dispatch path is guarded by `#[cfg(feature = "gpu")]`
//! which maps to `scirs2-core/gpu`.

// ---------------------------------------------------------------------------
// WGSL shader sources
// ---------------------------------------------------------------------------

/// WGSL compute shader for batch Gamma evaluation (Lanczos g=7 approximation).
///
/// Workgroup size 64.  Each invocation reads one `f32` from `input` and
/// writes the approximated `Γ(x)` into `output`.
/// The reflection formula `Γ(x) = π / (sin(π x) · Γ(1-x))` is applied when
/// `x < 0.5`.
pub const GAMMA_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const PI: f32 = 3.14159265358979323846;

// Lanczos g=7 coefficients (Spouge's form, 9 terms)
fn lanczos_gamma(x_in: f32) -> f32 {
    var x = x_in;
    var sign = 1.0f;
    if x < 0.5 {
        sign = PI / (sin(PI * x));
        x = 1.0 - x;
    }
    let g: f32 = 7.0;
    x = x - 1.0;

    let c0: f32 =  0.99999999999980993;
    let c1: f32 =  676.5203681218851;
    let c2: f32 = -1259.1392167224028;
    let c3: f32 =  771.32342877765313;
    let c4: f32 = -176.61502916214059;
    let c5: f32 =  12.507343278686905;
    let c6: f32 = -0.13857109526572012;
    let c7: f32 =  9.9843695780195716e-6;
    let c8: f32 =  1.5056327351493116e-7;

    let s = c0
        + c1 / (x + 1.0)
        + c2 / (x + 2.0)
        + c3 / (x + 3.0)
        + c4 / (x + 4.0)
        + c5 / (x + 5.0)
        + c6 / (x + 6.0)
        + c7 / (x + 7.0)
        + c8 / (x + 8.0);

    let t = x + g + 0.5;
    let result = sqrt(2.0 * PI) * pow(t, x + 0.5) * exp(-t) * s;
    if sign != 1.0 { return sign / result; }
    return result;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&input) { return; }
    output[idx] = lanczos_gamma(input[idx]);
}
"#;

/// WGSL compute shader for batch `erf` evaluation.
///
/// Uses the Abramowitz & Stegun 7.1.26 approximation (max error ≈ 1.5 × 10⁻⁷).
pub const ERF_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

fn approx_erf(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.3275911 * abs(x));
    let y = 1.0 - (((((
          1.061405429 * t
        - 1.453152027) * t
        + 1.421413741) * t
        - 0.284496736) * t
        + 0.254829592) * t * exp(-x * x));
    return select(-y, y, x >= 0.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&input) { return; }
    output[idx] = approx_erf(input[idx]);
}
"#;

/// WGSL compute shader for batch Bessel J₀ evaluation.
///
/// Uses the polynomial approximation from Abramowitz & Stegun §9.4 for
/// |x| < 8 and the asymptotic expansion for |x| ≥ 8.
pub const BESSEL_J0_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const PI: f32 = 3.14159265358979323846;

fn bessel_j0(x_in: f32) -> f32 {
    let x = abs(x_in);
    if x < 8.0 {
        let y = x * x;
        let p1: f32 =  57568490574.0;
        let p2: f32 = -13362590354.0;
        let p3: f32 =  651619640.7;
        let p4: f32 = -11214424.18;
        let p5: f32 =  77392.33017;
        let p6: f32 = -184.9052456;
        let q1: f32 =  57568490411.0;
        let q2: f32 =  1029532985.0;
        let q3: f32 =  9494680.718;
        let q4: f32 =  59272.64853;
        let q5: f32 =  267.8532712;
        let p = p1 + y * (p2 + y * (p3 + y * (p4 + y * (p5 + y * p6))));
        let q = q1 + y * (q2 + y * (q3 + y * (q4 + y * (q5 + y))));
        return p / q;
    } else {
        let z = 8.0 / x;
        let y = z * z;
        let xx = x - 0.785398164;
        let pv = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4
                 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
        let qv = -0.1562499995e-1 + y * (0.1430488765e-3
                 + y * (-0.6911147651e-5 + y * (0.7621095161e-6
                 - y * 0.934945152e-7)));
        return sqrt(0.636619772 / x) * (cos(xx) * pv - z * sin(xx) * qv);
    }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&input) { return; }
    output[idx] = bessel_j0(input[idx]);
}
"#;

// ---------------------------------------------------------------------------
// Dispatch error
// ---------------------------------------------------------------------------

/// Error type for WGSL/WebGPU dispatch.
#[derive(Debug, Clone)]
pub enum WgslDispatchError {
    /// No wgpu device is available (headless or non-WASM build).
    GpuNotAvailable,
    /// The wgpu pipeline setup or execution failed.
    RuntimeError(String),
}

impl std::fmt::Display for WgslDispatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WgslDispatchError::GpuNotAvailable => {
                write!(f, "wgpu GPU device not available")
            }
            WgslDispatchError::RuntimeError(msg) => {
                write!(f, "wgpu runtime error: {msg}")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Host-side dispatch stubs
// ---------------------------------------------------------------------------

/// Attempt batch Gamma evaluation on a WebGPU device.
///
/// # Current state
///
/// This is a stub.  It always returns [`WgslDispatchError::GpuNotAvailable`]
/// so that the call site in [`crate::gpu_dispatch`] falls back to the CPU
/// rayon path.
///
/// # Future integration
///
/// When `scirs2-core/gpu` gains a stable `wgpu` adapter API this function
/// should:
/// 1. Cast each `f64` in `xs` to `f32`.
/// 2. Upload the slice to a GPU buffer.
/// 3. Create a pipeline from [`GAMMA_WGSL`], dispatch `ceil(n/64)` work-groups.
/// 4. Download the result buffer, cast `f32` → `f64`, and return.
#[allow(unused_variables)]
pub fn gamma_batch_wgpu(xs: &[f64]) -> Result<Vec<f64>, WgslDispatchError> {
    Err(WgslDispatchError::GpuNotAvailable)
}

/// Attempt batch `erf` evaluation on a WebGPU device.
///
/// Stub — always returns [`WgslDispatchError::GpuNotAvailable`].
#[allow(unused_variables)]
pub fn erf_batch_wgpu(xs: &[f64]) -> Result<Vec<f64>, WgslDispatchError> {
    Err(WgslDispatchError::GpuNotAvailable)
}

/// Attempt batch Bessel J₀ evaluation on a WebGPU device.
///
/// Stub — always returns [`WgslDispatchError::GpuNotAvailable`].
#[allow(unused_variables)]
pub fn bessel_j0_batch_wgpu(xs: &[f64]) -> Result<Vec<f64>, WgslDispatchError> {
    Err(WgslDispatchError::GpuNotAvailable)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamma_wgsl_source_is_non_empty() {
        assert!(!GAMMA_WGSL.is_empty());
        // Basic structural checks
        assert!(GAMMA_WGSL.contains("@compute"));
        assert!(GAMMA_WGSL.contains("workgroup_size"));
        assert!(GAMMA_WGSL.contains("lanczos_gamma"));
    }

    #[test]
    fn test_erf_wgsl_source_is_non_empty() {
        assert!(!ERF_WGSL.is_empty());
        assert!(ERF_WGSL.contains("@compute"));
        assert!(ERF_WGSL.contains("approx_erf"));
    }

    #[test]
    fn test_bessel_j0_wgsl_source_is_non_empty() {
        assert!(!BESSEL_J0_WGSL.is_empty());
        assert!(BESSEL_J0_WGSL.contains("@compute"));
        assert!(BESSEL_J0_WGSL.contains("bessel_j0"));
    }

    #[test]
    fn test_gamma_batch_wgpu_returns_not_available() {
        let xs = vec![1.0_f64, 2.0, 3.0];
        let result = gamma_batch_wgpu(&xs);
        assert!(
            matches!(result, Err(WgslDispatchError::GpuNotAvailable)),
            "expected GpuNotAvailable, got {:?}",
            result
        );
    }

    #[test]
    fn test_erf_batch_wgpu_returns_not_available() {
        let xs = vec![0.0_f64, 1.0];
        let result = erf_batch_wgpu(&xs);
        assert!(matches!(result, Err(WgslDispatchError::GpuNotAvailable)));
    }

    #[test]
    fn test_bessel_j0_batch_wgpu_returns_not_available() {
        let xs = vec![0.0_f64, 2.405];
        let result = bessel_j0_batch_wgpu(&xs);
        assert!(matches!(result, Err(WgslDispatchError::GpuNotAvailable)));
    }

    #[test]
    fn test_wgsl_dispatch_error_display() {
        let e = WgslDispatchError::GpuNotAvailable;
        assert!(e.to_string().contains("not available"));
        let e2 = WgslDispatchError::RuntimeError("buffer overflow".into());
        assert!(e2.to_string().contains("buffer overflow"));
    }
}
