// GPU compute shader for gamma function calculation
// Uses Lanczos approximation for gamma computation

@group(0) @binding(0)
var<storage, read> input_data: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output_data: array<f32>;

// Lanczos coefficients for gamma approximation
const LANCZOS_G: f32 = 7.0;
const LANCZOS_COEFFS: array<f32, 9> = array<f32, 9>(
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7
);

// Approximate gamma function using Lanczos approximation
fn gamma_lanczos(x: f32) -> f32 {
    if (x < 0.5) {
        // Use reflection formula for x < 0.5
        let pi = 3.141592653589793;
        return pi / (sin(pi * x) * gamma_lanczos(1.0 - x));
    }
    
    let z = x - 1.0;
    var acc = LANCZOS_COEFFS[0];
    
    for (var i = 1u; i < 9u; i++) {
        acc += LANCZOS_COEFFS[i] / (z + f32(i));
    }
    
    let t = z + LANCZOS_G + 0.5;
    let sqrt_2pi = 2.5066282746310005;
    
    return sqrt_2pi * acc * pow(t, z + 0.5) * exp(-t);
}

// Special cases and optimizations
fn gamma_optimized(x: f32) -> f32 {
    // Handle special cases
    if (x <= 0.0) {
        return f32(0x7fc00000); // NaN
    }
    
    // For very small positive values
    if (x < 1e-6) {
        let euler_gamma = 0.5772156649015329;
        return 1.0 / x - euler_gamma + euler_gamma * euler_gamma * x * 0.5;
    }
    
    // Integer values (exact computation)
    let x_int = round(x);
    if (abs(x - x_int) < 1e-10 && x_int > 0.0 && x_int <= 12.0) {
        var result = 1.0;
        for (var i = 1; i < i32(x_int); i++) {
            result *= f32(i);
        }
        return result;
    }
    
    // Half-integer values
    if (abs(x * 2.0 - round(x * 2.0)) < 1e-10 && x > 0.0) {
        let sqrt_pi = 1.7724538509055159;
        if (abs(x - 0.5) < 1e-10) {
            return sqrt_pi;
        }
        if (abs(x - 1.5) < 1e-10) {
            return 0.5 * sqrt_pi;
        }
    }
    
    // Use Lanczos approximation for general case
    return gamma_lanczos(x);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= arrayLength(&input_data)) {
        return;
    }
    
    let x = input_data[index];
    output_data[index] = gamma_optimized(x);
}