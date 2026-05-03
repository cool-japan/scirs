// Cooley-Tukey radix-2 DIT FFT compute shader.
//
// One thread handles one butterfly pair.  The host dispatches this shader
// once per stage (0 to log2(n)−1), updating the `params.stage` uniform
// between dispatches.
//
// Complex numbers are stored as vec2<f32> in a storage buffer:
//   data[i].x = real part
//   data[i].y = imaginary part
//
// Twiddle factor for butterfly k in stage s:
//   W = exp(sign * 2π * i * k / n)
//   where sign = +1 for inverse FFT, −1 for forward FFT.
//
// Workgroup size: 64 threads. Host must dispatch ceil(n/2 / 64) workgroups.

struct FFTParams {
    /// Total number of complex samples (must be a power of two).
    n: u32,
    /// Current butterfly stage index (0 = first stage, log2(n)−1 = last).
    stage: u32,
    /// 0 → forward FFT (twiddle sign = −1)
    /// 1 → inverse FFT (twiddle sign = +1)
    inverse: u32,
    /// Padding to satisfy 16-byte alignment of the uniform buffer.
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> params: FFTParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    let n = params.n;
    let half_n = n >> 1u;

    // Each thread handles exactly one butterfly pair.  Exit early if this
    // thread index is out of range.
    if k >= half_n {
        return;
    }

    let stage  = params.stage;
    let stride = 1u << stage;           // distance between the two elements of a butterfly
    let group  = k / stride;            // which butterfly group this thread belongs to
    let pos    = k % stride;            // position within the group

    // Indices of the butterfly pair.
    let i = group * (stride << 1u) + pos;
    let j = i + stride;

    // Twiddle factor exponent:
    //   forward:  angle = −2π * group * stride / n  (negative exponent)
    //   inverse:  angle = +2π * group * stride / n  (positive exponent)
    let sign  = select(-1.0, 1.0, params.inverse != 0u);
    let angle = sign * 6.283185307179586 * f32(group * stride) / f32(n);

    let tw = vec2<f32>(cos(angle), sin(angle));

    let a = data[i];
    let b = data[j];

    // Complex multiply: bt = b * tw
    let bt = vec2<f32>(
        b.x * tw.x - b.y * tw.y,
        b.x * tw.y + b.y * tw.x,
    );

    // Butterfly output:
    //   data[i] = a + bt
    //   data[j] = a - bt
    data[i] = a + bt;
    data[j] = a - bt;
}
