# SciRS2-WASM: Scientific Computing in WebAssembly

High-performance scientific computing for JavaScript and TypeScript environments, powered by Rust compiled to WebAssembly. Part of the [SciRS2](https://github.com/cool-japan/scirs) ecosystem.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](../LICENSE)
[![Version](https://img.shields.io/badge/version-0.4.4-green)]()

## Overview

`scirs2-wasm` brings the full power of SciRS2's scientific computing capabilities to the browser and Node.js through WebAssembly. It exposes a `wasm-bindgen`-based interface with TypeScript type definitions, SIMD-accelerated operations (where supported by the runtime), and utilities for linear algebra, signal processing, statistics, machine learning, and streaming data processing.

## Features

- **Pure Rust**: 100% safe Rust code compiled to `wasm32-unknown-unknown`
- **wasm-bindgen Interface**: Direct JS/TS interop without glue code overhead
- **TypeScript Definitions**: Full type definitions in `ts-src/scirs2.ts` for strong IDE support
- **SIMD Acceleration**: Optional `wasm32-simd128` acceleration for vectorized operations
- **WasmMatrix Type**: First-class 2D matrix type for linear algebra in JS/TS
- **Async Operations**: Non-blocking computations with JavaScript Promises
- **WebWorker Support**: Offload heavy computations to worker threads
- **Streaming Processing**: Process large datasets incrementally without exhausting memory
- **Signal Processing**: FFT, filtering, spectrogram generation
- **Linear Algebra**: Matrix operations, SVD, eigenvalue decomposition
- **Statistics**: Descriptive statistics, distributions, hypothesis tests, regression
- **ML Utilities**: Forward pass helpers, activation functions, loss computation
- **Advanced Stats**: Time series primitives, regression diagnostics
- **Memory Efficient**: Optimized allocations for browser memory constraints
- **Zero JS Dependencies**: No external JavaScript runtime dependencies required

## Installation

### NPM

```bash
npm install scirs2-wasm
```

### Yarn

```bash
yarn add scirs2-wasm
```

### PNPM

```bash
pnpm add scirs2-wasm
```

## Quick Start

### Browser (ES Modules)

```javascript
import init, * as scirs2 from 'scirs2-wasm';

async function main() {
  await init();

  const a = new scirs2.WasmArray([1, 2, 3, 4]);
  const b = new scirs2.WasmArray([5, 6, 7, 8]);

  const sum  = scirs2.add(a, b);
  const mean = scirs2.mean(a);
  const std  = scirs2.std(a);

  console.log('Sum:', sum.to_array());
  console.log('Mean:', mean, 'Std:', std);
}

main();
```

### TypeScript

```typescript
import init, * as scirs2 from 'scirs2-wasm';

async function main(): Promise<void> {
  await init();

  // WasmMatrix for 2D linear algebra
  const mat: scirs2.WasmMatrix = scirs2.WasmMatrix.from_rows([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
  ]);

  const svd = scirs2.svd(mat);
  console.log('Singular values:', svd.s.to_array());

  // Check SIMD availability
  console.log('SIMD support:', scirs2.has_simd_support());
}

main();
```

### Node.js

```javascript
const scirs2 = require('scirs2-wasm');

async function main() {
  const matrix = scirs2.WasmArray.from_shape([2, 2], [1, 2, 3, 4]);

  const det   = scirs2.det(matrix);
  const trace = scirs2.trace(matrix);

  console.log('Determinant:', det);
  console.log('Trace:', trace);
}

main();
```

## API Reference

### Array Creation

```javascript
const arr      = new scirs2.WasmArray([1, 2, 3, 4]);
const typed    = new scirs2.WasmArray(new Float64Array([1, 2, 3, 4]));
const matrix   = scirs2.WasmArray.from_shape([2, 2], [1, 2, 3, 4]);
const zeros    = scirs2.WasmArray.zeros([3, 3]);
const ones     = scirs2.WasmArray.ones([5]);
const linspace = scirs2.WasmArray.linspace(0, 1, 50);
const arange   = scirs2.WasmArray.arange(0, 10, 0.5);
```

### Linear Algebra (WasmMatrix)

```javascript
const A = scirs2.WasmMatrix.from_rows([[1,2],[3,4]]);
const B = scirs2.WasmMatrix.from_rows([[5,6],[7,8]]);

const C         = scirs2.matmul(A, B);          // Matrix multiply
const inv       = scirs2.inv(A);                // Inverse
const det       = scirs2.det(A);                // Determinant
const trace     = scirs2.trace(A);              // Trace
const svd_res   = scirs2.svd(A);                // SVD: {U, s, Vt}
const eig_res   = scirs2.eig(A);                // Eigenvalues/vectors
const norm      = scirs2.norm_frobenius(A);     // Frobenius norm
const rank      = scirs2.matrix_rank(A);        // Rank
const x         = scirs2.solve(A, b_vec);       // Ax = b
const transpose = A.transpose();
```

### Signal Processing

```javascript
const signal = new scirs2.WasmArray([/* samples */]);

// FFT
const spectrum      = scirs2.fft(signal);
const power         = scirs2.periodogram(signal, 1024);
const spectrogram   = scirs2.spectrogram(signal, { nperseg: 256, noverlap: 128 });

// Filtering
const filtered_lp   = scirs2.lowpass_filter(signal, 0.2);
const filtered_bp   = scirs2.bandpass_filter(signal, 0.1, 0.4);
const filtered_hp   = scirs2.highpass_filter(signal, 0.3);

// Convolution
const kernel        = new scirs2.WasmArray([0.25, 0.5, 0.25]);
const convolved     = scirs2.convolve(signal, kernel);
```

### Statistics

```javascript
const data = new scirs2.WasmArray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

// Descriptive
const mean     = scirs2.mean(data);
const std      = scirs2.std(data);
const variance = scirs2.variance(data);
const median   = scirs2.median(data);
const p25      = scirs2.percentile(data, 25);
const p75      = scirs2.percentile(data, 75);
const skew     = scirs2.skewness(data);
const kurt     = scirs2.kurtosis(data);

// Correlation
const x    = new scirs2.WasmArray([1, 2, 3, 4, 5]);
const y    = new scirs2.WasmArray([2, 4, 6, 8, 10]);
const corr = scirs2.corrcoef(x, y);

// Advanced: regression diagnostics
const reg_result = scirs2.linear_regression(x, y);
console.log('Slope:', reg_result.slope, 'R2:', reg_result.r2);
```

### ML Utilities

```javascript
// Activation functions
const relu    = scirs2.relu(data);
const sigmoid = scirs2.sigmoid(data);
const softmax = scirs2.softmax(data);
const tanh    = scirs2.tanh_activation(data);

// Loss functions
const mse_loss  = scirs2.mse_loss(predictions, targets);
const ce_loss   = scirs2.cross_entropy_loss(logits, labels);

// Normalization
const normalized    = scirs2.layer_norm(data);
const batch_normed  = scirs2.batch_norm(data, mean_vec, var_vec);
```

### Streaming Data Processing

```javascript
// Process large datasets in chunks
const processor = new scirs2.StreamingProcessor({ window_size: 1024 });

for (const chunk of dataSource) {
  processor.push(chunk);
  const current_stats = processor.current_stats();
  console.log('Running mean:', current_stats.mean);
}

const final_result = processor.finalize();
```

### WebWorker Support

```javascript
// In main thread
const worker = new Worker('scirs2-worker.js');
worker.postMessage({ op: 'fft', data: signal_data });
worker.onmessage = (e) => console.log('FFT result:', e.data.result);
```

## Building from Source

### Prerequisites

- Rust 1.75+ with `wasm32-unknown-unknown` target
- `wasm-pack` 0.12+
- Node.js 18+

### Build Steps

```bash
# Install Rust WASM target
rustup target add wasm32-unknown-unknown

# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build for bundlers (webpack, rollup, Vite)
npm run build

# Build for web (plain ES modules, no bundler needed)
npm run build:web

# Build for Node.js (CommonJS)
npm run build:nodejs

# Build with SIMD acceleration
npm run build:simd

# Optimize binary with wasm-opt
npm run optimize
```

### Testing

```bash
# Run tests in headless Firefox
npm test

# Run tests in headless Chrome
npm run test:chrome

# Run tests in Node.js
npm run test:node

# Run Rust unit tests (non-WASM)
cargo test --lib
```

## Performance

SciRS2-WASM delivers near-native performance for scientific workloads:

| Operation | Performance vs Native | Notes |
|-----------|----------------------|-------|
| Array arithmetic | 80-95% | Scalar fallback without SIMD |
| Matrix multiply | Up to 90% | With `wasm32-simd128` |
| Statistical functions | 85-95% | Parallel not available in WASM |
| FFT | 75-90% | Depends on transform size |
| Random number generation | 70-85% | WASM RNG overhead |

### SIMD Acceleration

Build with `--features simd` for `wasm32-simd128` vectorization:

```bash
RUSTFLAGS="-C target-feature=+simd128" wasm-pack build --release
```

SIMD is supported in Chrome 91+, Firefox 89+, Safari 16.4+, Edge 91+, and Node.js 20+.

## Browser Compatibility

| Browser | Baseline | With SIMD |
|---------|----------|-----------|
| Chrome  | 91+      | 91+       |
| Firefox | 89+      | 89+       |
| Safari  | 16.4+    | Limited   |
| Edge    | 91+      | 91+       |
| Node.js | 18+      | 20+       |

## TypeScript Type Definitions

Full TypeScript definitions are provided in `ts-src/scirs2.ts`. The `WasmMatrix` and `WasmArray` types are strongly typed for ergonomic IDE support:

```typescript
import type { WasmMatrix, WasmArray, SvdResult, EigResult } from 'scirs2-wasm';

function process(m: WasmMatrix): SvdResult {
  return scirs2.svd(m);
}
```

## Memory Management

WASM uses a linear memory model. Best practices:

1. **Reuse arrays** when possible to reduce allocation overhead
2. **Process in chunks** for datasets larger than a few hundred MB
3. **Monitor memory** with browser DevTools Performance panel
4. **Dispose large temporaries** explicitly when done

## Module Architecture

```
scirs2-wasm/
├── src/
│   ├── lib.rs           - WASM entry point, init, version
│   ├── array.rs         - WasmArray type and array operations
│   ├── linalg.rs        - WasmMatrix, decompositions, solvers
│   ├── stats.rs         - Descriptive stats, distributions, tests
│   ├── fft.rs           - FFT, spectrogram, periodogram
│   ├── signal.rs        - Filters, convolution, denoising
│   ├── optimize.rs      - Optimization algorithms
│   ├── integrate.rs     - Numerical integration
│   ├── interpolate.rs   - Interpolation methods
│   ├── random.rs        - RNG and distributions
│   ├── utils.rs         - Performance timer, capabilities
│   └── error.rs         - WASM-friendly error types
├── ts-src/
│   └── scirs2.ts        - TypeScript definitions
└── Cargo.toml
```

## Related Projects

- [SciRS2](https://github.com/cool-japan/scirs) - Core Rust library
- [scirs2-python](../scirs2-python/) - Python/PyO3 bindings
- [wasm-bindgen](https://github.com/rustwasm/wasm-bindgen) - WASM-JS interop framework

## License

Licensed under the Apache License 2.0. See [LICENSE](../LICENSE) for details.

## Authors

COOLJAPAN OU (Team KitaSan)
