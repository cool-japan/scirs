/**
 * Browser-side benchmark: scirs2-wasm vs tfjs-wasm
 *
 * Usage: open comparison_bench.html in a browser that supports WASM.
 * Serve the page with COOP/COEP headers to enable SharedArrayBuffer (see
 * js/setup.js).
 *
 * The benchmark measures wall-clock time for identical operations across:
 *   - Plain JavaScript      (reference baseline)
 *   - scirs2-wasm           (via WebAssembly.instantiate)
 *   - TF.js-WASM            (optional; loads if tf global is present)
 *
 * ml5.js is a higher-level wrapper around TF.js; it does not expose the
 * low-level matrix/FFT primitives measured here, so it is omitted from this
 * benchmark.  Its interactive-ML helpers (image classifier, pose detector) can
 * be compared separately in browser DevTools.
 *
 * Operations benchmarked
 * ----------------------
 *   1. Matrix multiply       (64×64, 128×128 float32)
 *   2. Element-wise sigmoid  (16384 elements)
 *   3. Dot product           (16384 elements)
 *   4. Softmax               (4096 elements)
 *   5. Mean + variance       (65536 elements)
 */

/* global tf */

// ---------------------------------------------------------------------------
// Timing helper
// ---------------------------------------------------------------------------

/**
 * Run `fn` for at least `minDurationMs` and return statistics.
 *
 * @param {() => unknown} fn        - Synchronous function to benchmark
 * @param {number}        warmup    - Number of warm-up iterations
 * @param {number}        minDurationMs - Minimum benchmark duration
 * @returns {{ mean: number, p50: number, p95: number, iters: number }}
 */
export function bench(fn, warmup = 3, minDurationMs = 200) {
  for (let i = 0; i < warmup; i++) fn();

  const times = [];
  const deadline = performance.now() + minDurationMs;
  while (performance.now() < deadline) {
    const t0 = performance.now();
    fn();
    times.push(performance.now() - t0);
  }

  times.sort((a, b) => a - b);
  const mean = times.reduce((a, b) => a + b, 0) / times.length;
  const p50 = times[Math.floor(times.length * 0.5)];
  const p95 = times[Math.floor(times.length * 0.95)];
  return { mean, p50, p95, iters: times.length };
}

/**
 * Run an async `fn` and return statistics.
 *
 * @param {() => Promise<unknown>} fn
 * @param {number} warmup
 * @param {number} minDurationMs
 */
export async function benchAsync(fn, warmup = 3, minDurationMs = 200) {
  for (let i = 0; i < warmup; i++) await fn();

  const times = [];
  const deadline = performance.now() + minDurationMs;
  while (performance.now() < deadline) {
    const t0 = performance.now();
    await fn();
    times.push(performance.now() - t0);
  }

  times.sort((a, b) => a - b);
  const mean = times.reduce((a, b) => a + b, 0) / times.length;
  const p50 = times[Math.floor(times.length * 0.5)];
  const p95 = times[Math.floor(times.length * 0.95)];
  return { mean, p50, p95, iters: times.length };
}

// ---------------------------------------------------------------------------
// Pure-JS reference implementations
// ---------------------------------------------------------------------------

export const JS = {
  matmul(a, b, n) {
    const c = new Float32Array(n * n);
    for (let i = 0; i < n; i++)
      for (let k = 0; k < n; k++) {
        const aik = a[i * n + k];
        for (let j = 0; j < n; j++) c[i * n + j] += aik * b[k * n + j];
      }
    return c;
  },

  sigmoid(x) {
    return x.map((v) => 1 / (1 + Math.exp(-v)));
  },

  dot(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  },

  softmax(x) {
    const max = Math.max(...x);
    const exps = x.map((v) => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((e) => e / sum);
  },

  meanVariance(x) {
    const n = x.length;
    const mean = x.reduce((a, b) => a + b, 0) / n;
    const variance = x.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
    return { mean, variance };
  },
};

// ---------------------------------------------------------------------------
// scirs2-wasm shim (replace with real WASM exports once loaded)
// ---------------------------------------------------------------------------

export let scirs2 = null;

/**
 * Initialise scirs2-wasm.  In a real deployment, replace the URL with the
 * actual WASM package path (e.g. '/pkg/scirs2_wasm_bg.wasm').
 *
 * @param {string} wasmUrl
 */
export async function initScirs2(wasmUrl) {
  try {
    const response = await fetch(wasmUrl);
    const buffer = await response.arrayBuffer();
    const { instance } = await WebAssembly.instantiate(buffer, {});
    scirs2 = instance.exports;
    console.log("[scirs2] WASM loaded OK");
  } catch (err) {
    console.warn("[scirs2] Could not load WASM:", err.message, "— using JS fallback");
    scirs2 = JS; // Fall back to JS for demo purposes
  }
}

// ---------------------------------------------------------------------------
// tfjs-wasm helpers (requires <script src="...@tensorflow/tfjs"></script>)
// ---------------------------------------------------------------------------

const TF = {
  matmul(a, b, n) {
    if (typeof tf === "undefined") return null;
    const ta = tf.tensor2d(a, [n, n]);
    const tb = tf.tensor2d(b, [n, n]);
    const tc = tf.matMul(ta, tb);
    const data = tc.dataSync(); // synchronous — forces GPU→CPU transfer
    ta.dispose();
    tb.dispose();
    tc.dispose();
    return data;
  },
};

// ---------------------------------------------------------------------------
// scirs2-wasm helpers
//
// Each helper calls the corresponding WASM export when available.  The WASM
// memory layout expected:
//
//   matmul_f32(ptr_a, ptr_b, ptr_out, n)
//     - Reads n×n f32 row-major matrices from ptr_a, ptr_b in WASM memory
//     - Writes n×n result into ptr_out
//
//   dot_f32(ptr_a, ptr_b, len) → f32
//   sigmoid_f32(ptr_in, ptr_out, len)
//   softmax_f32(ptr_in, ptr_out, len)
//   mean_f32(ptr, len) → f32
//   variance_f32(ptr, mean, len) → f32
//
//   alloc_f32(len) → ptr   – allocate len f32 slots, return WASM pointer
//   free_f32(ptr, len)     – release the allocation
//
// When the WASM module is not loaded (scirs2 === JS), the same operations
// fall through to the pure-JS implementations already in the JS namespace.
// ---------------------------------------------------------------------------

/**
 * Write a Float32Array into WASM linear memory and return the pointer.
 *
 * @param {WebAssembly.Memory} mem   - WASM memory object
 * @param {Float32Array}       data  - Source data
 * @param {number}             ptr   - Byte offset in WASM memory
 */
function writeF32(mem, data, ptr) {
  new Float32Array(mem.buffer, ptr, data.length).set(data);
}

/**
 * Read a Float32Array slice from WASM linear memory.
 *
 * @param {WebAssembly.Memory} mem   - WASM memory object
 * @param {number}             ptr   - Byte offset
 * @param {number}             len   - Number of f32 elements
 * @returns {Float32Array}
 */
function readF32(mem, ptr, len) {
  return new Float32Array(mem.buffer, ptr, len).slice();
}

/**
 * Return true when the scirs2 global holds a real WASM instance (not the JS
 * fallback).  Detected by checking for a `memory` export — all scirs2-wasm
 * builds include one.
 */
function scirs2IsWasm() {
  return scirs2 !== null && scirs2 !== JS && typeof scirs2.memory !== "undefined";
}

export const SCIRS2 = {
  matmul(a, b, n) {
    if (!scirs2IsWasm()) return JS.matmul(a, b, n);
    const len = n * n;
    const ptrA   = scirs2.alloc_f32(len);
    const ptrB   = scirs2.alloc_f32(len);
    const ptrOut = scirs2.alloc_f32(len);
    writeF32(scirs2.memory, a, ptrA);
    writeF32(scirs2.memory, b, ptrB);
    scirs2.matmul_f32(ptrA, ptrB, ptrOut, n);
    const result = readF32(scirs2.memory, ptrOut, len);
    scirs2.free_f32(ptrA, len);
    scirs2.free_f32(ptrB, len);
    scirs2.free_f32(ptrOut, len);
    return result;
  },

  sigmoid(data) {
    if (!scirs2IsWasm()) return JS.sigmoid(data);
    const len    = data.length;
    const ptrIn  = scirs2.alloc_f32(len);
    const ptrOut = scirs2.alloc_f32(len);
    writeF32(scirs2.memory, data, ptrIn);
    scirs2.sigmoid_f32(ptrIn, ptrOut, len);
    const result = readF32(scirs2.memory, ptrOut, len);
    scirs2.free_f32(ptrIn, len);
    scirs2.free_f32(ptrOut, len);
    return result;
  },

  dot(a, b) {
    if (!scirs2IsWasm()) return JS.dot(a, b);
    const len  = a.length;
    const ptrA = scirs2.alloc_f32(len);
    const ptrB = scirs2.alloc_f32(len);
    writeF32(scirs2.memory, a, ptrA);
    writeF32(scirs2.memory, b, ptrB);
    const result = scirs2.dot_f32(ptrA, ptrB, len);
    scirs2.free_f32(ptrA, len);
    scirs2.free_f32(ptrB, len);
    return result;
  },

  softmax(data) {
    if (!scirs2IsWasm()) return JS.softmax(Array.from(data));
    const len    = data.length;
    const ptrIn  = scirs2.alloc_f32(len);
    const ptrOut = scirs2.alloc_f32(len);
    writeF32(scirs2.memory, data, ptrIn);
    scirs2.softmax_f32(ptrIn, ptrOut, len);
    const result = readF32(scirs2.memory, ptrOut, len);
    scirs2.free_f32(ptrIn, len);
    scirs2.free_f32(ptrOut, len);
    return result;
  },

  meanVariance(data) {
    if (!scirs2IsWasm()) return JS.meanVariance(data);
    const len = data.length;
    const ptr = scirs2.alloc_f32(len);
    writeF32(scirs2.memory, data, ptr);
    const mean     = scirs2.mean_f32(ptr, len);
    const variance = scirs2.variance_f32(ptr, mean, len);
    scirs2.free_f32(ptr, len);
    return { mean, variance };
  },
};

// ---------------------------------------------------------------------------
// Benchmark suite
// ---------------------------------------------------------------------------

export const SUITES = [
  {
    name: "matmul_64",
    label: "Matrix multiply (64×64)",
    setup() {
      const n = 64;
      const a = Float32Array.from({ length: n * n }, (_, i) => i * 0.001);
      const b = Float32Array.from({ length: n * n }, (_, i) => (n * n - i) * 0.001);
      return { a, b, n };
    },
    runJs({ a, b, n }) { return JS.matmul(a, b, n); },
    runScirs2({ a, b, n }) { return SCIRS2.matmul(a, b, n); },
    runTf({ a, b, n }) { return TF.matmul(a, b, n); },
  },
  {
    name: "matmul_128",
    label: "Matrix multiply (128×128)",
    setup() {
      const n = 128;
      const a = Float32Array.from({ length: n * n }, (_, i) => i * 0.001);
      const b = Float32Array.from({ length: n * n }, (_, i) => (n * n - i) * 0.001);
      return { a, b, n };
    },
    runJs({ a, b, n }) { return JS.matmul(a, b, n); },
    runScirs2({ a, b, n }) { return SCIRS2.matmul(a, b, n); },
    runTf({ a, b, n }) { return TF.matmul(a, b, n); },
  },
  {
    name: "sigmoid_16k",
    label: "Sigmoid (16 384 elements)",
    setup() {
      const data = Float32Array.from({ length: 16384 }, (_, i) => (i - 8192) * 0.01);
      return { data };
    },
    runJs({ data }) { return JS.sigmoid(data); },
    runScirs2({ data }) { return SCIRS2.sigmoid(data); },
    runTf({ data }) {
      if (typeof tf === "undefined") return null;
      const t = tf.tensor1d(data);
      const r = tf.sigmoid(t);
      const d = r.dataSync();
      t.dispose();
      r.dispose();
      return d;
    },
  },
  {
    name: "dot_16k",
    label: "Dot product (16 384 elements)",
    setup() {
      const a = Float32Array.from({ length: 16384 }, (_, i) => i);
      const b = Float32Array.from({ length: 16384 }, (_, i) => 16384 - i);
      return { a, b };
    },
    runJs({ a, b }) { return JS.dot(a, b); },
    runScirs2({ a, b }) { return SCIRS2.dot(a, b); },
    runTf({ a, b }) {
      if (typeof tf === "undefined") return null;
      const ta = tf.tensor1d(a);
      const tb = tf.tensor1d(b);
      const r = ta.dot(tb);
      const d = r.dataSync()[0];
      ta.dispose();
      tb.dispose();
      r.dispose();
      return d;
    },
  },
  {
    name: "softmax_4k",
    label: "Softmax (4 096 elements)",
    setup() {
      const data = Float32Array.from({ length: 4096 }, (_, i) => i * 0.1);
      return { data };
    },
    runJs({ data }) { return JS.softmax(Array.from(data)); },
    runScirs2({ data }) { return SCIRS2.softmax(data); },
    runTf({ data }) {
      if (typeof tf === "undefined") return null;
      const t = tf.tensor1d(data);
      const r = tf.softmax(t);
      const d = r.dataSync();
      t.dispose();
      r.dispose();
      return d;
    },
  },
  {
    name: "stats_64k",
    label: "Mean + Variance (65 536 elements)",
    setup() {
      const data = Float32Array.from({ length: 65536 }, (_, i) => i);
      return { data };
    },
    runJs({ data }) { return JS.meanVariance(data); },
    runScirs2({ data }) { return SCIRS2.meanVariance(data); },
    runTf({ data }) {
      if (typeof tf === "undefined") return null;
      const t = tf.tensor1d(data);
      const mean = t.mean().dataSync()[0];
      const variance = t.sub(mean).square().mean().dataSync()[0];
      t.dispose();
      return { mean, variance };
    },
  },
];

// ---------------------------------------------------------------------------
// Result rendering
// ---------------------------------------------------------------------------

/**
 * Render benchmark results into a DOM table.
 *
 * @param {Array<{label: string, js: object, scirs2: object|null, tf: object|null}>} results
 */
function renderResults(results) {
  const container = document.getElementById("results");
  if (!container) {
    console.table(results.map(({ label, js, scirs2: s, tf }) => ({
      Operation: label,
      "JS mean (ms)": js.mean.toFixed(3),
      "scirs2-wasm mean (ms)": s ? s.mean.toFixed(3) : "N/A",
      "scirs2 speedup": s ? `${(js.mean / s.mean).toFixed(2)}×` : "N/A",
      "TF.js mean (ms)": tf ? tf.mean.toFixed(3) : "N/A",
      "TF.js speedup": tf ? `${(js.mean / tf.mean).toFixed(2)}×` : "N/A",
    })));
    return;
  }

  const table = document.createElement("table");
  table.innerHTML = `
    <thead>
      <tr>
        <th>Operation</th>
        <th>Plain JS (ms)</th>
        <th>scirs2-wasm (ms)</th>
        <th>scirs2 speedup</th>
        <th>TF.js-WASM (ms)</th>
        <th>TF.js speedup</th>
        <th>Iterations</th>
      </tr>
    </thead>
    <tbody>
      ${results
        .map(
          ({ label, js, scirs2: s, tf }) => `
        <tr>
          <td>${label}</td>
          <td>${js.mean.toFixed(3)}</td>
          <td>${s ? s.mean.toFixed(3) : "N/A"}</td>
          <td>${s ? `${(js.mean / s.mean).toFixed(2)}×` : "N/A"}</td>
          <td>${tf ? tf.mean.toFixed(3) : "N/A"}</td>
          <td>${tf ? `${(js.mean / tf.mean).toFixed(2)}×` : "N/A"}</td>
          <td>${js.iters}</td>
        </tr>`
        )
        .join("")}
    </tbody>`;
  container.appendChild(table);
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

export async function runBenchmarks(wasmUrl = "/pkg/scirs2_wasm_bg.wasm") {
  console.log("[scirs2-bench] Starting benchmark suite");

  await initScirs2(wasmUrl);

  const results = [];
  for (const suite of SUITES) {
    console.log(`[scirs2-bench] Running: ${suite.label}`);
    const ctx = suite.setup();

    const jsStats = bench(() => suite.runJs(ctx));

    // scirs2-wasm: always available (falls back to JS when WASM not loaded)
    const scirs2Stats = suite.runScirs2
      ? bench(() => suite.runScirs2(ctx))
      : null;

    let tfStats = null;
    if (typeof tf !== "undefined" && suite.runTf) {
      tfStats = bench(() => suite.runTf(ctx));
    }

    results.push({ label: suite.label, js: jsStats, scirs2: scirs2Stats, tf: tfStats });
  }

  renderResults(results);
  console.log("[scirs2-bench] Done");
  return results;
}

// Auto-run when loaded via <script type="module"> in comparison_bench.html
if (typeof document !== "undefined") {
  document.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("run-btn");
    if (btn) {
      btn.addEventListener("click", () => runBenchmarks());
    } else {
      runBenchmarks();
    }
  });
}

// Export for Node.js test usage and ES module import from comparison_bench.html
if (typeof module !== "undefined") {
  module.exports = { runBenchmarks, bench, benchAsync, initScirs2, JS, SCIRS2, SUITES };
}
