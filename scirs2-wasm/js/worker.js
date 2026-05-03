/**
 * scirs2-wasm Web Worker template.
 *
 * Copy this file into your project and adjust the WASM module path.  This
 * worker loads the scirs2-wasm module and dispatches computation tasks off
 * the main thread, keeping the UI responsive.
 *
 * Message protocol
 * ----------------
 * Main → Worker:
 *   { type: 'init', wasmUrl: string }
 *   { type: 'matmul', id: number, a: Float32Array, b: Float32Array, n: number }
 *   { type: 'fft',    id: number, data: Float32Array }
 *   { type: 'stats',  id: number, data: Float32Array }
 *
 * Worker → Main:
 *   { type: 'ready' }
 *   { type: 'result', id: number, result: Float32Array | object }
 *   { type: 'error',  id: number, message: string }
 */

/* global importScripts, WebAssembly */

let wasmExports = null;

// ---------------------------------------------------------------------------
// Initialisation
// ---------------------------------------------------------------------------

async function initWasm(wasmUrl) {
  try {
    const response = await fetch(wasmUrl);
    const buffer = await response.arrayBuffer();
    const { instance } = await WebAssembly.instantiate(buffer, {
      env: {
        // Minimal stub imports — real scirs2-wasm uses wasm-bindgen generated
        // glue code; this template shows the raw WebAssembly path.
      },
    });
    wasmExports = instance.exports;
    self.postMessage({ type: "ready" });
  } catch (err) {
    self.postMessage({ type: "error", id: -1, message: String(err) });
  }
}

// ---------------------------------------------------------------------------
// Task handlers
// ---------------------------------------------------------------------------

/**
 * Naive O(n³) matrix multiply used as a WASM-side benchmark.
 * In production use the WasmMatrix.matmul() binding instead.
 *
 * @param {Float32Array} a - Row-major matrix A (n×n)
 * @param {Float32Array} b - Row-major matrix B (n×n)
 * @param {number}       n - Matrix side length
 * @returns {Float32Array} - Row-major result matrix C (n×n)
 */
function localMatmul(a, b, n) {
  const c = new Float32Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < n; k++) {
      const aik = a[i * n + k];
      for (let j = 0; j < n; j++) {
        c[i * n + j] += aik * b[k * n + j];
      }
    }
  }
  return c;
}

/**
 * Compute basic statistics over an array.
 *
 * @param {Float32Array} data
 * @returns {{ mean: number, std: number, min: number, max: number }}
 */
function localStats(data) {
  let sum = 0;
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < data.length; i++) {
    sum += data[i];
    if (data[i] < min) min = data[i];
    if (data[i] > max) max = data[i];
  }
  const mean = sum / data.length;
  let variance = 0;
  for (let i = 0; i < data.length; i++) {
    variance += (data[i] - mean) ** 2;
  }
  const std = Math.sqrt(variance / data.length);
  return { mean, std, min, max };
}

/**
 * Discrete Fourier Transform (reference O(n²) implementation).
 *
 * For production use the WASM-exported FFT which uses the OxiFFT backend.
 *
 * @param {Float32Array} re - Real part input
 * @returns {{ re: Float32Array, im: Float32Array }}
 */
function localDft(re) {
  const n = re.length;
  const outRe = new Float32Array(n);
  const outIm = new Float32Array(n);
  for (let k = 0; k < n; k++) {
    for (let t = 0; t < n; t++) {
      const angle = (-2 * Math.PI * k * t) / n;
      outRe[k] += re[t] * Math.cos(angle);
      outIm[k] += re[t] * Math.sin(angle);
    }
  }
  return { re: outRe, im: outIm };
}

// ---------------------------------------------------------------------------
// Message dispatcher
// ---------------------------------------------------------------------------

self.addEventListener("message", (event) => {
  const msg = event.data;

  switch (msg.type) {
    case "init":
      initWasm(msg.wasmUrl);
      break;

    case "matmul": {
      try {
        const result = localMatmul(msg.a, msg.b, msg.n);
        self.postMessage({ type: "result", id: msg.id, result }, [
          result.buffer,
        ]);
      } catch (err) {
        self.postMessage({ type: "error", id: msg.id, message: String(err) });
      }
      break;
    }

    case "fft": {
      try {
        const { re, im } = localDft(msg.data);
        self.postMessage({ type: "result", id: msg.id, result: { re, im } }, [
          re.buffer,
          im.buffer,
        ]);
      } catch (err) {
        self.postMessage({ type: "error", id: msg.id, message: String(err) });
      }
      break;
    }

    case "stats": {
      try {
        const result = localStats(msg.data);
        self.postMessage({ type: "result", id: msg.id, result });
      } catch (err) {
        self.postMessage({ type: "error", id: msg.id, message: String(err) });
      }
      break;
    }

    default:
      self.postMessage({
        type: "error",
        id: msg.id ?? -1,
        message: `Unknown message type: ${msg.type}`,
      });
  }
});

// ---------------------------------------------------------------------------
// Main-thread helper (import in your application code)
// ---------------------------------------------------------------------------

/**
 * Create a managed scirs2 worker with promise-based task submission.
 *
 * @example
 * import { createScirs2Worker } from './js/worker.js';
 *
 * const worker = createScirs2Worker('/js/worker.js', '/pkg/scirs2_wasm_bg.wasm');
 * await worker.ready;
 *
 * const n = 64;
 * const a = new Float32Array(n * n).fill(1);
 * const b = new Float32Array(n * n).fill(2);
 * const c = await worker.matmul(a, b, n);
 * console.log('Result[0]:', c[0]); // → 128
 */
function createScirs2Worker(workerUrl, wasmUrl) {
  const worker = new Worker(workerUrl);
  const pending = new Map();
  let nextId = 0;

  const ready = new Promise((resolve, reject) => {
    worker.addEventListener(
      "message",
      (event) => {
        if (event.data.type === "ready") resolve();
        if (event.data.type === "error" && event.data.id === -1) reject(new Error(event.data.message));
      },
      { once: true }
    );
  });

  worker.addEventListener("message", (event) => {
    const { type, id, result, message } = event.data;
    if (id === -1 || id === undefined) return;
    const handlers = pending.get(id);
    if (!handlers) return;
    pending.delete(id);
    if (type === "result") handlers.resolve(result);
    else handlers.reject(new Error(message));
  });

  function dispatch(type, payload, transferable) {
    return new Promise((resolve, reject) => {
      const id = nextId++;
      pending.set(id, { resolve, reject });
      worker.postMessage({ type, id, ...payload }, transferable ?? []);
    });
  }

  // Initialise WASM inside the worker
  worker.postMessage({ type: "init", wasmUrl });

  return {
    ready,
    matmul: (a, b, n) =>
      dispatch("matmul", { a, b, n }, [a.buffer, b.buffer]),
    fft: (data) => dispatch("fft", { data }, [data.buffer]),
    stats: (data) => dispatch("stats", { data }),
    terminate: () => worker.terminate(),
  };
}

// Export for use as an ES module (bundled) or as CommonJS (Node.js tests)
if (typeof module !== "undefined") {
  module.exports = { createScirs2Worker };
}
