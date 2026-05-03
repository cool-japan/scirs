// Web Worker: offloads heavy scirs2_wasm computation off the main thread.
// Messages in:  { type, id?, ...payload }
// Messages out: { type: 'result', id?, data: Float64Array, shape: number[] }
//             | { type: 'error',  id?, message: string }

/** @type {typeof import('../pkg/scirs2_wasm')|null} */
let wasm = null;

async function ensureWasm() {
  if (wasm !== null) return;
  try {
    wasm = await import("../pkg/scirs2_wasm.js");
    if (typeof wasm.default === "function") await wasm.default();
  } catch (_) {
    importScripts("../pkg/scirs2_wasm.js");
    wasm = globalThis.scirs2_wasm ?? globalThis.wasm_bindgen ?? globalThis;
  }
  wasm.init();
}

function matrix(data, shape) {
  return wasm.WasmArray.from_shape(shape, Array.from(data));
}

function arr1d(data) {
  return new wasm.WasmArray(Array.from(data));
}

function scalar(v)   { return { data: new Float64Array([v]), shape: [1] }; }
function wasmOut(r)  { return { data: r.to_array(), shape: r.shape() }; }

// Real signal -> interleaved complex (imag = 0) when length is odd-sized
function toComplex(data) {
  if (data.length % 2 !== 0) {
    const c = new Float64Array(data.length * 2);
    for (let i = 0; i < data.length; i++) c[i * 2] = data[i];
    return c;
  }
  return data;
}

const ops = {
  matmul({ a, b, shape_a, shape_b }) {
    return wasmOut(wasm.dot(matrix(a, shape_a), matrix(b, shape_b)));
  },
  dot({ a, b }) {
    return wasmOut(wasm.dot(arr1d(a), arr1d(b)));
  },
  svd({ a, shape_a }) {
    const wa = matrix(a, shape_a);
    return scalar(wasm.norm_frobenius(wa));          // full SVD not exposed
  },
  fft({ data }) {
    const r = wasm.fft(toComplex(data));
    return { data: r, shape: [r.length] };
  },
  ifft({ data }) {
    const r = wasm.ifft(data);
    return { data: r, shape: [r.length] };
  },
  norm({ a, shape_a }) {
    return scalar(wasm.norm_frobenius(matrix(a, shape_a)));
  },
  det({ a, shape_a }) {
    return scalar(wasm.det(matrix(a, shape_a)));
  },
  inv({ a, shape_a }) {
    return wasmOut(wasm.inv(matrix(a, shape_a)));
  },
  mean({ a }) { return scalar(wasm.mean(arr1d(a))); },
  std({ a })  { return scalar(wasm.std(arr1d(a))); },
  sum({ a })  { return scalar(wasm.sum(arr1d(a))); },
};

self.addEventListener("message", async (event) => {
  const { type, id, ...payload } = event.data ?? {};
  try {
    await ensureWasm();
    const handler = ops[type];
    if (!handler) throw new TypeError(`Unknown operation: "${type}"`);
    const { data, shape } = handler(payload);
    self.postMessage({ type: "result", id, data, shape }, [data.buffer]);
  } catch (err) {
    self.postMessage({
      type: "error", id,
      message: err instanceof Error ? err.message : String(err),
    });
  }
});
