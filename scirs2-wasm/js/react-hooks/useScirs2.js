/**
 * React hooks for scirs2-wasm integration.
 *
 * @module @scirs2/react-hooks
 *
 * Usage:
 *   import { useScirs2, useScirs2Compute, useScirs2Array } from '@scirs2/react-hooks';
 *
 *   function MyComponent() {
 *     const { wasm, isLoading, error } = useScirs2('/pkg/scirs2_wasm.js');
 *
 *     const [result, compute, isComputing] = useScirs2Compute(wasm, 'matrix_multiply');
 *
 *     const array = useScirs2Array(wasm, 256);
 *
 *     if (isLoading) return <p>Loading WASM…</p>;
 *     if (error)     return <p>Error: {error.message}</p>;
 *     return <button onClick={() => compute([1, 2, 3])}>Run</button>;
 *   }
 *
 * Peer dependencies: react >= 18.0.0
 */

import { useState, useEffect, useCallback } from 'react';

// ---------------------------------------------------------------------------
// useScirs2 — load and initialise the WASM module
// ---------------------------------------------------------------------------

/**
 * Load and initialise a scirs2-wasm module asynchronously.
 *
 * The hook handles the full lifecycle: loading, initialisation, cancellation
 * on unmount, and error capture.
 *
 * @param {string} wasmUrl - URL or module specifier for the wasm-bindgen JS
 *   glue file (e.g. `'/pkg/scirs2_wasm.js'`).
 * @returns {{ wasm: object|null, isLoading: boolean, error: Error|null }}
 */
export function useScirs2(wasmUrl) {
    const [wasm, setWasm] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        let cancelled = false;

        async function load() {
            try {
                const mod = await import(wasmUrl);
                // wasm-bindgen glue exposes an async default export for init.
                if (typeof mod.default === 'function') {
                    await mod.default();
                }
                if (!cancelled) {
                    setWasm(mod);
                    setIsLoading(false);
                }
            } catch (err) {
                if (!cancelled) {
                    setError(err instanceof Error ? err : new Error(String(err)));
                    setIsLoading(false);
                }
            }
        }

        load();

        return () => {
            cancelled = true;
        };
    }, [wasmUrl]);

    return { wasm, isLoading, error };
}

// ---------------------------------------------------------------------------
// useScirs2Compute — run a WASM function with loading state
// ---------------------------------------------------------------------------

/**
 * Wrap a named scirs2-wasm function so that React components can trigger it
 * as an action and observe the result.
 *
 * @template T
 * @param {object|null} wasm - The initialised wasm module (from `useScirs2`).
 * @param {string} fnName    - Name of the exported wasm function to call.
 * @returns {[T|null, (...args: unknown[]) => Promise<void>, boolean]}
 *   A tuple of `[result, compute, isComputing]`.
 */
export function useScirs2Compute(wasm, fnName) {
    const [result, setResult] = useState(null);
    const [isComputing, setIsComputing] = useState(false);

    const compute = useCallback(
        async (...args) => {
            if (wasm === null || typeof wasm[fnName] !== 'function') return;
            setIsComputing(true);
            try {
                const res = await Promise.resolve(wasm[fnName](...args));
                setResult(res);
            } catch (_err) {
                // Errors are surfaced to the caller if they wrap in try/catch.
                throw _err;
            } finally {
                setIsComputing(false);
            }
        },
        [wasm, fnName],
    );

    return [result, compute, isComputing];
}

// ---------------------------------------------------------------------------
// useScirs2Array — manage a typed array backed by WASM memory
// ---------------------------------------------------------------------------

/**
 * Allocate a `Float64Array` backed by WASM linear memory of `size` `f64`
 * elements.  When the wasm module exports `alloc_f64_array` / `free_f64_array`
 * the buffer lives inside the WASM heap, enabling zero-copy access from Rust.
 * Falls back to a plain JS `Float64Array` when the exports are absent (e.g.
 * during testing or when using an older build).
 *
 * The buffer is re-created whenever `wasm` or `size` changes.  On unmount (or
 * before the next effect run) WASM-allocated memory is freed via
 * `wasm.free_f64_array(ptr, size)` to prevent leaks.
 *
 * @param {object|null} wasm - The initialised wasm module (from `useScirs2`).
 * @param {number} size      - Number of `f64` elements to allocate.
 * @returns {{ size: number, data: Float64Array, ptr: number }|null}
 *   `ptr` is the raw WASM heap offset (0 for JS-only fallback allocations).
 */
export function useScirs2Array(wasm, size) {
    const [array, setArray] = useState(null);

    useEffect(() => {
        if (wasm === null) {
            setArray(null);
            return;
        }

        // --- allocation ---------------------------------------------------
        let ptr = 0;
        let buf;

        if (typeof wasm.alloc_f64_array === 'function') {
            // Allocate inside the WASM heap for true zero-copy access.
            ptr = wasm.alloc_f64_array(size);
            if (typeof wasm.view_f64_array === 'function') {
                // Use the Rust-provided view helper when available.
                buf = wasm.view_f64_array(ptr, size);
            } else if (wasm.memory && wasm.memory.buffer) {
                // Construct a typed-array view directly from WebAssembly.Memory.
                buf = new Float64Array(wasm.memory.buffer, ptr, size);
            } else {
                // Should not happen in a correctly-linked wasm-bindgen build,
                // but fall back gracefully rather than crashing.
                buf = new Float64Array(size);
                ptr = 0;
            }
        } else {
            // Fallback: ordinary JS heap allocation (no WASM integration).
            buf = new Float64Array(size);
        }

        setArray({ size, data: buf, ptr });

        // --- cleanup ------------------------------------------------------
        return () => {
            if (ptr !== 0 && typeof wasm.free_f64_array === 'function') {
                wasm.free_f64_array(ptr, size);
            }
            setArray(null);
        };
    }, [wasm, size]);

    return array;
}
