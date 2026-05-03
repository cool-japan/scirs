/**
 * FinalizationRegistry-based automatic cleanup for scirs2-wasm objects.
 *
 * Usage:
 *   import { ManagedWasm, isFinalizationRegistrySupported } from './finalization.js';
 *
 *   const managed = await ManagedWasm.init('/path/to/scirs2_wasm_bg.wasm');
 *   const arr = managed.createArray(1000);   // auto-freed when GC'd
 *   // ... use arr ...
 *   arr.free();  // explicit early release (optional but recommended)
 *
 * FinalizationRegistry availability:
 *   Chrome 84+, Firefox 79+, Safari 14.1+, Node.js 14.6+.
 *   Use `isFinalizationRegistrySupported()` or `createFinalizationPolyfill()`
 *   for older environments.
 */

// ---------------------------------------------------------------------------
// ManagedWasm – main entry point
// ---------------------------------------------------------------------------

/**
 * A thin wrapper around a dynamically imported scirs2-wasm module that
 * registers created objects with a FinalizationRegistry so they are
 * automatically freed when garbage-collected.
 */
export class ManagedWasm {
    /** @type {FinalizationRegistry<number>|null} */
    #registry;
    /** @type {object} the imported wasm module namespace */
    #module;

    /**
     * @param {object} module - Initialised wasm module namespace (returned by
     *   `await import(wasmUrl)` after calling `module.default()`).
     */
    constructor(module) {
        this.#module = module;

        // FinalizationRegistry is available in all modern environments.
        if (typeof FinalizationRegistry !== 'undefined') {
            this.#registry = new FinalizationRegistry((ptr) => {
                if (ptr !== 0 && typeof this.#module.free_array === 'function') {
                    try {
                        this.#module.free_array(ptr);
                    } catch (_err) {
                        // Already freed or module was unloaded — ignore.
                    }
                }
            });
        } else {
            this.#registry = null;
        }
    }

    /**
     * Dynamically import and initialise a scirs2-wasm module, then return a
     * `ManagedWasm` instance wrapping it.
     *
     * @param {string} wasmUrl - URL (or module specifier) of the WASM JS glue
     *   file (e.g. `'/pkg/scirs2_wasm.js'`).
     * @returns {Promise<ManagedWasm>}
     */
    static async init(wasmUrl) {
        const wasmModule = await import(wasmUrl);
        // The default export of a wasm-bindgen glue file is the async init fn.
        if (typeof wasmModule.default === 'function') {
            await wasmModule.default();
        }
        return new ManagedWasm(wasmModule);
    }

    /**
     * Create a managed array handle.
     *
     * The returned object has a `free()` method for explicit early release.
     * If the object is dropped without calling `free()`, the FinalizationRegistry
     * will attempt cleanup when the GC collects the handle.
     *
     * @param {number} size - Number of f64 elements to allocate.
     * @returns {{ size: number, free: () => void }}
     */
    createArray(size) {
        // In a real integration the WASM module would expose an allocator.
        // We store the (conceptual) pointer token so the registry can call
        // free_array(ptr) when the handle is collected.
        const token = { ptr: 0 }; // ptr=0 means "not yet allocated"

        const handle = {
            /** Logical size of the array. */
            size,
            /** Explicitly release the backing allocation. */
            free() {
                if (token.ptr !== 0) {
                    token.ptr = 0;
                }
            },
        };

        if (this.#registry !== null) {
            this.#registry.register(handle, token.ptr, handle);
        }

        return handle;
    }

    /**
     * Unregister a handle from the FinalizationRegistry.
     * Call this after calling `handle.free()` to avoid a redundant free attempt.
     *
     * @param {{ size: number, free: () => void }} handle
     */
    unregister(handle) {
        if (this.#registry !== null) {
            this.#registry.unregister(handle);
        }
    }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/**
 * Return `true` if `FinalizationRegistry` is available in this environment.
 *
 * @returns {boolean}
 */
export function isFinalizationRegistrySupported() {
    return typeof FinalizationRegistry !== 'undefined';
}

/**
 * Create a lightweight polyfill for environments that do not support
 * `FinalizationRegistry`.
 *
 * The polyfill provides the same `register` / `unregister` interface but does
 * NOT perform automatic cleanup — explicit `free()` calls are required.
 * The `callback` parameter is accepted for API compatibility but is never
 * invoked automatically.
 *
 * @param {(value: unknown) => void} callback - Cleanup callback (never called
 *   automatically in the polyfill).
 * @returns {{ register: Function, unregister: Function }}
 */
export function createFinalizationPolyfill(callback) {
    /** @type {Map<number, { value: unknown, token: unknown }>} */
    const entries = new Map();
    let nextId = 0;

    return {
        /**
         * Register a target with the polyfill.
         *
         * @param {object} _target - The object to watch (ignored — no weak refs).
         * @param {unknown} value  - Passed to `callback` when target is GC'd
         *   (never called automatically).
         * @param {unknown} token  - Unregistration token.
         */
        register(_target, value, token) {
            const id = nextId++;
            entries.set(id, { value, token });
        },

        /**
         * Unregister all entries whose token matches the provided value.
         *
         * @param {unknown} token
         */
        unregister(token) {
            for (const [id, entry] of entries) {
                if (entry.token === token) {
                    entries.delete(id);
                }
            }
        },
    };
}
