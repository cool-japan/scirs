//! Setup guide for scirs2-wasm deployment.
//!
//! Run: `cargo run --example setup_guide -p scirs2-wasm`
//!
//! # SharedArrayBuffer Requirements
//!
//! To enable `SharedArrayBuffer` (required for `Atomics`-based synchronisation
//! between the main thread and WebWorkers), the server **must** set the
//! following HTTP response headers on every page and resource:
//!
//! ```text
//! Cross-Origin-Opener-Policy: same-origin
//! Cross-Origin-Embedder-Policy: require-corp
//! ```
//!
//! Without these headers the browser will set `SharedArrayBuffer` to
//! `undefined` even when the feature is otherwise available.  The zero-copy
//! array sharing implemented in `scirs2-wasm` (`parallel/` module) depends on
//! this API.

fn main() {
    println!("=== scirs2-wasm Setup Guide ===");
    println!();
    println!("1. Required HTTP Headers for SharedArrayBuffer / Atomics");
    println!("   --------------------------------------------------------");
    println!("   Cross-Origin-Opener-Policy: same-origin");
    println!("   Cross-Origin-Embedder-Policy: require-corp");
    println!();
    println!("   Both headers must be present on EVERY response (page, JS,");
    println!("   WASM binary, worker scripts) for Cross-Origin Isolation to");
    println!("   take effect.  Check status via:");
    println!("     window.crossOriginIsolated  // → true if configured correctly");
    println!();
    println!("2. Express.js configuration");
    println!("   -------------------------");
    println!("   const COEP_COOP_MIDDLEWARE = (req, res, next) => {{");
    println!("     res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');");
    println!("     res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');");
    println!("     next();");
    println!("   }};");
    println!("   app.use(COEP_COOP_MIDDLEWARE);");
    println!();
    println!("3. Next.js (next.config.js)");
    println!("   ------------------------");
    println!("   module.exports = {{");
    println!("     async headers() {{");
    println!("       return [{{");
    println!("         source: '/:path*',");
    println!("         headers: [");
    println!("           {{ key: 'Cross-Origin-Opener-Policy', value: 'same-origin' }},");
    println!("           {{ key: 'Cross-Origin-Embedder-Policy', value: 'require-corp' }},");
    println!("         ],");
    println!("       }}];");
    println!("     }},");
    println!("   }};");
    println!();
    println!("4. Vite / webpack-dev-server");
    println!("   -------------------------");
    println!("   // vite.config.ts");
    println!("   export default defineConfig({{");
    println!("     server: {{");
    println!("       headers: {{");
    println!("         'Cross-Origin-Opener-Policy': 'same-origin',");
    println!("         'Cross-Origin-Embedder-Policy': 'require-corp',");
    println!("       }},");
    println!("     }},");
    println!("   }});");
    println!();
    println!("5. Static file servers");
    println!("   -------------------");
    println!("   Nginx:");
    println!("     add_header Cross-Origin-Opener-Policy same-origin;");
    println!("     add_header Cross-Origin-Embedder-Policy require-corp;");
    println!();
    println!("   Caddy:");
    println!("     header Cross-Origin-Opener-Policy same-origin");
    println!("     header Cross-Origin-Embedder-Policy require-corp");
    println!();
    println!("6. Verifying the configuration");
    println!("   ----------------------------");
    println!("   Open the browser console and run:");
    println!("     console.log(window.crossOriginIsolated);");
    println!("   Expected: true");
    println!();
    println!("7. Safari notes");
    println!("   -------------");
    println!("   Safari < 15.2 does not support SharedArrayBuffer even with");
    println!("   COOP/COEP headers.  scirs2-wasm detects this and falls back to");
    println!("   standard ArrayBuffer (no zero-copy, copy overhead applies).");
    println!("   `has_simd_support()` also returns false on older Safari.");
    println!();
    println!("8. WASM module loading snippet");
    println!("   ---------------------------");
    println!("   import init, {{ WasmMatrix }} from './pkg/scirs2_wasm.js';");
    println!();
    println!("   async function run() {{");
    println!("     await init();");
    println!("     const mat = WasmMatrix.zeros(4, 4);");
    println!(
        "     console.log('WASM module loaded, matrix size:', mat.nrows(), 'x', mat.ncols());"
    );
    println!("   }}");
    println!("   run();");
    println!();
    println!("For detailed usage see: https://github.com/cool-japan/scirs/tree/master/scirs2-wasm");
}
