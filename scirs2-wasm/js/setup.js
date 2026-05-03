/**
 * scirs2-wasm server configuration helpers.
 *
 * Copy the snippets below into your server framework of choice to enable
 * Cross-Origin Isolation (required for SharedArrayBuffer / Atomics).
 *
 * Without these headers the browser sets SharedArrayBuffer to `undefined`,
 * disabling the zero-copy path in scirs2-wasm.
 */

// ---------------------------------------------------------------------------
// Required headers (must appear on EVERY response — page, JS, WASM, workers)
// ---------------------------------------------------------------------------

const COOP_COEP_HEADERS = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

// ---------------------------------------------------------------------------
// Express.js middleware
// ---------------------------------------------------------------------------

/**
 * Express middleware that sets COOP/COEP headers on every response.
 *
 * @example
 * import express from 'express';
 * import { coopCoepMiddleware } from './js/setup.js';
 *
 * const app = express();
 * app.use(coopCoepMiddleware);
 * app.use(express.static('dist'));
 * app.listen(3000);
 */
function coopCoepMiddleware(req, res, next) {
  res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
  res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
  next();
}

// ---------------------------------------------------------------------------
// Vite plugin (vite.config.ts snippet)
// ---------------------------------------------------------------------------

/**
 * Returns a Vite plugin that injects COOP/COEP headers in dev mode.
 *
 * @example
 * // vite.config.ts
 * import { coopCoepVitePlugin } from './js/setup.js';
 * export default defineConfig({ plugins: [coopCoepVitePlugin()] });
 */
function coopCoepVitePlugin() {
  return {
    name: "scirs2-coop-coep",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
        res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
        next();
      });
    },
  };
}

// ---------------------------------------------------------------------------
// Next.js headers() config (next.config.js snippet)
// ---------------------------------------------------------------------------

/**
 * Returns a Next.js `headers()` async function entry that applies COOP/COEP
 * headers to all routes.
 *
 * @example
 * // next.config.js
 * import { nextCoopCoepHeaders } from './js/setup.js';
 * module.exports = { async headers() { return [nextCoopCoepHeaders()]; } };
 */
function nextCoopCoepHeaders() {
  return {
    source: "/:path*",
    headers: [
      { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
      { key: "Cross-Origin-Embedder-Policy", value: "require-corp" },
    ],
  };
}

// ---------------------------------------------------------------------------
// Verification helper (browser only)
// ---------------------------------------------------------------------------

/**
 * Check whether Cross-Origin Isolation is active.
 *
 * Call this in the browser after loading your page.  If it returns false,
 * SharedArrayBuffer will be unavailable and zero-copy transfers will be
 * disabled.
 *
 * @returns {boolean}
 */
function isCrossOriginIsolated() {
  return typeof window !== "undefined" && window.crossOriginIsolated === true;
}

/**
 * Check whether SharedArrayBuffer is available.
 *
 * @returns {boolean}
 */
function hasSharedArrayBuffer() {
  return typeof SharedArrayBuffer !== "undefined";
}

// ---------------------------------------------------------------------------
// scirs2-wasm initialisation helper
// ---------------------------------------------------------------------------

/**
 * Initialise the scirs2-wasm module and return the exports.
 *
 * Logs a warning when SharedArrayBuffer is unavailable (COOP/COEP not set).
 *
 * @param {string} wasmUrl - URL to the compiled WASM binary (e.g. '/pkg/scirs2_wasm_bg.wasm')
 * @returns {Promise<WebAssembly.Instance>}
 *
 * @example
 * import { initScirs2Wasm } from './js/setup.js';
 * const wasm = await initScirs2Wasm('/pkg/scirs2_wasm_bg.wasm');
 */
async function initScirs2Wasm(wasmUrl) {
  if (!isCrossOriginIsolated()) {
    console.warn(
      "[scirs2-wasm] Cross-Origin Isolation is NOT active. " +
        "SharedArrayBuffer and zero-copy transfers are disabled. " +
        "Set Cross-Origin-Opener-Policy: same-origin and " +
        "Cross-Origin-Embedder-Policy: require-corp on your server."
    );
  }

  const response = await fetch(wasmUrl);
  const buffer = await response.arrayBuffer();
  const { instance } = await WebAssembly.instantiate(buffer, {});
  return instance;
}

// ---------------------------------------------------------------------------
// Static configuration strings for copy-paste into server configs
// ---------------------------------------------------------------------------

const NGINX_SNIPPET = `
# Add inside your location block or server block:
add_header Cross-Origin-Opener-Policy same-origin always;
add_header Cross-Origin-Embedder-Policy require-corp always;
`.trim();

const CADDY_SNIPPET = `
# Add inside your site block:
header Cross-Origin-Opener-Policy same-origin
header Cross-Origin-Embedder-Policy require-corp
`.trim();

const APACHE_SNIPPET = `
# Add inside <VirtualHost> or <Directory>:
Header always set Cross-Origin-Opener-Policy "same-origin"
Header always set Cross-Origin-Embedder-Policy "require-corp"
`.trim();

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

module.exports = {
  COOP_COEP_HEADERS,
  coopCoepMiddleware,
  coopCoepVitePlugin,
  nextCoopCoepHeaders,
  isCrossOriginIsolated,
  hasSharedArrayBuffer,
  initScirs2Wasm,
  NGINX_SNIPPET,
  CADDY_SNIPPET,
  APACHE_SNIPPET,
};
