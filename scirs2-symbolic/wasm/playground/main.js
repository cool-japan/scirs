/**
 * SciRS2 Symbolic CAS Playground — main.js
 *
 * Loads the WASM module (built with `wasm-pack build --target web` into
 * ./pkg/) and wires all UI controls to their respective WASM API functions.
 *
 * The WASM module exposes:
 *   wasm_canonicalize(expr: string): string
 *   wasm_simplify(expr: string): string
 *   wasm_grad(expr: string, wrt: number): string
 *   wasm_eval(expr: string, bindings_json: string): string
 *   wasm_is_identity(expr1: string, expr2: string): string
 */

/** Parse a comma-separated list of floats into a JSON array string. */
function parseBindings(raw) {
  const parts = raw.split(',').map(s => s.trim()).filter(s => s.length > 0);
  const nums = parts.map(s => {
    const n = Number(s);
    if (isNaN(n)) throw new Error(`"${s}" is not a valid number`);
    return n;
  });
  return JSON.stringify(nums);
}

/** Apply an "error" CSS class when the result starts with "Error". */
function setResult(el, text) {
  el.textContent = text;
  if (text.startsWith('Error')) {
    el.classList.add('error');
  } else {
    el.classList.remove('error');
  }
}

async function initPlayground() {
  const banner = document.getElementById('error-banner');

  let wasm;
  try {
    // The WASM package is expected at ./pkg/ (output of wasm-pack build --target web).
    wasm = await import('./pkg/scirs2_symbolic_wasm.js');
    await wasm.default();          // Initialise the WASM module.
  } catch (err) {
    banner.textContent =
      'WASM module not loaded. ' +
      'Build first with: wasm-pack build --target web ' +
      '(from scirs2-symbolic/wasm/). ' +
      'Then serve this directory with a local HTTP server. ' +
      `Error: ${err}`;
    banner.style.display = 'block';
    return;
  }

  // ------------------------------------------------------------------
  // Canonicalize
  // ------------------------------------------------------------------
  document.getElementById('canonicalize-btn').addEventListener('click', () => {
    const expr = document.getElementById('expr-input').value.trim();
    const out = document.getElementById('canonicalize-result');
    if (!expr) { setResult(out, '(empty input)'); return; }
    setResult(out, wasm.wasm_canonicalize(expr));
  });

  // ------------------------------------------------------------------
  // Simplify
  // ------------------------------------------------------------------
  document.getElementById('simplify-btn').addEventListener('click', () => {
    const expr = document.getElementById('simplify-input').value.trim();
    const out = document.getElementById('simplify-result');
    if (!expr) { setResult(out, '(empty input)'); return; }
    setResult(out, wasm.wasm_simplify(expr));
  });

  // ------------------------------------------------------------------
  // Gradient
  // ------------------------------------------------------------------
  document.getElementById('grad-btn').addEventListener('click', () => {
    const expr = document.getElementById('grad-expr-input').value.trim();
    const varIdx = parseInt(document.getElementById('grad-var-input').value, 10);
    const out = document.getElementById('grad-result');
    if (!expr) { setResult(out, '(empty input)'); return; }
    if (isNaN(varIdx) || varIdx < 0) {
      setResult(out, 'Error: variable index must be a non-negative integer');
      return;
    }
    setResult(out, wasm.wasm_grad(expr, varIdx));
  });

  // ------------------------------------------------------------------
  // Evaluate
  // ------------------------------------------------------------------
  document.getElementById('eval-btn').addEventListener('click', () => {
    const expr = document.getElementById('eval-expr-input').value.trim();
    const rawBindings = document.getElementById('eval-bindings-input').value.trim();
    const out = document.getElementById('eval-result');
    if (!expr) { setResult(out, '(empty input)'); return; }

    let bindingsJson;
    try {
      bindingsJson = parseBindings(rawBindings);
    } catch (e) {
      setResult(out, `Error: ${e.message}`);
      return;
    }
    setResult(out, wasm.wasm_eval(expr, bindingsJson));
  });

  // ------------------------------------------------------------------
  // Identity Check
  // ------------------------------------------------------------------
  document.getElementById('id-btn').addEventListener('click', () => {
    const expr1 = document.getElementById('id-expr1-input').value.trim();
    const expr2 = document.getElementById('id-expr2-input').value.trim();
    const out = document.getElementById('id-result');
    if (!expr1 || !expr2) { setResult(out, '(empty input)'); return; }
    const result = wasm.wasm_is_identity(expr1, expr2);
    setResult(out, result);
  });

  // Enable pressing Enter in any text input to trigger its button.
  const inputButtonMap = [
    ['expr-input', 'canonicalize-btn'],
    ['simplify-input', 'simplify-btn'],
    ['grad-expr-input', 'grad-btn'],
    ['eval-expr-input', 'eval-btn'],
    ['id-expr1-input', 'id-btn'],
    ['id-expr2-input', 'id-btn'],
  ];
  for (const [inputId, btnId] of inputButtonMap) {
    document.getElementById(inputId).addEventListener('keydown', (e) => {
      if (e.key === 'Enter') document.getElementById(btnId).click();
    });
  }
}

initPlayground();
