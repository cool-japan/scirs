#!/usr/bin/env node
/**
 * Example Node.js application using SciRS2-WASM
 *
 * Run with: node node_example.js
 */

const scirs2 = require('../pkg/scirs2_wasm');

async function main() {
    console.log('🦀 SciRS2-WASM Node.js Example\n');
    console.log('═'.repeat(50) + '\n');

    // Get capabilities
    const caps = scirs2.capabilities();
    console.log('System Capabilities:');
    console.log(JSON.stringify(caps, null, 2));
    console.log();

    // Array creation
    console.log('1. Array Creation');
    console.log('─'.repeat(50));
    const arr = new scirs2.WasmArray([1, 2, 3, 4, 5]);
    console.log(`Created array: length=${arr.len()}, ndim=${arr.ndim()}`);
    console.log();

    // Array operations
    console.log('2. Array Operations');
    console.log('─'.repeat(50));
    const a = new scirs2.WasmArray([1, 2, 3, 4]);
    const b = new scirs2.WasmArray([5, 6, 7, 8]);
    const sum = scirs2.add(a, b);
    const dot = scirs2.dot(a, b);
    console.log('a = [1, 2, 3, 4]');
    console.log('b = [5, 6, 7, 8]');
    console.log(`a + b = [${sum.to_array()}]`);
    console.log(`dot(a, b) = ${dot.get(0)}`);
    console.log();

    // Statistical functions
    console.log('3. Statistical Functions');
    console.log('─'.repeat(50));
    const data = new scirs2.WasmArray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    console.log('Data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]');
    console.log(`Mean:     ${scirs2.mean(data)}`);
    console.log(`Std Dev:  ${scirs2.std(data).toFixed(3)}`);
    console.log(`Median:   ${scirs2.median(data)}`);
    console.log(`Min:      ${scirs2.min(data)}`);
    console.log(`Max:      ${scirs2.max(data)}`);
    console.log();

    // Linear algebra
    console.log('4. Linear Algebra');
    console.log('─'.repeat(50));
    const matrix = scirs2.WasmArray.from_shape([2, 2], [1, 2, 3, 4]);
    console.log('Matrix:');
    console.log('[ 1  2 ]');
    console.log('[ 3  4 ]');
    console.log(`Determinant: ${scirs2.det(matrix)}`);
    console.log(`Trace:       ${scirs2.trace(matrix)}`);
    console.log();

    // Random number generation
    console.log('5. Random Number Generation');
    console.log('─'.repeat(50));
    const uniform = scirs2.random_uniform([100]);
    const normal = scirs2.random_normal([100], 0, 1);
    console.log(`Uniform random [0, 1): mean=${scirs2.mean(uniform).toFixed(4)}`);
    console.log(`Normal(0, 1):          mean=${scirs2.mean(normal).toFixed(4)}, ` +
                `std=${scirs2.std(normal).toFixed(4)}`);
    console.log();

    // Performance test
    console.log('6. Performance Benchmark');
    console.log('─'.repeat(50));
    const sizes = [100, 1000, 10000];
    for (const size of sizes) {
        const timer = new scirs2.PerformanceTimer(`size=${size}`);
        const x = scirs2.WasmArray.ones([size]);
        const y = scirs2.WasmArray.ones([size]);
        const result = scirs2.add(x, y);
        const elapsed = timer.elapsed();
        console.log(`Array addition (${size} elements): ${elapsed.toFixed(3)}ms`);
    }
    console.log();

    console.log('✓ All examples completed successfully!');
}

// Handle errors
main().catch(err => {
    console.error('Error:', err);
    process.exit(1);
});
