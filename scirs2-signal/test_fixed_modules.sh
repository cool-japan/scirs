#!/bin/bash

echo "=== Testing Fixed Modules ==="
echo "This will run tests for each module we fixed..."
echo

# Array of modules to test
modules=(
    "wvd"
    "window"
    "stft"
    "spline"
    "sswt"
    "reassigned"
    "lombscargle"
    "cqt"
)

# Run tests for each module
for module in "${modules[@]}"; do
    echo "Testing $module module..."
    cargo test --lib $module 2>&1 | grep -E "(test result:|passed|failed|test .* \.\.\. ok)" | tail -5
    echo
done