#!/bin/bash

# Test script for matrix norm operations
# Usage: ./scripts/test_norms.sh [norm_type]
# Where norm_type can be: frobenius, spectral, nuclear, all

set -e

NORM_TYPE=${1:-all}

echo "Testing matrix norm operations..."

case $NORM_TYPE in
    "frobenius")
        echo "Running Frobenius norm tests..."
        cargo test --package scirs2-autograd test_frobenius_norm -- --nocapture --ignored
        ;;
    "spectral")
        echo "Running spectral norm tests..."
        cargo test --package scirs2-autograd test_spectral_norm -- --nocapture --ignored
        ;;
    "nuclear")
        echo "Running nuclear norm tests..."
        cargo test --package scirs2-autograd test_nuclear_norm -- --nocapture --ignored
        ;;
    "all")
        echo "Running all norm tests..."
        cargo test --package scirs2-autograd norm_ops_tests -- --nocapture --ignored
        ;;
    *)
        echo "Invalid norm type: $NORM_TYPE"
        echo "Valid options: frobenius, spectral, nuclear, all"
        exit 1
        ;;
esac

echo "Norm tests completed."