#!/bin/bash
# Script to fix all remaining ignored doc examples

cd /home/kitasan/work/scirs/scirs2-linalg

# List of files to process
files=(
    "src/lowrank/mod.rs"
    "src/extended_precision/mod.rs"
    "src/extended_precision/eigen.rs"
    "src/kronecker/mod.rs"
    "src/matrix_calculus/mod.rs"
    "src/specialized/banded.rs"
    "src/specialized/symmetric.rs"
    "src/specialized/tridiagonal.rs"
    "src/mixed_precision/mod.rs"
)

# Replace ```ignore with ``` in each file
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "Processing $file..."
        # Use sed to replace ```ignore with ```
        sed -i 's/```ignore/```/g' "$file"
        echo "Done with $file"
    else
        echo "File not found: $file"
    fi
done

echo "All files processed!"