#!/bin/bash
# Script to fix the enhanced_kriging.rs file by moving orphaned methods into the impl block

# Create a backup
cp /home/kitasan/github/scirs/scirs2-interpolate/src/advanced/enhanced_kriging.rs /home/kitasan/github/scirs/scirs2-interpolate/src/advanced/enhanced_kriging.rs.bak

# Find the end of the EnhancedKrigingBuilder impl block
LINE_NUMBER=$(grep -n "^}" /home/kitasan/github/scirs/scirs2-interpolate/src/advanced/enhanced_kriging.rs | head -8 | tail -1 | cut -d: -f1)

# Extract the orphaned methods until line 1394 (where the next impl block starts)
sed -n "$((LINE_NUMBER+1)),1394p" /home/kitasan/github/scirs/scirs2-interpolate/src/advanced/enhanced_kriging.rs > /tmp/orphaned_methods.txt

# Create the new file by:
# 1. Copy everything up to the closing brace of EnhancedKrigingBuilder impl
# 2. Add the orphaned methods
# 3. Add the closing brace for the impl block
# 4. Copy everything from the next impl block to the end
sed -n "1,$((LINE_NUMBER-1))p" /home/kitasan/github/scirs/scirs2-interpolate/src/advanced/enhanced_kriging.rs > /home/kitasan/github/scirs/scirs2-interpolate/src/advanced/enhanced_kriging.rs.new
cat /tmp/orphaned_methods.txt >> /home/kitasan/github/scirs/scirs2-interpolate/src/advanced/enhanced_kriging.rs.new
echo "}" >> /home/kitasan/github/scirs/scirs2-interpolate/src/advanced/enhanced_kriging.rs.new
sed -n "1395,\$p" /home/kitasan/github/scirs/scirs2-interpolate/src/advanced/enhanced_kriging.rs >> /home/kitasan/github/scirs/scirs2-interpolate/src/advanced/enhanced_kriging.rs.new

# Replace the original file with the fixed one
mv /home/kitasan/github/scirs/scirs2-interpolate/src/advanced/enhanced_kriging.rs.new /home/kitasan/github/scirs/scirs2-interpolate/src/advanced/enhanced_kriging.rs

echo "Fixed enhanced_kriging.rs by correctly moving orphaned methods into the impl block"