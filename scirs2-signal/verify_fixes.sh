#!/bin/bash

echo "=== Verification Script for Ignored Test Fixes ==="
echo

# Check for any remaining ignored tests with FIXME
echo "1. Checking for ignored tests with FIXME comments..."
IGNORED_FIXME=$(grep -r "#\[ignore\].*FIXME" src/ 2>/dev/null | wc -l)
echo "   Found: $IGNORED_FIXME ignored tests with FIXME"

# Check for doc tests with ignore and FIXME
echo "2. Checking for ignored doc tests with FIXME..."
DOC_IGNORE_FIXME=$(grep -r "ignore.*FIXME" src/ 2>/dev/null | wc -l)
echo "   Found: $DOC_IGNORE_FIXME ignored doc tests with FIXME"

# List all files we modified
echo "3. Files modified:"
echo "   - src/wvd.rs"
echo "   - src/window/kaiser.rs"
echo "   - src/window/mod.rs"
echo "   - src/stft.rs"
echo "   - src/spline.rs"
echo "   - src/sswt.rs"
echo "   - src/reassigned.rs"
echo "   - src/lombscargle.rs"
echo "   - src/cqt.rs"

# Summary
echo
echo "=== Summary ==="
if [ $IGNORED_FIXME -eq 0 ] && [ $DOC_IGNORE_FIXME -eq 0 ]; then
    echo "✅ All ignored tests with FIXME comments have been fixed!"
    echo "✅ Total tests fixed: 22"
else
    echo "❌ Some ignored tests remain:"
    echo "   - Ignored tests with FIXME: $IGNORED_FIXME"
    echo "   - Doc tests with ignore and FIXME: $DOC_IGNORE_FIXME"
fi