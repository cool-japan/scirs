#\!/bin/bash
sed -i 's/\.into_shape(\([^)]*\))/.into_shape_with_order(IxDyn(\&[\1]))/g' scirs2-neural/src/evaluation/metrics.rs
