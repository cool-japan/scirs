# ⚠️ MOVED: scirs2-optim → OptiRS

This module has been **moved to an independent project** as of **SciRS2 v0.1.0-beta.2**.

## 🔄 Migration Guide

### Old (scirs2-optim)
```toml
[dependencies]
scirs2-optim = "0.1.0-beta.1"  # ❌ No longer available
```

### New (OptiRS)
```toml
[dependencies]
optirs = "0.1.0"  # ✅ New independent project
# Or specific modules:
optirs-core = "0.1.0"
optirs-gpu = { version = "0.1.0", features = ["cuda"] }
optirs-learned = "0.1.0"
optirs-nas = "0.1.0"
```

## 📍 New Location

**OptiRS Project**: [`https://github.com/cool-japan/optirs`](https://github.com/cool-japan/optirs)

## 🎯 Why the Move?

- **Focus**: Dedicated optimization research and development
- **Performance**: Specialized hardware acceleration (GPU/TPU)
- **Modularity**: Use only the components you need
- **Independence**: Faster release cycles for ML optimization advances

## 📦 What's in OptiRS?

- **`optirs-core`**: Basic optimizers (SGD, Adam, etc.), schedulers, regularizers
- **`optirs-gpu`**: Multi-GPU optimization, CUDA/Metal/OpenCL/WebGPU
- **`optirs-tpu`**: TPU pod coordination and XLA integration
- **`optirs-learned`**: Transformer/LSTM optimizers, meta-learning
- **`optirs-nas`**: Neural Architecture Search algorithms
- **`optirs-bench`**: Performance analysis and benchmarking tools

## 🚀 Quick Start with OptiRS

```rust
use optirs::prelude::*;
use ndarray::Array2;

// Create an optimizer (same API as before)
let optimizer = Adam::new(0.001)?;

// Use with your gradients
let gradients = Array2::zeros((10, 10));
let updated_params = optimizer.step(&gradients)?;
```

## 📚 Documentation

Full documentation and examples are available in the OptiRS repository.

---

**For SciRS2 core scientific computing**, continue using:
```toml
[dependencies]
scirs2 = "0.1.0-rc.1"  # Scientific computing without optimization
```