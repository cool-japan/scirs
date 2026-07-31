# scirs2-neural TODO

## Status: v0.6.5 (2026-07-31)

First neural-specific change since the 0.6.1 verification below (0.6.2/0.6.3/0.6.4 were all
untouched — no neural-specific changes shipped in any of them, so the crate source was unchanged
and the 0.6.1 verification carried forward as-is). This release, found via the workspace-wide
`#[ignore]`-legitimacy audit (see root `CHANGELOG.md` `[0.6.5]`), fixed a real training-correctness
bug:

- **`Lstm`** (`src/layers/recurrent/lstm.rs`), the **`Transformer`** encoder/decoder stack
  (`src/transformer/`: `model.rs`, `encoder.rs`, `decoder.rs`), and the **`BatchNorm`/`LayerNorm`**
  (`src/layers/normalization.rs`) plus **`RMSNorm`/`GroupNorm`/`InstanceNorm`/`WeightNorm`**
  (`src/layers/norm_variants.rs`) layers gained real `backward()` implementations computing
  genuine parameter/input gradients, replacing paths that previously returned a zero gradient or an
  input-shaped placeholder. Forward inference through all of these layers was already correct;
  training was not — gradient descent through an LSTM, a Transformer, or any of these
  normalization layers was silently a no-op (zero gradient) or propagated a garbage
  (wrong-but-plausibly-shaped) gradient into upstream layers, with no error or warning either way.
  `Lstm::backward` now does real backpropagation-through-time (per-gate `da_i/da_f/da_g/da_o`
  accumulation across all sixteen weight/bias parameters); `LayerNorm`/`BatchNorm`'s `backward`
  now implement the standard cached-`x_hat`/variance chain-rule derivation
  (`dgamma`/`dbeta`/`dx` from cached normalized activations); the `Transformer` encoder/decoder
  gradient now flows end-to-end through positional encoding, cross-attention, and both stacks via
  `backward_train`/`backward_with_encoder`.
- scirs2-neural itself had 0 `#[ignore]`-marked tests going into the audit (see Known Issues
  below), so this fix came from following the audit's broader review of silent-failure patterns
  across the workspace rather than from un-skipping a disabled test in this crate.

The verification below (from the 0.6.1 run, carried forward unchanged through 0.6.2/0.6.3/0.6.4)
predates this fix and has **not** been re-run as part of this docs-only update — the crate source
did change this cycle, so treat the exact pass counts as stale pending the next full `cargo
nextest` run. The qualitative claims (0 stub macros, real `metrics_integration` gating) are
unaffected by this fix and still expected to hold.

### v0.6.1 Verification (2026-07-15)

- `cargo nextest run -p scirs2-neural` (default features): **1814 tests run: 1814 passed (4 slow), 0 skipped**
- `cargo nextest run -p scirs2-neural --all-features`: **1863 tests run: 1863 passed, 0 skipped**
- `todo!()`/`unimplemented!()` macros in `src/`: 0
- `scirs2_metrics` usage: real, gated behind the non-default `metrics_integration` feature (`src/callbacks/metrics/scirs_metrics.rs::ScirsMetricsCallback`); not exercised with default features, so `cargo-udeps`' "possibly unused" flag on the `scirs2-metrics` dependency is expected with default features and not a bug
- The `scirs2-autograd` dev-dependency uses a `path = "../scirs2-autograd"` reference instead of `workspace = true` (packaging fix, no functional change)

## v0.3.3 Completed

### Attention Mechanisms
- Rotary Position Embeddings (RoPE)
- Grouped Query Attention (GQA)
- Linear attention
- Efficient attention
- Sparse attention

### Mixture of Experts
- Top-k routing with load balancing
- Expert capacity and auxiliary loss
- MoE transformer block integration

### Capsule Networks
- Dynamic routing between capsules
- Squash activation
- EM routing variant

### Spiking Neural Networks (SNN)
- Leaky Integrate-and-Fire (LIF) neurons
- Spike-Timing Dependent Plasticity (STDP)
- Rate coding and temporal coding

### Graph Neural Networks (GNN)
- Graph Convolutional Networks (GCN)
- Graph Attention Networks (GAT)
- GraphSAGE
- Graph Isomorphism Network (GIN)
- Message Passing Neural Networks
- DiffPool and SAGPool graph pooling
- Global add/mean/max pooling

### Vision Architectures
- SWIN Transformer (shifted window self-attention)
- Vision Transformer (ViT) with patch embeddings
- UNet encoder-decoder
- CLIP dual-encoder (vision + text)
- ConvNeXt (Tiny, Small, Base, Large, XLarge)
- PatchEmbedding module

### NLP / Sequence Architectures
- GPT-2 causal language model
- T5 encoder-decoder
- Full transformer (encoder + decoder)
- Positional encodings: sinusoidal, learned, RoPE, relative

### Generative Models
- Generative Adversarial Networks (GAN)
- Variational Autoencoders (VAE)
- Diffusion models (DDPM)
- Normalizing flow models
- Energy-based models

### Training Infrastructure
- Knowledge distillation (response-based and feature-based)
- Continual learning (EWC)
- Meta-learning (MAML-style)
- Contrastive learning (SimCLR, MoCo)
- Multitask learning
- Self-supervised pretraining
- Magnitude-based and structured pruning
- Post-training quantization and QAT
- DPO (Direct Preference Optimization)
- PPO for RLHF
- Reward modeling and preference data
- Gradient checkpointing
- Half-precision (FP16) training utilities

### Serialization
- Model graph serialization format
- Portable weight format (versioned)

### Compression
- Model compression utilities
- On-device optimization

## v0.4.2 / v0.4.3 — Completed

### State Space Models — Implemented in v0.4.2
- [x] Mamba/SSM architecture (selective state space model) in src/models/architectures/mamba.rs
  - `MambaConfig` with builder methods (d_model, d_state, d_conv, expand, n_layers, vocab_size, num_classes)
  - `SelectiveSSM` (S6 selective scan with ZOH discretization)
  - `MambaBlock` (Conv1D causal convolution, SiLU gating, residual connection)
  - `Mamba` full model (Layer trait impl, optional classifier head, final LayerNorm)
  - `S4Layer` (non-selective SSM with HiPPO initialization)
  - 10 tests passing (config, creation, forward, classifier, numerical stability, conv1d, SSM, S4, block)

### Wave 44 Additions (v0.4.2)
- [x] NAS module repaired and re-enabled in lib.rs (74 tests)
- [ ] CMA-ES optimizer — not found under this name anywhere in `scirs2-neural/src/` (checked `optimizers/` and full-crate grep); likely a cross-crate mix-up with `scirs2-optimize`, which does have a real CMA-ES optimizer. Unverified for this crate as of 2026-07-15.
- [x] Enhanced BPE tokenizer (`src/nlp/tokenizer.rs::BpeTokenizer`) — verified. "+ chat templates" could not be verified: no `ChatTemplate`/`chat_template` symbol found anywhere in `src/`; removed from the claim pending re-verification.
- [x] Pipeline parallelism + tensor parallelism wired into training infrastructure
- [x] Numerical validation tests (40)
- [x] Cross-crate consistency tests (16)

## v0.4.3 Status (2026-05-03)

- Workspace version bumped to 0.4.3; all crate dependency strings updated.
- ~2,786 `#[test]` functions across `src/` and `tests/`.
- cargo check + clippy: clean (no warnings, no errors)
- No-unwrap policy: PASS (no production violations)

## v0.4.0 Roadmap

### Attention — Implemented in v0.4.0
- [x] Flash Attention v2 (tiled memory-efficient attention)
- [x] Multi-query attention (MQA)
- [x] Grouped Query Attention (GQA)

### Quantization — Implemented in v0.4.0
- [x] INT4 weight-only quantization (src/quantization/int4.rs — group-quantized, nibble-packed)
- [x] INT8 activation quantization (src/quantization/int8.rs)
- [x] GPTQ-style post-training quantization (src/quantization/gptq.rs)

### Export and Interop — Implemented in v0.4.0
- [x] ONNX-like model export (src/export/onnx.rs — pure-Rust, oxicode serialization)
- [x] Weight conversion utilities for interop with other frameworks (src/export/weights.rs)

### Efficient Fine-Tuning — Implemented in v0.4.0
- [x] LoRA (Low-Rank Adaptation) (src/lora/linear.rs)
- [x] Adapter layers (src/lora/adapter.rs — bottleneck adapters with optional residual)
- [x] Prefix tuning (src/layers/prefix_tuning.rs — reparameterized prefix tokens)

### Distributed Training — Implemented in v0.4.0
- [x] Gradient compression (TopK sparsification, PowerSGD) (src/training/gradient_compression.rs)
- [x] Pipeline parallelism (src/training/pipeline_parallel.rs — GPipe FThenB + 1F1B schedules)
- [x] Tensor parallelism primitives (src/training/tensor_parallel.rs — column/row parallel + parallel embedding)

### Architecture Search — Implemented in v0.4.0, repaired in v0.4.2
- [x] Neural Architecture Search (NAS) integration (src/nas/ — ENAS, multi-objective, hardware-aware)
- [x] Differentiable NAS (DARTS/GDAS/SNAS) (src/nas/gdas.rs, src/nas/snas.rs)
- [x] NAS module re-enabled in lib.rs (Wave 44, v0.4.2): truncated source files repaired, 74 tests passing

## Known Issues / Technical Debt

- WASM target requires additional feature gating for large model weights
- `gpu.rs` (feature `gpu`) does real device detection/dispatch via `scirs2-core::gpu`; a few narrow gaps remain in `hardware/` — `accelerator.rs` has a placeholder comment for generic (non-matmul/conv) kernel execution, and `model_partitioning.rs` returns a runtime error (not a panic) for custom partitioning strategies
- `serving/packager.rs` intentionally returns "not yet implemented" errors for `PackageFormat::WebAssembly`, `AndroidAAR`, `IOSFramework`, and `PythonWheel` (only `Native`, `CSharedLibrary`, and `Docker` packaging are implemented)
- Verified 2026-07-15: no `#[ignore]`-marked tests remain anywhere in `src/`, `tests/`, `examples/`, or `benches/`; 0 `todo!()`/`unimplemented!()` macros in `src/`
