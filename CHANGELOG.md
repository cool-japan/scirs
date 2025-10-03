# Changelog

All notable changes to the SciRS2 project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-RC.1] - 2025-10-03

### 🎯 Release Candidate 1 - Major Architectural Milestone

This release represents a complete architectural transformation of the SciRS2 ecosystem, achieving 100% compliance with the SciRS2 POLICY across all 23 crates.

### Added

#### Architecture & Infrastructure
- **SciRS2 POLICY Implementation**: Complete migration to centralized dependency management
  - Only `scirs2-core` now has direct external dependencies
  - All 23 crates route through `scirs2-core` abstractions
  - Unified `array` and `random` features in scirs2-core
  - Enhanced type safety and API consistency

#### scirs2-core Enhancements
- **Unified Array Feature**: Complete ndarray ecosystem integration
  - `ndarray-rand`, `ndarray-linalg`, `ndarray-stats`, `ndarray-npy`
  - Full macro support: `array!`, `s!`, `azip!`
- **Complete Random Module**: All rand/rand_distr distributions
  - Added `WeightedIndex` re-export
  - New `rand_prelude` module with essential traits
  - `IndexedRandom`, `IndexedMutRandom` support
- **Numeric Traits Module**: Complete num-traits/num-complex abstraction
  - `Float`, `Zero`, `One`, `NumCast`, `Complex64`, etc.
  - Consistent trait bounds across all modules

#### Documentation
- **SCIRS2_POLICY.md**: Comprehensive policy documentation
  - Architectural principles and benefits
  - Migration guidelines and examples
  - Technical policies for SIMD, GPU, parallel operations
- **Migration Guides**:
  - `/tmp/scirs2_policy_migration_rc1_summary.md` - Detailed migration report
  - `/tmp/scirs2_policy_quick_reference.md` - Developer quick reference
- **CLAUDE.md Updates**: POLICY-compliant code examples and patterns
- **README.md**: RC.1 release status and highlights

### Changed

#### All Non-Core Crates (22 crates)
- **Dependency Migration**: Removed direct external dependencies
  - Removed: `ndarray`, `num-traits`, `num-complex`, `rand`, `rand_distr`
  - Added: `scirs2-core` with `array` and `random` features
- **Import Standardization**: ~600+ import statements updated
  - `use ndarray::*` → `use scirs2_core::ndarray::*`
  - `use num_traits::*` → `use scirs2_core::numeric::*`
  - `use rand::*` → `use scirs2_core::random::*`

#### Specific Crate Updates

**scirs2-linalg** (100% compliant)
- All array operations through scirs2-core
- Numeric traits from scirs2_core::numeric
- BLAS/LAPACK integration maintained

**scirs2-stats** (100% compliant)
- Complete random module migration
- All distributions through scirs2_core::random
- Statistical functions using core numeric traits

**scirs2-fft** (100% compliant)
- Fixed `accuracy_comparison.rs` Zero trait import
- Complex number operations through scirs2-core

**scirs2-signal** (100% compliant)
- NumCast type inference fixes
- Waveform generation using core abstractions

**scirs2-cluster** (100% compliant)
- WeightedIndex integration
- IndexedRandom for sampling operations

**scirs2-neural** (100% compliant)
- Uniform::new() error handling improvements
- Distribution trait scope fixes

**scirs2-transform** (100% compliant)
- Array2::random manual generation workaround
- NumCast pattern updates

**scirs2-metrics** (100% compliant)
- Real → Float trait replacement (15+ files)
- Consistent numeric abstractions

**scirs2-vision** (100% compliant)
- IndexedMutRandom trait integration

**scirs2-series** (100% compliant)
- ThreadRng type fixes
- GPU acceleration module updates

**All other crates**: scirs2-ndimage, scirs2-optimize, scirs2-integrate, scirs2-interpolate, scirs2-special, scirs2-sparse, scirs2-spatial, scirs2-io, scirs2-datasets, scirs2-autograd, scirs2-graph, scirs2-text, scirs2 (main)

### Fixed

#### Compilation Issues
- **NumCast Type Inference** (E0790): Fully qualified syntax for type conversion
- **Distribution Trait Scope** (E0599): Explicit Distribution imports
- **Uniform::new() Error Handling** (E0599): Proper Result handling with map_err
- **IndexedRandom Migration** (E0599): SliceRandom → IndexedRandom transition
- **ThreadRng Type Mismatches** (E0308): Explicit ThreadRng imports
- **WeightedIndex Availability** (E0432): Re-export in scirs2-core
- **Real Trait References** (E0405): Replaced with Float trait

### Technical Details

#### Build Verification
- ✅ Full workspace build: `cargo build --all` - **ZERO errors, ZERO warnings**
- ✅ POLICY compliance: Only scirs2-core has direct external deps
- ✅ Build time: 49.77s (optimized compilation)
- ✅ Files modified: 467 files across 23 crates

#### Migration Statistics
- **Total dependency removals**: ~100+ direct external dependencies
- **Import statement updates**: ~600+ replacements
- **Cargo.toml modifications**: 23 files
- **Source file updates**: 440+ Rust files

### Benefits Achieved

#### Maintainability
- **Single Source of Truth**: All external dependency versions in scirs2-core
- **Simplified Upgrades**: External library updates only affect core
- **Reduced Coupling**: Clear separation between modules

#### Type Safety
- **Consistent Trait Bounds**: Unified numeric and random abstractions
- **Compile-time Verification**: Type mismatches caught at build time
- **No External Type Mixing**: Prevents version conflicts

#### Performance
- **Optimization Opportunities**: Core can optimize all external library usage
- **SIMD/Parallel Abstractions**: Applied uniformly across crates
- **Memory Management**: Centralized strategies

### Documentation Updates
- SCIRS2_POLICY.md: Phase 1-4 completion status
- CLAUDE.md: RC.1 status, POLICY-compliant examples
- README.md: Release highlights and new features
- Migration guides in `/tmp/`

### Known Issues
- PyTorch compatibility in scirs2-transform requires version 2.0.0 or environment variables
  - Workaround documented in CLAUDE.md

### Notes for Developers
- **MANDATORY**: All new code must follow SCIRS2 POLICY
- **Quick Reference**: `/tmp/scirs2_policy_quick_reference.md`
- **Examples**: See any of the 23 migrated crates for patterns
- **CI Checks**: Automated policy enforcement planned for RC.2

---

## [0.1.0-beta.4] - 2025-10-01

### Changed
- scirs2-core enhancements for POLICY preparation
- Policy documentation initial draft
- Dependency analysis and planning

### Notes
- Beta.4 was the preparation phase for POLICY migration
- Full implementation completed in RC.1

---

## [0.1.0-beta.3] - Previous Release

[Earlier releases documented in git history]

---

## Migration Guide

### From Beta.4 to RC.1

If you have custom code using SciRS2:

1. **Update Cargo.toml**:
   ```toml
   # Remove these
   # ndarray = "..."
   # num-traits = "..."
   # rand = "..."

   # Update this
   scirs2-core = { workspace = true, features = ["array", "random"] }
   ```

2. **Update imports**:
   ```rust
   // Change
   use ndarray::{Array1, Array2};
   use num_traits::Float;
   use rand::thread_rng;

   // To
   use scirs2_core::ndarray::{Array1, Array2};
   use scirs2_core::numeric::Float;
   use scirs2_core::random::thread_rng;
   ```

3. **Build and fix**:
   - Run `cargo build`
   - Fix trait scope issues (add explicit Distribution import if needed)
   - Fix type inference (use fully qualified syntax for NumCast if needed)

See `/tmp/scirs2_policy_quick_reference.md` for comprehensive migration patterns.

---

*For detailed technical information, see [SCIRS2_POLICY.md](SCIRS2_POLICY.md)*
