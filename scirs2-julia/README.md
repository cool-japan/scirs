# scirs2-julia

[![Alpha](https://img.shields.io/badge/status-alpha-orange)]()
[![Version](https://img.shields.io/badge/version-0.4.4-green)]()
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)]()

**Julia JLL-compatible shared library (`libscirs2_julia`) for the SciRS2 scientific computing library.**

This crate builds a `cdylib` / `staticlib` that exposes a stable C ABI designed for consumption by the Julia [`SciRS2_jll`](https://github.com/cool-japan/scirs) package. It is a versioned wrapper around `scirs2-core/ffi` and provides:

- **Version introspection** — `scirs2_julia_version()` and `scirs2_julia_abi_version()` so Julia can verify compatibility at runtime.
- **Capability flags** — `scirs2_julia_capabilities()` encodes which sub-modules are compiled in (linalg, stats, fft, optimize).
- **Re-exported `sci_*` symbols** from `scirs2-core/ffi` so the existing `SciRS2.jl` wrapper works without modification.
- **Batch helper functions** that reduce `ccall` round-trip overhead for hot paths: `scirs2_batch_mean`, `scirs2_batch_variance`, `scirs2_batch_std`, `scirs2_batch_minmax`, `scirs2_batch_dot`, `scirs2_batch_norm_l2`, `scirs2_batch_norm_l1`, `scirs2_batch_sum`, `scirs2_batch_scale`, `scirs2_batch_sort_ascending`, `scirs2_batch_cumsum`, `scirs2_batch_correlation`.

## ABI Safety

All `extern "C"` functions uphold the following invariants:

1. No panics cross the FFI boundary (`std::panic::catch_unwind`).
2. All pointers are validated before dereferencing.
3. Memory allocated by this library must be freed by this library (use `scirs2_julia_free_error`).

The `ABI_VERSION` constant is incremented whenever breaking changes are made to the exported symbols. Julia's `SciRS2_jll` package checks this at load time.

## Features

| Feature    | Description                              |
|------------|------------------------------------------|
| `linalg`   | Linear algebra batch operations          |
| `stats`    | Statistical batch helpers                |
| `fft`      | FFT support (via `scirs2-core/ffi`)      |
| `optimize` | Optimization helpers                     |
| `full`     | All of the above (JLL packaging target)  |

Default features: `linalg`, `stats`, `fft`, `optimize`.

## Julia JLL Metadata

| Field         | Value                    |
|---------------|--------------------------|
| JLL name      | `SciRS2_jll`             |
| Minimum Julia | 1.8                      |
| Soname        | `libscirs2_julia.so.0`   |
| BB product    | `libscirs2_julia`        |

## Usage from Julia

```julia
using SciRS2_jll

# Version check
ver = unsafe_string(ccall((:scirs2_julia_version, libscirs2_julia), Cstring, ()))
abi = ccall((:scirs2_julia_abi_version, libscirs2_julia), Cuint, ())
@assert abi == 1  # expected ABI version

# Batch mean example
data = Float64[1.0, 2.0, 3.0, 4.0, 5.0]
out  = Ref{Float64}(0.0)
GC.@preserve data begin
    ccall((:scirs2_batch_mean, libscirs2_julia),
          Cvoid, (Ptr{Float64}, Csize_t, Ptr{Float64}),
          pointer(data), length(data), out)
end
println("mean = ", out[])  # 3.0
```

## Building

```bash
# Build the shared library
cargo build --release -p scirs2-julia

# Build with all features (JLL packaging mode)
cargo build --release -p scirs2-julia --features full
```

The resulting `.so` / `.dylib` / `.dll` will be located in `target/release/`.

## Documentation

Full API documentation: <https://docs.rs/scirs2-julia>

SciRS2 main project: <https://github.com/cool-japan/scirs>

## License

Licensed under [Apache-2.0](LICENSE).

Copyright © COOLJAPAN OU (Team Kitasan)
