# scirs2-ndimage TODO

## Status: v0.6.5 (2026-07-31)

**0.6.5:** fixed a real header-parsing bug in `mmap_io.rs`'s `load_regular_array` (the plain,
non-memory-mapped array loader, sibling to `loadimage_mmap`): it hand-parsed the file at raw byte
offset 0 with a hand-rolled little-endian f32/f64-only decode loop, ignoring the variable-length
header that `saveimage_mmap`/`create_mmap` actually prepend before the array data — silently
shifting every loaded element back by the header size instead of failing loudly. `loadimage_mmap`
itself carried the identical bug and was fixed the same way earlier in this cycle. Both now delegate
to the already header-aware `open_mmap` + `MemoryMappedArray::as_array`, which also removes the old
f32/f64-only restriction (works for any `T: Float`). New regression tests in `src/mmap_io.rs`:
`test_load_regular_array_respects_header`, `test_load_regular_array_rejects_shape_mismatch`,
`test_smart_loadimage_small_file_uses_regular_array` — all use per-cell distinct values (not a
constant fill) so a misaligned/corrupted round trip would actually be detected. See `CHANGELOG.md`
`[0.6.5]` for full detail.

**0.6.3:** Untouched by this release's fix work — no ndimage-specific changes shipped; the survey,
real gaps, and known issues below (re-verified against `src/` 2026-07-15, last reviewed for the
0.6.2 release on 2026-07-22) remain accurate for 0.6.3 since the crate source is unchanged.

**0.6.2:** the CUDA backend's kernel JIT (`backend/cuda.rs`) now compiles CUDA-C to PTX via the pure-Rust `oxicuda-nvrtc` crate (runtime `dlopen` of `libnvrtc`, zero build-time CUDA SDK dependency) instead of an embedded NVRTC loader; `libloading` is no longer a direct dependency of the crate. The `cuda` feature remains off-by-default, NVIDIA-only, and experimental. See `CHANGELOG.md` `[0.6.2]` for detail.

Source survey re-verified against `src/` 2026-07-15 (predates the 0.6.2 change above): 0 `todo!()`/`unimplemented!()` macros in `src/`; ~1765 public
`fn`/`struct`/`enum`/`trait` items. The core SciPy-equivalent surface (filters, morphology,
measurements, segmentation, interpolation, feature detection, texture/co-occurrence, medical
and hyperspectral helpers) is real, non-stub code. Freshly measured via `cargo nextest run
-p scirs2-ndimage`: default features 1170 tests run (1170 passed, 1 skipped, 0 failed);
`--all-features` 1199 tests run (1199 passed, 3 skipped, 0 failed). Both clean — 0 failures.

Real gaps found during this pass (see "v0.4.0+ Roadmap" and "Known Issues" below for detail):
- GPU kernel dispatch is scaffolded (device detection, buffer management, feature flags
  `gpu`/`cuda`/`opencl`/`metal`) but the generic execution path explicitly returns
  `NdimageError::NotImplementedError` rather than running on real hardware.
- The `src/texture/` directory (GLSZM, NGTDM, Laws' texture energy) is present on disk but is
  **not** declared anywhere in `src/lib.rs` (no `mod texture;`), so it does not compile into the
  crate today; treat those three as not-yet-shipped despite earlier `[x]` marks below.
- `src/superpixel/mod.rs` (slic_zero, seeds_superpixels, compact_watershed_superpixels) is
  likewise never declared as a module and is not part of the compiled crate.
- SLIC superpixels are 2D-grayscale only (`segmentation_advanced::superpixels_slic`); there is
  no wired-in 3D SLIC.
- The crate has no dependency on scirs2-neural; the README's former "deep learning hooks" claim
  has been corrected — feature extraction here is classical/heuristic (Gabor, HOG, SIFT-lite),
  not an actual neural-network integration point.
- **`interpolation::geometric_transform` is a confirmed no-op stub**: it validates its
  arguments and then discards the caller's mapping closure entirely, returning a copy of the
  input unchanged (`src/interpolation/transform.rs`, `#[allow(dead_code)] pub fn
  geometric_transform` with a literal `// Placeholder implementation returning a copy of the
  input`). It has no `todo!()`/`unimplemented!()` marker, so it does not show up in stub scans —
  it just silently returns the wrong answer. The "v0.3.3 Completed" claim below listing
  `geometric_transform` as done predates this finding and is inaccurate; `map_coordinates`,
  `affine_transform`, `shift`, `rotate`, and `zoom` were all individually re-verified as
  genuinely implemented and are unaffected. `rotate`'s `reshape` option is also accepted but not
  honored (output always keeps the input's shape).
- **Three more confirmed silent stubs found by the same pattern** (crate-root-reachable, no
  `todo!()`/`unimplemented!()`, doc comment says "placeholder" and the body really is one):
  - `morphology::iterate_structure` (`src/morphology/structuring.rs`) — for any `iterations > 0`
    it still just returns a copy of the input structuring element; growing/iterating the
    structure is not actually performed.
  - `peak_prominences` (`src/measurements/extrema.rs`) — returns `vec![T::one(); peaks.len()]`
    (all `1.0`) regardless of the actual signal; its doctests only assert `.len()`, so they pass
    despite the fake values.
  - `peak_widths` (`src/measurements/extrema.rs`) — returns hardcoded
    `widths=[1.0..], heights=[0.0..], left_ips=[0.0..], right_ips=[len-1..]` regardless of input;
    one of its own doctests is honest about this ("Placeholder returns 1.0") but the function doc
    and other doctests read as if it were fully functional.
  - `morphology::binary_fill_holes` (`src/morphology/binary/operations.rs`) — comment says
    "Currently not fully implemented, return a copy of the input" and the body does exactly that.
    The 2D-specific `fill_holes_2d` (`src/morphology/advanced_morph.rs`) is a real, working
    implementation (marker/mask complement + `morphological_reconstruction_by_dilation`) — use
    that for 2D instead. `binary_hit_or_miss` was also checked and is genuinely implemented for
    both the 1D (`binary_hit_or_miss_1d`, itself a stub, "not fully implemented for 1D yet") and
    2D (`binary_hit_or_miss_2d`, real) dispatch paths — only the 1D path is currently fake.
  - By contrast, `local_extrema` and `measurements::region_properties` (formerly `regionprops`)
    have similarly pessimistic "placeholder" doc comments but their bodies are genuinely real,
    working implementations — the doc comments there are just stale. Doc comments alone were not
    a reliable signal in this file; only reading each function body settled it.
  - Recommendation for the next pass: `grep -rn "Placeholder implementation\|placeholder and
    needs\|placeholder returning" src/` and check each hit's actual body (not just its comment)
    before trusting or distrusting it.

## Status: v0.4.3 Released (May 3, 2026)

All v0.4.3 features are complete and production-ready. Wave 3 stub-check resolved the
`fusion_processing` zero-cascade issue via consciousness-amplitude superposition initialisation,
restoring all multi-channel fusion paths.

## Status: v0.3.4 Released (March 18, 2026)

## v0.3.3 Completed

### Image Filtering
- Gaussian filter, gaussian_filter1d, gaussian_gradient_magnitude, gaussian_laplace
- Median filter (N-dimensional)
- Rank filters: minimum, maximum, percentile (full n-dimensional support)
- Edge detection: Sobel, Prewitt, Laplacian, Scharr, Roberts
- Bilateral filter (edge-preserving)
- Uniform (box) filter
- Generic filter with custom functions
- N-dimensional convolution (convolve, convolve1d)
- All boundary modes: reflect, nearest, wrap, mirror, constant
- Fourier filters: Gaussian, uniform, ellipsoid, shift

### Morphological Operations
- Binary erosion, dilation, opening, closing, propagation, hole filling
- Binary hit-or-miss transform
- Grayscale erosion, dilation, opening, closing
- White/black top-hat transforms, morphological gradient, Laplace
- Distance transforms: Euclidean (O(n) Felzenszwalb-Huttenlocher), city-block, chessboard
- Connected component labeling, find objects, remove small objects
- Structuring element generators: disk, square, diamond
- Skeletonization (topological thinning)

### Image Measurements
- Region statistics per label: sum, mean, variance, std, min, max
- Raw, central, normalized, and Hu moments
- Region properties: area, perimeter, centroid, bounding box, eccentricity, orientation
- Center of mass (N-dimensional)
- Local and global extrema
- Per-label histograms
- Inertia tensor

### Image Segmentation
- Thresholding: binary, Otsu, adaptive mean/Gaussian
- Standard watershed and marker-controlled watershed
- Active contours (snakes) with gradient vector flow
- Chan-Vese level set segmentation (single and multi-phase)
- Graph cut segmentation with interactive refinement (max-flow/min-cut)
- SLIC superpixels (2D and 3D)
- Atlas-based segmentation (label transfer via registration)

### Feature Detection
- Canny edge detector
- Harris corners, FAST corners
- SIFT descriptor computation
- HOG (Histogram of Oriented Gradients)
- Template matching (NCC, zero-mean NCC)
- Gabor filter bank (multi-scale, multi-orientation)
- Shape analysis (moments-based descriptors, matching)

### Geometric Interpolation
- map_coordinates (0th-5th order splines)
- affine_transform (N-dimensional)
- geometric_transform (general, custom coordinate mapping)
- shift, rotate, zoom
- spline_filter, spline_filter1d

### 3D Volume Analysis
- 3D morphology (all binary and grayscale operations)
- 3D Gaussian, Sobel, Laplacian, bilateral filters
- 3D region properties (surface area, Euler characteristic)
- Slice-by-slice processing for 3D stacks

### Medical Image Processing
- Frangi vesselness filter (multi-scale)
- Bone enhancement for CT
- Lung nodule candidate generation

### Hyperspectral Image Analysis
- Per-band filtering and morphology
- NDVI, NDWI, and custom spectral index computation
- Linear spectral unmixing
- Cloud and shadow masking
- Pan-sharpening (Brovey, IHS, PCA)

### Texture Analysis
- GLCM (gray-level co-occurrence matrix, 2D and 3D)
- Texture features from GLCM: contrast, correlation, energy, homogeneity
- Local binary patterns (LBP)
- Gabor feature maps

### Co-occurrence Matrices
- Multi-direction GLCM computation
- Haralick texture features

### Deep Feature Extraction Interface
- Hooks for forwarding arrays through external feature extractors
- Integration interface with scirs2-neural

### Performance
- SIMD-accelerated morphology and edge detection (via scirs2-core SIMD)
- Rayon parallel processing for large arrays (auto-switch at 10K elements)
- Chunked processing for images larger than RAM
- O(n) EDT via Felzenszwalb-Huttenlocher separable algorithm

## v0.4.0+ Roadmap (re-verified 2026-07-15)

### GPU-Accelerated Convolutions — scaffolded, not functionally wired
- [x] GPU backend abstraction with device detection and feature flags (`gpu`, `cuda`, `opencl`,
      `metal`; see `src/backend/`, `src/gpu_operations.rs`, `src/gpu_chunked.rs`)
- [ ] Generic GPU kernel execution: `backend::gpu_acceleration_framework` explicitly returns
      `NdimageError::NotImplementedError` instead of dispatching to real hardware
      (`src/backend/gpu_acceleration_framework.rs`) — CPU remains the only production path
- [ ] Automatic CPU/GPU dispatch based on array size and GPU availability, end-to-end
- [ ] Memory-efficient tiled GPU convolution for images larger than VRAM validated on real GPU hardware

### 4D (Temporal 3D) Imaging — substantially implemented
- [x] 4D array support for time-lapse volumetric data (`Array4D`, `src/array4d/mod.rs`)
- [x] 4D Gaussian filter, temporal differencing, max-intensity projection
      (`gaussian_filter_4d`, `diff_4d_temporal`, `max_intensity_projection_4d`)
- [x] 4D connected-component labeling and region tracking across time
      (`connected_components_4d`, `track_regions_4d` / `TrackletResult`)
- [ ] 4D optical flow (spatiotemporal motion estimation) — not found
- [ ] 4D morphological operations (spatiotemporal erosion/dilation) — not found

### Deep Segmentation Models — not implemented
- [ ] UNet-based segmentation integration (via scirs2-neural) — crate has no scirs2-neural dependency
- [ ] nnU-Net-style automatic configuration for medical segmentation
- [ ] Foundation model interface (SAM-style prompt-based segmentation)
- [ ] Transfer learning support for domain-specific segmentation

### Advanced Texture and Material Analysis
- [x] Run-length matrix (RLM) features (implemented — see `src/co_occurrence.rs`, `run_length_matrix` fn ~line 289)
- [ ] Gray-level size zone matrix (GLSZM) — **discrepancy found 2026-07-15**: `src/texture/glszm.rs`
      exists and looks complete, but `mod texture;` is never declared anywhere in `src/lib.rs` (or
      any other file), so the whole `src/texture/` directory is orphaned and not compiled into the
      crate. Unchecked until it is actually wired in.
- [ ] Neighborhood gray-tone difference matrix (NGTDM) — same orphaned-module issue (`src/texture/ngtdm.rs`)
- [ ] Laws' texture energy measures — same orphaned-module issue (`src/texture/laws.rs`); the
      previously-claimed "re-exported from src/lib.rs:168" was inaccurate — that line is a plain
      comment above `pub mod radiomics;`, not a re-export of any Laws function
- Note: basic LBP (local binary patterns) is unaffected by the above — it is real and reachable
  via `filters::advanced_simd_multi_scale_lbp`, `texture_segmentation::lbp_segment`, and
  `analysis`'s internal LBP-uniformity metric, none of which live under the orphaned `src/texture/` tree.

### Enhanced Segmentation — mostly implemented
- [x] Geodesic active contours (level set with external image energy) —
      `levelset::geodesic_active_contour`, `level_set::geodesic_contour_2d`
- [x] 3D watershed with topological constraints —
      `watershed3d::watershed_3d` + `watershed3d::topological_number_3d`
- [x] Conditional Random Fields (CRF) for label smoothing post-processing —
      `crf::dense_crf::{DenseCrf, apply_to_segmentation_2d}`
- [ ] Dedicated topology-preserving segmentation algorithm — `src/topology.rs` provides topology
      *measures* (`euler_number`, `genus`, `connected_components_count`, `hole_filling`) but no
      distinct simple-point-constrained segmentation routine was found

### Advanced Measurement — partially implemented
- [x] Graph-based region adjacency —
      `radiomics::region_adjacency::{build_rag_2d, build_rag_3d, RegionAdjacencyGraph, rag_to_adjacency_matrix}`
- [ ] Reachability and overlap between labeled regions — not found
- [ ] Multi-label volumetric statistics as a dedicated function — not confirmed beyond existing per-label 3D stats
- [ ] Radiomics feature extraction, full PyRadiomics-equivalent set — a substantial subset now
      exists (intensity statistics: `radiomics::intensity_statistics`; shape features:
      `radiomics::shape_features`; region adjacency: `radiomics::region_adjacency`; GLCM/GLRLM:
      `co_occurrence.rs`), but this has not been verified feature-by-feature against the full
      PyRadiomics reference set, and GLSZM/NGTDM/Laws remain unwired (see above)

## Known Issues

- Bilateral filter performance degrades significantly for large kernel sizes (>21x21); use a fast approximation for large kernels
- 3D watershed can be memory-intensive for large volumes; use chunked processing via the `chunked` module
- Chan-Vese segmentation convergence depends strongly on the `mu`, `lambda1`, `lambda2` parameters; automatic initialization is not yet implemented
- SLIC superpixels (`segmentation_advanced::superpixels_slic`, 2D grayscale only — there is no wired-in 3D variant) may not maintain exactly the requested number of superpixels due to boundary effects
- Atlas-based segmentation requires pre-registered atlas; no built-in registration is performed
