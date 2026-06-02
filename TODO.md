# SciRS2 Development Roadmap

**Current Version**: 0.5.0
**Status**: Production Ready — ~36,082 tests passing (Wave 77 baseline, 2026-05-25)
**Scale**: ~4.2M lines total, ~3.87M Rust SLoC, ~8,129 source files, ~29 workspace crates
**Last Updated**: 2026-06-02

This document tracks the development roadmap for SciRS2. Completed items in v0.3.4 are documented here for historical reference; the active roadmap is the v0.4.0 section.

---

## Module Reference

**Core Scientific Computing**
- [scirs2-core](./scirs2-core/TODO.md): Core utilities and abstractions (mandatory base for all modules)
- [scirs2-linalg](./scirs2-linalg/TODO.md): Linear algebra with iterative solvers, tensor decompositions, matrix functions
- [scirs2-stats](./scirs2-stats/TODO.md): Statistical distributions, Bayesian inference, survival analysis, copulas
- [scirs2-optimize](./scirs2-optimize/TODO.md): Scientific optimization — MIP/SDP/SOCP, metaheuristics, Bayesian BO, NSGA-III
- [scirs2-integrate](./scirs2-integrate/TODO.md): Numerical integration and ODE/PDE/SDE solvers (LBM, DG, phase-field, BEM)
- [scirs2-interpolate](./scirs2-interpolate/TODO.md): Interpolation — RBF, PCHIP, MLS, kriging, spherical harmonics
- [scirs2-special](./scirs2-special/TODO.md): Special functions — Mathieu, Coulomb, Wigner, Jacobi theta, Fox H-function
- [scirs2-fft](./scirs2-fft/TODO.md): FFT and spectral — sparse FFT, Prony, MUSIC, Lomb-Scargle, NTT
- [scirs2-signal](./scirs2-signal/TODO.md): Signal processing — CFAR radar, Kalman/EKF/UKF, compressed sensing, MFCC
- [scirs2-sparse](./scirs2-sparse/TODO.md): Sparse matrices — LOBPCG/IRAM, AMG, BCSR/ELLPACK, recycled Krylov
- [scirs2-spatial](./scirs2-spatial/TODO.md): Spatial — R*-Tree, Fortune's Voronoi, geodata, trajectory analysis

**Advanced Modules**
- [scirs2-cluster](./scirs2-cluster/TODO.md): Clustering — GMM, SOM, HDBSCAN, Dirichlet process, biclustering, topological
- [scirs2-ndimage](./scirs2-ndimage/TODO.md): N-dimensional image processing — SIFT, watershed, optical flow, 3D morphology
- [scirs2-io](./scirs2-io/TODO.md): Scientific data I/O — Protobuf/CBOR/BSON/Avro/Parquet/Feather, streaming, ETL
- [scirs2-datasets](./scirs2-datasets/TODO.md): Datasets and generators for benchmarking and testing

**AI/ML Modules**
- [scirs2-autograd](./scirs2-autograd/TODO.md): Automatic differentiation — JVP/VJP, checkpointing, mixed precision
- [scirs2-neural](./scirs2-neural/TODO.md): Neural networks — Transformers, GNNs, diffusion models, SNN, PPO/DPO, MoE
- [scirs2-graph](./scirs2-graph/TODO.md): Graph algorithms — community detection, VF2 isomorphism, Node2Vec, max-flow
- [scirs2-transform](./scirs2-transform/TODO.md): Dimensionality reduction — UMAP, t-SNE, sparse PCA, persistent homology
- [scirs2-metrics](./scirs2-metrics/TODO.md): ML metrics — IoU/AP/mAP, NDCG, FID/IS, fairness, streaming
- [scirs2-text](./scirs2-text/TODO.md): NLP — BPE/WordPiece, CRF, FastText, NER, LDA, coreference resolution
- [scirs2-vision](./scirs2-vision/TODO.md): Computer vision — stereo depth, ICP, PnP, SLAM, panoptic segmentation, SfM
- [scirs2-series](./scirs2-series/TODO.md): Time series — TFT/N-BEATS/DeepAR, VAR/VECM, EGARCH, FDA, conformal prediction
- [scirs2-wasm](./scirs2-wasm/TODO.md): WebAssembly — WasmMatrix, TypeScript bindings, WASM SIMD, Web Workers
- [scirs2-python](./scirs2-python/TODO.md): Python bindings via PyO3 for 15+ modules

---

## v0.3.3 — RELEASED (March 17, 2026)

### Release Statistics
- 19,684 tests — 100% pass rate
- 2,586,908 lines of Rust
- 6,660 source files, 45+ workspace crates

### Changes
- [x] Upgraded pyo3 to 0.28.2 (Python::with_gil -> Python::attach migration)
- [x] Fixed #[pyclass] deprecation warnings (from_py_object attribute)
- [x] Replaced deprecated criterion::black_box with std::hint::black_box in benchmarks

---

## v0.3.1 — RELEASED (March 9, 2026)

### Release Statistics
- 19,685 tests — 100% pass rate (72% increase from v0.2.0's ~11,400)
- 2,586,887 lines of Rust (35% increase from ~1.9M)
- 6,660 source files, 45+ workspace crates
- 0 compilation errors, 0 test failures (166 skipped by design)
- 0 unwrap() in new code — no-unwrap policy enforced throughout

### Completed — scirs2-neural
- [x] Attention variants: RoPE, GQA, linear/efficient/sparse attention, multi-head latent attention
- [x] Mixture of Experts (MoE): top-k routing, load balancing, expert capacity management
- [x] Capsule networks with dynamic routing and squash activation
- [x] Spiking Neural Networks (SNN): LIF neurons, spike timing, plasticity rules
- [x] Reinforcement learning: PPO, DPO, reward modeling, preference data handling
- [x] Graph Neural Networks: GCN, GAT, GraphSAGE, GIN, graph pooling (DiffPool/SAGPool)
- [x] Vision architectures: SWIN Transformer, UNet, CLIP dual-encoder, ConvNeXt, ViT, PatchEmbedding
- [x] Transformer architectures: GPT-2 (causal masking), T5 (encoder-decoder), full transformer with cross-attention
- [x] Generative models: DDPM/DDIM diffusion models, VAE, GAN, normalizing flows, energy-based models
- [x] Training techniques: federated learning, knowledge distillation, pruning, quantization, continual learning, MAML, contrastive learning, self-supervised learning
- [x] Gradient checkpointing: segment-based memory-efficient backpropagation
- [x] Model serialization: weight format v2 with quantization, computational graph export
- [x] Normalization: LayerNorm2D, RMSNorm, GroupNorm, AdaptiveLayerNorm
- [x] Recurrent cells: GRU/LSTM with peephole connections and layer normalization variants

### Completed — scirs2-stats
- [x] Sequential Monte Carlo (SMC): particle filter with systematic/stratified/multinomial resampling, adaptive tempering
- [x] MCMC: Gibbs sampler, slice sampler, NUTS, HMC
- [x] Distributions: stable (alpha-stable, Levy), GPD, von Mises-Fisher, Tweedie, truncated
- [x] Copulas: Frank, Clayton, Gumbel, Gaussian, Student-t with tail dependence
- [x] Gaussian process regression: Matern/RBF/periodic kernels, sparse GP, deep kernel learning, GP classification
- [x] Hierarchical Bayesian models: mixed effects, multilevel regression, empirical Bayes
- [x] Nonparametric Bayes: Dirichlet process mixtures, CRP, stick-breaking
- [x] Survival analysis: Cox PH (time-varying covariates), Kaplan-Meier, Nelson-Aalen, AFT, competing risks (Fine-Gray)
- [x] Panel data: fixed/random effects, Hausman test, within/between estimators
- [x] Causal inference: causal graph learning, do-calculus, instrumental variables, diff-in-diff
- [x] Bayesian networks: structure learning (PC algorithm, score-based), parameter estimation, exact/approximate inference
- [x] Extreme value theory: GEV/GPD fitting, return level estimation, block maxima, peaks over threshold
- [x] Spatial statistics: variogram, kriging (ordinary/universal/co-kriging), Moran's I, Geary's C
- [x] Information theory: mutual information, KL divergence, Jensen-Shannon divergence, entropy estimators
- [x] Multiple testing corrections: Bonferroni, Holm, Benjamini-Hochberg, Benjamini-Yekutieli
- [x] Effect sizes: Cohen's d, eta-squared, omega-squared, Glass's delta, Hedges' g
- [x] Robust statistics: M-estimators, S-estimators, MM-estimators, minimum covariance determinant

### Completed — scirs2-core
- [x] Work-stealing task scheduler: deque-based stealing, adaptive thread pool sizing, task priorities
- [x] Parallel iterators: parallel map/filter/fold/scan with automatic chunking
- [x] Async utilities: semaphore, barrier, rwlock, channel
- [x] Validation framework: schema validation, type coercion, constraint checking, assertions
- [x] Cache-oblivious algorithms: matrix transpose, merge sort, van Emde Boas layout
- [x] Persistent data structures: HAMT, Red-Black tree with path copying, persistent queue
- [x] Memory management: NUMA-aware allocator, object pool, slab allocator, arena allocator, zero-copy buffers
- [x] Distributed computing: AllReduce/Broadcast/Scatter/Gather, parameter server, ring-AllReduce
- [x] String algorithms: KMP, Boyer-Moore, Rabin-Karp, Aho-Corasick, suffix arrays
- [x] Quantum simulation: qubit state management, quantum gate library, quantum circuit simulation
- [x] Combinatorics: permutations, combinations, partitions with iterator support
- [x] Metrics collection: Prometheus-compatible histograms, counters, gauges
- [x] ML pipeline abstractions: transformer, predictor, evaluator, pipeline
- [x] Interval arithmetic: interval types with basic arithmetic and relational operations

### Completed — scirs2-linalg
- [x] Iterative solvers: GMRES (standard/restarted), PCG, BiCGStab, MINRES, SYMMLQ, QMR
- [x] Tensor decompositions: CP-ALS, Tucker, tensor train, hierarchical Tucker, NTT
- [x] Matrix functions: expm, logm, sqrtm, signm, cosm/sinm/tanm via Schur/Pade
- [x] Structured matrices: Cauchy, companion, Vandermonde, circulant solvers
- [x] Matrix ODEs: Riccati, Lyapunov, Sylvester ODE solvers
- [x] Randomized algorithms: Nystrom approximation, randomized range finder, sketch-and-solve
- [x] Control theory: ARE/DARE solvers, Lyapunov stability, controllability/observability
- [x] Perturbation theory: condition number estimation, backward error analysis, componentwise bounds

### Completed — scirs2-signal
- [x] Radar signal processing: matched filter, CA-CFAR/OS-CFAR/GO-CFAR/SO-CFAR, range-Doppler, pulse compression
- [x] State estimation filters: Kalman, EKF, UKF, particle filter, adaptive Kalman
- [x] Compressed sensing: OMP, ISTA/FISTA, CoSaMP, subspace pursuit
- [x] Audio/speech features: MFCC, chroma features, spectral centroid/bandwidth/rolloff, ZCR
- [x] Time-frequency analysis: EMD, HHT, synchrosqueezing, Wigner-Ville, Zoom FFT
- [x] Wavelet processing: wavelet packet transform, wavelet denoising, continuous wavelet transform
- [x] Array signal processing: MUSIC, ESPRIT, beamforming (delay-and-sum, MVDR/Capon), DOA estimation
- [x] Source separation: FastICA, JADE, SOBI, NMF for audio, convolutive BSS
- [x] Adaptive filtering: LMS, RLS, NLMS, affine projection, Kalman-based adaptive

### Completed — scirs2-graph
- [x] Community detection: Louvain, Girvan-Newman, label propagation, Leiden algorithm
- [x] GNNs: GCN, GAT, Node2Vec, spectral graph convolution
- [x] Graph isomorphism: VF2 algorithm, Weisfeiler-Lehman graph kernels
- [x] Maximum flow: Dinic's algorithm, push-relabel, min-cut, multi-commodity flow
- [x] Layout algorithms: Fruchterman-Reingold, Sugiyama hierarchical, circular, spectral layout
- [x] Visualization: SVG rendering, JSON/DOT export
- [x] Temporal graphs: time-expanded graphs, temporal reachability, contact sequences
- [x] Hypergraphs: hyperedge operations, clique expansion, star expansion, partitioning
- [x] Planarity: LR-planarity testing, planar embedding, Kuratowski subgraph extraction
- [x] Social network analysis: betweenness/closeness/eigenvector/PageRank centrality, structural holes

### Completed — scirs2-series
- [x] Deep learning forecasting: TFT, N-BEATS (interpretable/generic), DeepAR (probabilistic), neural ODE
- [x] VAR/VECM: Granger causality, impulse response functions, FEVD, Johansen cointegration
- [x] Dynamic Factor Model (DFM): EM algorithm, Kalman filter/smoother, factor extraction
- [x] Volatility models: EGARCH, FIGARCH, GJR-GARCH, APARCH, realized volatility
- [x] Functional Data Analysis (FDA): B-spline basis, functional PCA, functional regression/clustering
- [x] Classical methods: Prophet decomposition, Theta method, BATS/TBATS
- [x] Online learning: ADWIN drift detection, online ARIMA, reservoir sampling
- [x] Conformal prediction: time series conformal intervals, rolling/adaptive conformal sets
- [x] Hierarchical forecasting: bottom-up/top-down, MinT reconciliation, OLS reconciliation
- [x] Long memory: ARFIMA, FIGARCH, fractional differencing, Hurst exponent estimation
- [x] Panel time series: common factor models, cross-sectional dependence, panel unit root tests

### Completed — scirs2-optimize
- [x] MIP: branch-and-bound with LP relaxation, cutting planes (Gomory cuts), heuristic upper bounds
- [x] Conic programming: SDP via ADMM, SOCP, self-dual embedding
- [x] Bayesian optimization: constrained BO, multi-fidelity BO (MFBO), transfer BO, warm-start BO
- [x] Metaheuristics: ACO, Differential Evolution, Simulated Annealing, Harmony Search
- [x] Multi-objective: NSGA-III with reference point adaptation, MOEA/D, hypervolume-based selection
- [x] Stochastic: SGD/Nesterov, Adam/AdaW/AMSGrad, SVRG/SARAH/SPIDER variance reduction
- [x] Convex: ADMM, proximal gradient, LASSO, ridge, elastic net, SVM dual, NNLS
- [x] Combinatorial: TSP (2-opt/3-opt/LKH), knapsack (DP/greedy/FPTAS), graph coloring

### Completed — scirs2-fft
- [x] Sparse FFT, Prony method, Lomb-Scargle, Burg AR spectral estimation
- [x] MUSIC spectral estimator for non-uniform sampling
- [x] Advanced transforms: CZT, FRFT, NTT over finite fields
- [x] Mixed-radix FFT: generalized mixed-radix, Rader's and Bluestein's algorithms
- [x] Polyphase filterbank: analysis/synthesis with perfect reconstruction
- [x] DCT/DST all 8 variants (Type I-IV), Modulated Lapped Transform
- [x] Reassigned spectrogram, multi-taper spectrogram, superlet transform

### Completed — scirs2-cluster
- [x] GMM via EM algorithm, Dirichlet process mixture, variational Bayes GMM
- [x] SOM: batch/online learning, Gaussian/Mexican-hat neighborhoods
- [x] HDBSCAN, density peaks, OPTICS, density ratio clustering
- [x] Topological clustering: Mapper (TDA), Vietoris-Rips, Reeb graph clustering
- [x] Deep clustering: DEC, deep k-means, self-supervised clustering
- [x] Stream/online: CluStream, DenStream, D-Stream, BIRCH
- [x] Biclustering: Cheng-Church, FABIA, PLAID, spectral biclustering
- [x] Subspace clustering: SSC, LRR (low-rank representation), ORCLUS
- [x] Time series clustering: DTW-based, feature-based, HMM-based, shapelet-based

### Completed — scirs2-sparse
- [x] Preconditioners: Block Jacobi, SPAI, Additive Schwarz, polynomial preconditioners
- [x] Storage formats: BCSR, ELLPACK, DIA, SELL-C-sigma
- [x] Eigensolvers: LOBPCG, IRAM (Implicitly Restarted Arnoldi), Krylov-Schur
- [x] AMG: classical AMG, smoothed aggregation, unsmoothed aggregation
- [x] Augmented Krylov: GCRO-DR, recycled GMRES, flexible GMRES
- [x] Saddle point systems: block preconditioners, constraint preconditioners
- [x] Domain decomposition: overlapping/non-overlapping Schwarz, FETI, balancing Neumann-Neumann
- [x] Ordering: AMD, Nested Dissection, Reverse Cuthill-McKee

### Completed — scirs2-ndimage
- [x] Feature detection: Gabor filter bank, SIFT, HOG, FAST corners, Harris corner detector
- [x] Segmentation: GrabCut, watershed, SLIC superpixels, random walker, atlas-based
- [x] Quality metrics: PSNR, SSIM, MS-SSIM, FSIM, perceptual quality metrics
- [x] Optical flow: Farneback dense, Lucas-Kanade sparse, Horn-Schunck variational
- [x] 3D operations: 3D morphology, 3D convolution, volumetric analysis, 3D connected components
- [x] Medical imaging: DICOM-like metadata, Hounsfield unit conversion, MRI utilities
- [x] Texture analysis: GLCM, LBP, Gabor texture features, fractal dimension

### Completed — scirs2-special
- [x] Mathieu functions, Mathieu characteristic values
- [x] Coulomb wave functions, regular/irregular, phase shift
- [x] Spherical harmonics Y_lm (real/complex), vector spherical harmonics
- [x] Wigner 3j/6j/9j symbols, Gaunt coefficients, Clebsch-Gordan coefficients
- [x] Jacobi theta functions (theta 1/2/3/4, nome, elliptic nome)
- [x] Debye D_n functions, Clausen function, Fox H-function (Talbot method)
- [x] Heun functions, Appell F1/F2/F3/F4, q-analogs (q-Pochhammer, q-Bessel)
- [x] Parabolic cylinder functions, Whittaker M and W functions
- [x] Weierstrass p-function, Lerch transcendent, Bose-Einstein/Fermi-Dirac integrals
- [x] Extended combinatorics: Bell/Bernoulli/Stirling/Eulerian numbers, partition functions
- [x] Lattice functions: Epstein zeta, lattice theta series, Madelung constants

### Completed — scirs2-transform
- [x] UMAP with fuzzy simplicial set construction
- [x] Barnes-Hut t-SNE with quad/oct-tree acceleration (O(N log N))
- [x] Sparse PCA: LASSO-penalized PCA, dictionary learning-based sparse coding
- [x] Persistent homology: Vietoris-Rips complex, persistent diagram, Betti numbers
- [x] Optimal transport: Wasserstein distance (exact LP), Sinkhorn (regularized), sliced Wasserstein
- [x] Archetypal analysis: simplex-constrained factorization, convex hull approximation
- [x] Metric learning: LMNN, ITML, Siamese/triplet loss
- [x] Multiview learning: CCA, kernel CCA, deep CCA, multiview clustering
- [x] Nonlinear methods: Isomap, LLE, Laplacian eigenmaps, diffusion maps
- [x] NMF variants: L1/L2/KL/Itakura-Saito, convex NMF, semi-NMF
- [x] Feature selection: mRMR, ReliefF, SPEC spectral feature selection, stability selection
- [x] Online methods: incremental PCA, online NMF, streaming UMAP

### Completed — scirs2-vision
- [x] Stereo vision: rectification, SGM/BM disparity estimation, depth from stereo, calibration
- [x] Point cloud: ICP registration, normal estimation, plane fitting, RANSAC alignment
- [x] Camera pose (PnP): EPnP solver, RANSAC-based robust PnP, camera calibration
- [x] Dense optical flow: Farneback, TV-L1, evaluation (EPE, F1 metrics)
- [x] SLAM interface: feature-based SLAM, map management, loop closure detection
- [x] Segmentation: panoptic segmentation, semantic segmentation, instance segmentation
- [x] 3D reconstruction: SfM pipeline, bundle adjustment interface, dense reconstruction
- [x] Image quality: BRISQUE, NIQE, perceptual hash, image fingerprinting

### Completed — scirs2-interpolate
- [x] RBF: thin-plate spline, multiquadric, inverse multiquadric, compact support RBF
- [x] MLS: weighted polynomial fitting, adaptive bandwidth selection
- [x] PCHIP: shape-preserving cubic Hermite interpolation
- [x] Spherical interpolation: spherical harmonics expansion, SLERP, spherical RBF
- [x] Kriging: ordinary, universal, co-kriging, indicator kriging
- [x] Barycentric: Floater-Hormann weights, Lebesgue constant minimization
- [x] B-spline surfaces: bivariate B-spline fitting, NURBS surfaces, surface refinement
- [x] Tensor product: full tensor product, sparse grid, dimension-adaptive
- [x] Natural neighbor interpolation: Sibson/Laplace weights, Voronoi-based

### Completed — scirs2-spatial
- [x] R*-Tree: STR bulk loading, forced reinsertion, split algorithm selection
- [x] Fortune's Voronoi: sweep line algorithm, half-edge data structure
- [x] Geodata: WGS84/GRS80 ellipsoid, Mercator/UTM/Lambert/Albers projections, datum transformations
- [x] Spatial statistics: Ripley's K/L functions, pair correlation, spatial scan statistics
- [x] Trajectory analysis: Douglas-Peucker simplification, Frechet distance, trajectory clustering
- [x] Sweep line: Bentley-Ottmann intersection, polygon clipping (Sutherland-Hodgman, Weiler-Atherton)
- [x] 3D convex hull: Quickhull, half-edge mesh, convex hull properties

### Completed — scirs2-io
- [x] Protocol Buffers (lite), MessagePack, CBOR, BSON, Avro with schema registry
- [x] Parquet (lite), Feather/Arrow IPC, ORC (lite) columnar formats
- [x] Streaming: NDJSON, streaming CSV with schema inference, streaming Arrow
- [x] Cloud storage: S3/GCS/Azure-compatible abstraction, presigned URLs, multipart upload
- [x] Schema management: schema registry, evolution with compatibility modes, versioning
- [x] Data catalog: metadata catalog, lineage tracking, data versioning
- [x] ETL pipeline: source/transform/sink, backpressure handling, typed transforms
- [x] HDF5-lite: pure Rust HDF5-like hierarchical data format

### Completed — scirs2-autograd
- [x] Custom gradient rules: user-defined backward passes, gradient overrides
- [x] Gradient checkpointing: segment-based rematerialization
- [x] Finite differences: forward/backward/central, Richardson extrapolation
- [x] JVP (Jacobian-vector product, forward mode) and VJP (vector-Jacobian product, reverse mode)
- [x] Implicit differentiation: implicit function theorem, fixed-point differentiation
- [x] Mixed precision: FP16/BF16/FP32 mixed precision, loss scaling
- [x] Distributed gradient: gradient synchronization, gradient compression (top-k), accumulation
- [x] Higher-order: Hessian, Jacobian, Taylor-mode AD

### Completed — scirs2-wasm
- [x] TypeScript bindings: complete TS type definitions auto-generated from Rust types
- [x] WasmMatrix: matrix operations exposed to JS/TS, zero-copy where possible
- [x] WASM workers: Web Worker-based parallel computation, message passing protocol
- [x] SIMD operations: WebAssembly SIMD (128-bit), vectorized math in browser

### Completed — scirs2-metrics
- [x] Detection metrics: IoU, AP, mAP, NMS
- [x] Ranking metrics: NDCG, MAP, MRR, Precision@K, Recall@K, BPREF, infAP, GMAP
- [x] Generative metrics: FID, IS, LPIPS, CLIP score
- [x] Fairness metrics: demographic parity, equalized odds, individual fairness, counterfactual fairness
- [x] Segmentation metrics: panoptic quality, semantic IoU, instance AP, boundary F-measure
- [x] Streaming metrics: online sliding window, incremental updates, batching/buffering/partitioning/windowing patterns

### Completed — scirs2-text
- [x] BPE (Byte-Pair Encoding) with merges vocabulary, WordPiece tokenizer, Unigram language model tokenizer
- [x] CRF, HMM-based sequence labeling, BiLSTM-CRF interface
- [x] FastText: subword n-gram embeddings, OOV handling, FastText classification
- [x] NER: rule-based, statistical NER, neural NER interface
- [x] Topic modeling: LDA (collapsed Gibbs), NMF-based, hierarchical LDA
- [x] Semantic parsing: constituency/dependency parsing, CCG supertags
- [x] Coreference resolution: rule-based, mention detection, entity clustering
- [x] Discourse analysis: RST, discourse relation detection
- [x] Knowledge graphs: triple extraction, relation classification, entity linking

### Bug Fixes Completed in v0.3.1
- [x] Bicubic Hermite matrix transpose — incorrect transpose in tensor product kernel construction
- [x] Lanczos QL eigensolver — rewrote tqli with proper implicit shifted QL and deflation
- [x] Bartels-Stewart Sylvester solver — fixed 2x2 Schur block handling for quasi-triangular forms
- [x] LockFreeQueue race condition — eliminated CAS-before-read race (ManuallyDrop + ptr::read, UB-free)
- [x] BDF ODE solver sign error — fixed sign error in residual computation
- [x] PnP RANSAC degeneracy — coplanar point detection and fallback in P3P solver
- [x] External merge sort key mismatch — fixed key function application in scirs2-io
- [x] Burg AR PSD early-stopping — fixed premature termination in Burg's method
- [x] Wavelet polyphase decimation — fixed aliasing in polyphase decimation step
- [x] lfilter off-by-one — corrected IIR filter initial condition computation
- [x] STFT frequency bin count — fixed formula for number of frequency bins
- [x] ReLU gradient mask — corrected subgradient mask (>=0 vs >0) in autograd backward
- [x] DFM Kalman covariance — added symmetrization and Tikhonov regularization
- [x] Watts-Strogatz edge accumulation — fixed duplicate edge accumulation in rewiring
- [x] Spectral clustering eigenvalue sort — corrected ascending/descending sort order
- [x] GPT causal masking — fixed causal attention mask broadcasting for variable lengths
- [x] Frank copula Debye integral — fixed Debye D_1 function in Frank copula parameter estimation
- [x] NUTS MCMC tolerances — adjusted energy conservation tolerance
- [x] GMRES-DR/recycled Krylov — rewrote GCRO-style deflated GMRES with correct harmonic Ritz pairs

---

## v0.4.0 — RELEASED (March 26, 2026)

### scirs2-neural
- [x] Flash Attention v2 — memory-efficient attention with O(N) memory complexity
- [x] Quantization-aware training (INT4/INT8/FP8) with calibration
- [x] ONNX export support for neural network models
- [x] Model tracing/compilation (TorchScript-like static graph)
- [x] Distributed data-parallel training with gradient compression (top-k, random-k)
- [x] LoRA / adapter layers for parameter-efficient fine-tuning
- [x] Speculative decoding for language model inference acceleration
- [x] Mixture-of-Experts load-balancing improvements (auxiliary loss tuning)
- [x] Sparse attention patterns (BigBird, Longformer-style for long sequences)
- [x] 3D convolution layers for video understanding

### scirs2-linalg
- [x] GPU-accelerated matrix operations via OxiBLAS GPU backend
- [x] Structured matrix solvers: Toeplitz, circulant, Hankel in O(N log N)
- [x] Mixed-precision arithmetic: f16/bf16 matrix operations
- [x] Randomized NLA improvements: block Krylov, subspace iteration with deflation
- [x] Hierarchical matrix (H-matrix) representation for dense-but-compressible systems
- [x] Extended precision accumulation in GEMM (for better numerical stability)

### scirs2-stats
- [x] Variational inference: ADVI (automatic differentiation variational inference), ELBO optimization
- [x] INLA (integrated nested Laplace approximation) for latent Gaussian models
- [x] Causal discovery: FCI algorithm, LiNGAM, continuous-time causal discovery
- [x] Frailty models and multilevel survival analysis
- [x] Functional regression enhancements: scalar-on-function, function-on-function
- [x] Improved SMC with adaptive resampling thresholds and parallel particle propagation
- [x] Bayesian neural network approximations (Laplace approximation, SWAG)

### scirs2-signal
- [x] GPU-accelerated FFT pipeline for large-scale signal processing
- [x] Real-time streaming signal processing with bounded latency guarantees
- [x] Advanced beamforming: STAP (space-time adaptive processing), MVDR with diagonal loading
- [x] Deep learning-based denoising: denoising autoencoders, diffusion-model denoisers
- [x] Phase estimation: ESPRIT phase estimation, instantaneous frequency estimation
- [x] Acoustic echo cancellation (AEC) with multi-delay filter bank

### scirs2-graph
- [x] Temporal graph neural networks (TGNN): TGAT, TGN architectures
- [x] Graph transformers: GraphGPS, Graphormer, Exphormer
- [x] Heterogeneous graph learning: HAN (heterogeneous attention network), R-GCN
- [x] Large-scale graph partitioning: METIS-like recursive bisection, streaming partitioning
- [x] Graph condensation: dataset distillation for graphs
- [x] Signed and directed graph learning with specialized embeddings
- [x] Network alignment algorithms (IsoRank, Grasp)

### scirs2-optimize
- [x] Differentiable programming integration: differentiable LP/QP layers (OptNet-style)
- [x] Second-order stochastic optimization: L-BFGS-B improvements, SR1, SLBFGS
- [x] Distributed optimization: ADMM with warm-starting, PDMM, EXTRA
- [x] Hardware-aware neural architecture search (NAS): hardware performance predictor
- [x] Quantum-inspired optimization: QAOA simulation, VQE for combinatorial problems
- [x] Robust optimization: distributionally robust optimization (DRO) with Wasserstein ball constraints
- [x] Multi-fidelity optimization: Hyperband, successive halving, Fabolas

### scirs2-series
- [x] Improved conformal prediction: adaptive conformal sets with coverage guarantees
- [x] Online meta-learning for rapid adaptation to new time series
- [x] Multivariate deep learning models: iTransformer, PatchTST, TimesNet
- [x] Hierarchical forecasting enhancements: bottom-up deep learning reconciliation
- [x] Probabilistic electricity/energy market forecasting
- [x] Continuous-time state space models: HIPPO, S4, Mamba state space model
- [x] Causal time series discovery: PCMCI, structural VAR identification

### scirs2-integrate
- [x] GPU-accelerated PDE solvers (FEM/FDM on GPU via OxiBLAS GPU backend)
- [x] Adaptive mesh refinement (AMR): h-refinement, p-refinement, hp-refinement
- [x] Discontinuous Galerkin improvements: curved elements, high-order DG, entropy-stable schemes
- [x] Port-Hamiltonian discretization: structure-preserving time integration
- [x] Neural-network-assisted PDE solvers: PINN (Physics-Informed Neural Networks) interface
- [x] Uncertainty quantification: polynomial chaos expansion, stochastic Galerkin

### scirs2-vision
- [x] Neural Radiance Fields (NeRF): volume rendering, instant-NGP hash encoding
- [x] 3D object detection: PointPillar-like pillar feature extraction, VoxelNet-style
- [x] Foundation model integration: SAM-compatible prompt-based segmentation interface
- [x] Real-time multi-object tracking (MOT): SORT, DeepSORT, ByteTrack
- [x] Event camera support: event-based optical flow, event-to-frame conversion
- [x] Depth completion: sparse-to-dense depth from LiDAR + camera fusion

### scirs2-core
- [x] GPU memory pooling improvements: defragmentation, async allocation
- [x] NUMA-aware work stealing: topology-aware task migration
- [x] Lock-free data structure enhancements: lock-free skiplist, lock-free B-tree
- [x] Distributed computing: fault-tolerant parameter server, gossip-based AllReduce
- [x] Reactive programming utilities: signal/slot, push-pull dataflow
- [x] Extended precision arithmetic integration: quad precision (f128), double-double
- [x] Task graph dependency tracking: topological execution ordering with cycle detection
- [x] Tracing improvements: OpenTelemetry 0.30+ compatibility, structured logging

### scirs2-sparse
- [x] GPU-accelerated SpMV: cuSPARSE-compatible interface via OxiBLAS GPU backend
- [x] Parallel AMG coarsening: parallel strength-of-connection, parallel coarsening algorithms
- [x] Machine learning-guided preconditioner selection
- [x] Low-rank updates to sparse factorizations
- [x] Sparse Cholesky modifications (rank-1 updates/downdates)
- [x] Integrated multigrid with deep learning error smoothers

### scirs2-fft
- [x] GPU-accelerated FFT pipeline via OxiFFT GPU backend
- [x] Adaptive sparse FFT: parameter-free sparsity estimation
- [x] High-performance multidimensional FFT with GPU tiling
- [x] Compressed sensing recovery via FFT-based measurements
- [x] Fast multipole method (FMM) integration for N-body problems

### Pure Rust Policy Enforcement and Dependency Cleanup

#### Completed in v0.3.4
- [x] Removed `ndarray-npy` from scirs2-core — eliminated `zip` crate from dependency tree entirely
- [x] Removed unused workspace deps: `x509-parser`, `itertools`, `num-rational`, `gmp-mpfr-sys`
- [x] Removed unused OTel deps: `opentelemetry-prometheus`, `opentelemetry-semantic-conventions`
- [x] Removed unused scirs2-io stubs: `mongodb`, `redis`, `prost` (direct dep)
- [x] Fixed dangling feature refs: `opentelemetry-semantic-conventions` in scirs2-core `instrumentation` feature, `itertools` in scirs2-graph

#### Transitive Policy Violations (indirect deps from external crates)
These are pulled in by external crates we depend on — not direct violations, but targets for future Pure Rust replacements:

- [x] `flate2` + `zlib-rs` — pulled in transitively by:
  - `parquet` (scirs2-io) — flate2 for Parquet compression
  - `png` / `tiff` (image crate) — flate2 for image codec
  - `ureq` (scirs2-datasets) — flate2 for HTTP content encoding
  - **Action**: Evaluate Pure Rust Parquet alternatives or contribute upstream patches; for image/ureq these are internal to the crate and not directly replaceable
- [x] `snap` (Snappy) — pulled in by `parquet`
  - **Action**: Feature-gate `parquet` support so users who don't need it avoid these deps
- [x] `prost` (transitive) — pulled in by `opentelemetry-otlp` (scirs2-core)
  - **Action**: Consider whether OTel OTLP export is needed; if not, remove `opentelemetry-otlp` or feature-gate it

#### Barely-Used Dependencies (audit candidates)
- [x] `ureq` (8 uses) — overlaps with `reqwest` (80+ uses); consolidate HTTP client to one
- [x] `cron` (4 uses) — ensure properly feature-gated, not pulled into default builds
- [x] `opentelemetry_sdk` (4 uses) — verify necessity; most OTel setup is stub code
- [x] `num_bigint` (4 uses) — verify `arbitrary-precision` feature is not in default features
- [x] `egui` / `eframe` (~40 uses) — GUI deps; ensure feature-gated (visualization only)

#### Dead Code Cleanup
- [x] Audit `scirs2-core` JIT module — Cranelift crates removed but JIT stub code may remain in `array_protocol/jit_impl.rs`
- [x] Audit `scirs2-io` mongodb/redis feature stubs — feature names kept but deps removed; clean up cfg blocks if they contain only stub code
- [x] Audit `array_io` feature in scirs2-core — ndarray-npy removed; feature now only gates `["std", "array"]` which may be redundant

#### Dependency Reduction Goals for v0.4.0
- [x] Reduce total workspace dependency count by 15-20%
- [x] Ensure all default feature sets are 100% Pure Rust (no C/Fortran transitive deps)
- [x] Run `cargo tree --workspace --all-features -d` regularly to detect unnecessary version duplicates
- [x] Implement `cargo-scirs2-policy` linter check for banned direct deps (zip, flate2, bincode, openblas, etc.)

### Infrastructure and Tooling
- [x] WebGPU backend for scirs2-wasm (browser-side GPU compute via WebGPU API)
- [x] Python PyPI wheel distribution via maturin (Linux/macOS/Windows wheels)
- [x] mdBook documentation website with interactive examples and tutorials
- [x] Comprehensive benchmark regression suite (criterion-based with CI integration)
- [x] Jupyter notebook examples via evcxr Rust kernel
- [x] cargo-scirs2-policy linter: detect direct `use rand::*` / `use ndarray::` in non-core crates
- [x] Cross-platform Windows test suite improvements (full parity with Linux/macOS)
- [x] Julia binding improvements: JLL-based distribution for easier Julia package installation
- [x] Protocol buffer schema registry integration for scirs2-io

---

## v0.4.2 — RELEASED (April 12, 2026)

### scirs2-core
- [x] Metal GPU batch dispatch mode (begin_batch/end_batch/try_batch_dispatch)
- [x] Metal GPU async dispatch (dispatch_no_wait + gpu_sync)
- [x] Removed all .expect() violations from GPU backends

### scirs2-optimize
- [x] Bayesian optimization integer dimension enforcement (enforce_integer_dims)
- [x] GDAS (Gumbel-DARTS) NAS with Gumbel-Softmax sampling and temperature annealing
- [x] SNAS (Stochastic NAS) with concrete distribution relaxation and resource penalty
- [x] Predictor-based NAS with kernel ridge regression surrogate and active learning
- [x] NAS module consolidation — nas/ module wired up to lib.rs

### scirs2-integration-tests
- [x] Real integration tests for sparse linear solvers (CG/BiCGSTAB/GMRES)
- [x] Real integration tests for statistical analysis (correlation, hypothesis testing)
- [x] Enhanced FFT integration tests (spectral peaks, filtering, convolution theorem)
- [x] Optimizer convergence integration tests

### Wave 42 (April 2026)
- [x] scirs2-core: Async GPU buffer transfer pipeline with overlapping CPU/GPU transfers
- [x] scirs2-core: Unified memory allocator (CPU+GPU shared pages)
- [x] scirs2-core: Persistent vector (RRB-tree) with structural sharing
- [x] scirs2-core: Tracy profiler integration (feature-gated)
- [x] scirs2-core: NUMA-local allocator with libnuma feature gate
- [x] scirs2-signal: GPU-accelerated spectrogram computation (GpuSpectrogram)
- [x] scirs2-signal: GPU matched filter bank (MatchedFilterBank)
- [x] scirs2-linalg: Auto mixed-precision selection by condition number
- [x] scirs2-linalg: GPU eigensolver interface (Householder + QL + Lanczos)
- [x] scirs2-linalg: Mixed CPU/GPU solver with iterative refinement
- [x] scirs2-io: Apache Iceberg table format support
- [x] scirs2-io: DataFusion-compatible table provider interface
- [x] scirs2-io: Vectorized expression evaluation for filter/project
- [x] scirs2-io: Hash join, merge join, nested-loop join algorithms
- [x] scirs2-special: Hecke L-functions and Maass forms
- [x] scirs2-special: Elliptic curve L-functions (BSD numerics)
- [x] scirs2-special: Validated numerics (ball arithmetic with certified enclosures)
- [x] scirs2-special: Connection formula generator (Bessel, hypergeometric, Legendre, Kummer)
- [x] scirs2-integration-tests: ML pipeline (datasets → neural → optimize → metrics)
- [x] scirs2-integration-tests: Signal analysis pipeline (signal → fft → stats)
- [x] scirs2-integration-tests: NLP pipeline (text → neural → metrics)
- [x] scirs2-integration-tests: Computer vision pipeline (ndimage → vision → metrics)
- [x] scirs2-integration-tests: Graph ML pipeline (graph → linalg → metrics)
- [x] scirs2-integration-tests: Scientific computing pipeline (integrate → linalg → sparse)

### Wave 43 (April 2026)
- [x] scirs2-core: Per-stream allocation (StreamAllocator, StreamId) in gpu/stream_allocator.rs
- [x] scirs2-core: Memory defragmentation (DefragPlanner, OnlineDefragmenter) in memory/defrag.rs
- [x] scirs2-core: Cross-NUMA bandwidth measurement and routing in memory/numa_bandwidth.rs
- [x] scirs2-core: Automatic NUMA-aware placement (optimal_placement_node)
- [x] scirs2-io: Object-store abstraction layer (LocalFsStore, MemoryStore, S3/GCS/Azure stubs)
- [x] scirs2-io: AWS S3 multipart upload state machine (feature-gated)
- [x] scirs2-io: Adaptive compression with entropy-based algorithm selection (OxiARC-backed)
- [x] scirs2-io: Mini-batch sampler with shuffle, stratified splitting, train/val/test split
- [x] scirs2-io: SafeTensors, ONNX proto, TFRecord ML formats (already existed — verified and documented)
- [x] scirs2-special: GPU auto-dispatch (batch_gamma, batch_erf, batch_bessel_j0)
- [x] scirs2-special: Mixed-precision f16 accumulation (batch_eval_gamma_f16, batch_eval_erf_f16)
- [x] scirs2-special: Clebsch-Gordan series for SU(2), SU(3), SO(5) Lie groups
- [x] scirs2-special: Hall polynomials for p-group extensions
- [x] scirs2-series: All 6 v0.4.0 items already implemented — verified and marked done
- [x] scirs2-sparse: Mixed CPU/GPU preconditioning (ILU(0) + preconditioned CG)
- [x] scirs2-optimize: Subspace embedding methods (Gaussian/Sparse/JL + sketched least-squares)
- [x] scirs2-python: scirs2.special, scirs2.interpolate, scirs2.integrate Python bindings; no-unwrap fixes
- [x] scirs2-numpy: DLPack protocol (__dlpack__, __dlpack_device__), masked arrays, structured dtype, PyUntypedArray, runtime dtype inspection
- [x] scirs2-neural: Pipeline parallelism, tensor parallelism wired up; all 9 v0.4.0 TODO items confirmed complete; INT4 quantization verified

### Wave 44 (April 2026)
- [x] scirs2-neural: NAS module repair — GDAS/SNAS/predictor-based NAS, 74 tests passing
- [x] scirs2-neural: Mamba SSM (state-space model) verified working
- [x] scirs2-core: Numerical validation tests (40 tests) for core mathematical operations
- [x] scirs2-core: Cross-crate consistency tests (16 tests)
- [x] scirs2-optimize: CMA-ES (covariance matrix adaptation evolution strategy) optimizer, 10 tests
- [x] scirs2-text: Enhanced BPE tokenizer with chat templates (14 tests)

### Wave 45 (April 2026)
- [x] scirs2-linalg: H-matrix compression (hierarchical matrix representation), 10 tests
- [x] scirs2-special: Spheroidal wave functions + Mathieu-Hill enhancements, 25 tests
- [x] scirs2-fft: Streaming FFT (out-of-core ring-buffer STFT), 18 tests
- [x] scirs2-signal: Batched Welch PSD + EFDD modal analysis, 12 tests
- [x] scirs2-numpy: Array protocol + DLPack extensions, 27 tests
- [x] scirs2-datasets: HuggingFace integration + sharding + generators, 493 lib tests
- [x] scirs2-io: GCS + Azure SAS presigned URLs + exactly-once semantics, 35 tests
- [x] scirs2-interpolate: GPU RBF + physics-informed + deep kriging + active learning, 25 tests
- [x] scirs2-text: Sentence embeddings + multilingual + HDP topic modeling, 34 tests
- [x] scirs2: Feature groups + prelude module (facade crate enhancements)
- [x] scirs2-metrics: Rotated IoU + bounding box overlap utilities, 17 tests
- [x] scirs2-integrate: GPU LBM + ODE ensemble + sparse grid quadrature, 27 tests

---

## v0.4.3 — RELEASED (May 3, 2026)

See CHANGELOG.md `[0.4.3]` for the full feature list. Highlights: 34,883 tests passing; core dependency upgrades (rayon 1.12, rand 0.10.1, nalgebra 0.34.2); `scirs2-special::printpdf` feature-gated; pathfinder_simd ARM NEON patch.

---

## v0.4.4 — RELEASED (May 15, 2026)

**Centerpiece**: `scirs2-symbolic` is now a complete EML-IR-native CAS substrate with seamless `scirs2-autograd` integration and a real OxiZ-backed SMT layer. See `scirs2-symbolic/TODO.md` for the per-phase status grid.

### scirs2-symbolic — Phase 0 (Substrate) — COMPLETE (13/13, Wave 53)
- [x] `eml::tree` Arc-shared `EmlNode` / `EmlTree` with thread-local hash-cons (u128 structural hash)
- [x] `eml::canonical` — 27 canonical EML constructors covering every elementary function
- [x] `eml::op` + `eml::lower` — `LoweredOp` flat IR with `OxiOp` stack-machine tape; `lower`/`raise`
- [x] `eml::parser` + `eml::display::to_latex` — text round-trip + LaTeX export with π/e exact-match
- [x] `eml::eval` — iterative stack-machine real + complex evaluator; stack-safe on 543-deep canonical sin
- [x] `eml::simplify` — fixed-point rewrite (constant folding, identity rules, inverse cancellation, hash-based commutative ordering)
- [x] `eml::grad` — symbolic gradient; `grad`, `grad_all`, `jacobian`, `hessian` all public
- [x] `eml::interval` — outward-rounded interval arithmetic with monotone-region splitting
- [x] `eml::bridge` — `Expr ↔ LoweredOp` adapter with deterministic `VarMap`
- [x] oxieml v0.1.0 parity test harness at 1e-9 tolerance (`tests/oxieml_parity.rs`)
- [x] OxiZ workspace dep registered (`oxiz = "0.2.1"` behind `smt` feature)
- [x] ADR-0001: clean-room native EML implementation
- [x] Cycle-prevention CI gate (`scripts/check-no-symbolic-in-core.sh`)

### scirs2-symbolic — Phase 1 (Native API) — 11/13 (Waves 54–56)
- [x] `regression::discover` ndarray API (SR engine, Config builder, Pareto front)
- [x] `regression::discover_multi` for vector-valued targets
- [x] `regression::discover_ode` SINDy-style API
- [x] `units::UnitAware` SI dimensional analysis
- [x] `regression::with_constraints` SMT-pruned search (BoundedOutput, Monotonic)
- [x] `compile::to_jit` Cranelift CPU JIT + `JitCache`
- [x] `interval::eval_interval` public surface
- [x] LaTeX / pretty / JSON / oxicode round-trip
- [x] Python bindings (`scirs2-python::symbolic`)
- [x] Physics examples (4 examples)
- [x] Integration test suite (~564 `#[test]` markers across the crate)
- [x] ~~Criterion benchmark suite vs PySR on FSReD dataset~~ (DONE Wave 55, 2026-05-04)
- [x] **`docs/cas_tutorial.md` end-to-end walkthrough** (planned 2026-05-15) (DONE Wave 75, 2026-05-15 — docs/cas_tutorial.md 1488 lines; primary verification at scirs2-symbolic/tests/cas_tutorial_compile.rs with 13 #[test] functions, one per section)
  - **Goal:** A 1000-1500-line tutorial markdown that walks a new user through using `scirs2-symbolic` from raw parsing to JIT-compiled execution, exercising every Phase-2 entry-point. Becomes the canonical "how to use the CAS" reference for v0.5.0.
  - **Design:** 13-section structure covering: EML substrate, parsing/pretty-printing, canonicalization, identity DB + SMT (OxiZ NLSAT 0.2.1 limitation noted), cas::solve (linear/poly/system/ODE), integrate_rational (Risch-LITE), GradGraph differentiation, Cranelift JIT, GPU WGSL codegen, Python+WASM bindings, cross-crate integration (optimize::symbolic::newton), diffgeom Schwarzschild Ricci scalar, reference and next-steps.
  - **Files:** `/Users/kitasan/work/scirs/docs/cas_tutorial.md` (new, ~1200 lines); `/Users/kitasan/work/scirs/scirs2-symbolic/tests/cas_tutorial_compile.rs` (new, ≥13 `#[test]` functions, one per tutorial section).
  - **Tests:** `cargo nextest run -p scirs2-symbolic --test cas_tutorial_compile` — ≥13 tests pass. This is the PRIMARY harness (doc lives outside crate; `cargo test --doc` can't reach it).
  - **Risk:** API drift between Wave 70 and 75 could invalidate samples; `tests/cas_tutorial_compile.rs` is the single source of truth.

### scirs2-symbolic — Phase 1 Design-freedom — 3/4 + 1 partial
- [x] ~~SR engine on `scirs2-core` NUMA-aware scheduler (partial — rayon path live; NUMA worker pinning deferred to v0.4.5)~~ (DONE Wave 73, 2026-05-08)
- [x] JIT routes through `scirs2-core` GPU pipeline (`compile::to_gpu`, `to_jit_auto`)
- [x] Symbolic gradient as native AD tape backend (seamless via `scirs2-autograd::symbolic_backend::EmlOp` + `eml_scalar_op`; provenance-dispatched)
- [x] SMT calls OxiZ directly (Wave 57 real QF_NRA integration; note: OxiZ 0.2.1 NLSAT incomplete for surface commutativity — always canonicalize first)

### scirs2-symbolic — Phase 2 (EML-IR-Native CAS) — 13/15
- [x] `cas::canonicalize` — EML-native canonical form with 7 algebraic rewrite rules (Wave 57); fixed-point idempotent; 32 tests
- [x] `cas::pattern` — pattern matching engine (712 LoC); prerequisite for identity-db and e-graph rule application; hooked throughout (Wave 59)
- [x] `cas::identity_db` — 11 standard trig/hyperbolic/log identities via `IdentityDb::standard()`; O(1) hash lookup; hooked into `cas::canonicalize` fixed-point loop; 73 tests (Wave 59)
- [x] `cas::smt` Ackermann transcendental encoding — `encode_transcendental` for 16 ops; Pythagorean axiom; cache keyed on canonical hash; 10 new tests (`smt` feature, Wave 59)
- [x] `cas::certified_rewrite` — `CertifiedRule` trait + RAII push/pop safety + `MAX_CERT_ITER=8`; rejects counterexample-producing rules; 7 tests (`smt` feature, Wave 59)
- [x] `cas/e_graph/` — full egg-style equality saturation (6 files, 1,983 LoC): `UnionFind`, `ENode`/`EClass`, `EGraph::{add,union,rebuild}`, pattern matching, `SaturationBudget`, DP extraction, `canonicalize_egraph`; 16 tests (Wave 59)
- [x] `units::infer_dimension` + `Constraint::DimensionMatch { target, var_dims }` — iterative post-order dimensional inference; 22 tests (Wave 59)
- [x] `cas::cse_dag::CseDag` — O(unique-nodes) structural-hash CSE DAG with topological-order eval; 11 tests (Wave 61)
- [x] `cas::series` — Taylor + Padé approximants in EML form; `taylor`, `pade`, `SeriesError`; 8 tests (Wave 61)
- [x] Property-based EML-rewrite testing via proptest — 3 properties × 1024 cases; tests/cas_rewrite_proptest.rs (Wave 61)
- [x] `cas::certified_value::CertifiedValue` — certified `[lo,hi]` interval for symbolic values; `certify`, `certify_const`, `tighten_to`; 9 tests
- [x] `cas::solve` — invertible-chain + polynomial solver (degree 1-2 exact); `SolveResult`, `SolveError`; 10 tests
- [x] `eml_pattern!` proc-macro DSL (`scirs2-symbolic-macros`) — write Pattern rules without boilerplate; 18 unary + 5 binary; 13 tests
- [x] EML rewriter criterion benchmark baseline — `scirs2-symbolic/benches/cas_bench.rs` (5 groups: canonicalize, identity_db, egraph, CseDag, series/taylor+pade; completed 2026-05-04; SymPy subprocess comparison deferred to v0.4.5)
- [x] `cas::identity_proof::discover_identity` — SR + canonicalize + hash-match pipeline; `ProofCertificate`; 8 tests
- [x] ~~native AD kernel; WASM playground; Risch integration~~ (DONE Wave 67+70, 2026-05-04)

### scirs2-symbolic — Phase 3 (Cross-crate integration) — 18/12+
- [x] `scirs2-optimize::symbolic::newton` — symbolic Newton with exact gradient + Hessian via `eml::grad`; Gaussian-elimination linear solve; 6 tests
- [x] `scirs2-autograd::symbolic_backend::EmlOp` + `eml_scalar_op` — seamless symbolic-tensor integration; forward via `eval_real`, backward via exact `sym_grad`; composable with stock autograd ops; 8 integration tests
- [x] `scirs2-integrate::eml` — `solve_ivp_symbolic` (BDF1 stiff ODE + symbolic Jacobian), `quad_gauss_legendre_symbolic`; 15 tests (Wave 60)
- [x] `scirs2-stats::mle_symbolic` — `fit_mle_symbolic` gradient descent with backtracking line search; 8 tests (Wave 60)
- [x] `scirs2-neural::{activations,losses}::symbolic` — `SymbolicActivation` (Activation+Layer) + `SymbolicLoss` (Loss) via `eval_real`; 10 tests (Wave 60)
- [x] `scirs2-linalg::symbolic` — `det_symbolic` (Leibniz n≤4), `eigenvalues_symbolic_2x2`, `condition_number_symbolic`; 12 tests (Wave 60)
- [x] scirs2-autograd: float-tape vs EML symbolic gradient parity suite (12 ops × 100 points, 1e-10 tolerance)
- [x] scirs2-optimize: L-BFGS symbolic + trust-region symbolic — two-loop L-BFGS + dogleg; 8 new tests
- [x] scirs2-optimize: Lagrangian + KKT — `build_kkt` + `solve_lagrangian_symbolic`; Newton on N×N KKT system; 6 tests
- [x] scirs2-autograd: EML vs float-tape criterion benchmark (eml_vs_tape.rs; 5 groups; x², sin, exp, composition, multi-input)
- [x] `scirs2-linalg::symbolic::recognize` — `StructureKind` {Scalar,Diagonal,LowRankUpdate,Circulant,General}; `recognize()` + `inverse_by_structure()` (Sherman-Morrison for rank-1); 8 tests; 615 LoC (Wave 69)
- [x] `scirs2-linalg::symbolic::expm` — `expm_symbolic_2x2` + `expm_symbolic_3x3` wrapping `cas::matrix_exp`; diagonal fast path; `ExpmSymbolicError`; 26 tests; 538 LoC (Wave 69)
- [x] `scirs2-linalg::symbolic::spectral` — `eigenvalues_circulant` (DFT formula as `LoweredOp`); `eigenpairs_symmetric_2x2`; `structured_eigenvalues` dispatch-by-structure; 7 tests; 430 LoC (Wave 69)
- [x] `scirs2-symbolic::cas::moments_catalog` — closed-form mean/variance/MGF for Normal, Exp, Bernoulli, Geometric, Uniform (no MGF) as `LoweredOp`; `MomentsCatalog`, `MomentsError`; 8 tests; 308 LoC (Wave 70)
- [x] `scirs2-symbolic::cas::expected_fisher_catalog` — per-sample expected Fisher matrix for Normal, Exp, Bernoulli, Geometric (Uniform rejected); 4 tests; 209 LoC (Wave 70)
- [x] `scirs2-symbolic::cas::noether_conservation` — Poisson-bracket-based conservation detection 1-DOF and n-DOF; `ConservationCheck`; harmonic, free particle, anharmonic, 2-DOF angular momentum; 10 tests; 480 LoC (Wave 70)
- [x] `scirs2-neural::symbolic::rope_attention` — closed-form RoPE attention logit proving relative-position-only dependence; 9 tests; 419 LoC (Wave 70)
- [ ] All remaining Phase-3 integration targets (extended SR-as-prior, transformer-shape attention beyond RoPE, etc.)

### scirs2-symbolic — Phase 4 (Research) — 9/N
- 7 items completed through Wave 68 (cas::inverse_symbolic, cas::matrix_ops, cas::matrix_exp, cas::spectral_2x2, cas::mle_catalog, cas::observed_fisher, cas::quadratic_line_search)
- [x] `cas::reversible` — `RewriteStep` + `RewriteTrace` + `canonicalize_traced`; `is_fully_reversible()` + `reverse()`; batch-pass tracing; 8 tests; 413 LoC (Wave 69)
- [x] `cas::integrate_rational` — Risch-LITE: ∫P(x)/Q(x) for literal-coeff rationals; partial fractions for degree-2 denominators (real distinct / repeated / complex conjugate); 16 tests; 860 LoC (Wave 70)
- Remaining Phase-4 items (neural-guided EML, Coq/Lean proof export, differentially-private SR, quantum-symbolic, differential geometry, inverse-calc at scale, LM program synthesis) remain open and explicitly research-grade.

### Documentation
- [x] LaTeX export module docs (`eml::display::to_latex` rustdoc)
- [x] `scirs2-autograd::symbolic_backend` rustdoc
- [x] CHANGELOG.md v0.4.4 entry covering Waves 53–58
- [x] Root TODO.md and scirs2-symbolic/TODO.md status snapshots
- [x] Root README.md v0.4.4 highlights section

### Quality Gate
- 0 clippy warnings across 8 scirs2-symbolic feature configs (no-default, serde, smt, jit, gpu, parallel, numa, all-features)
- scirs2-symbolic: 743 tests passing (was 735 before Wave 66)
- scirs2-autograd (with symbolic): 1,165 tests; scirs2-stats (with symbolic): 2,469 tests
- scirs2-linalg (with symbolic): 1,997 tests; scirs2-integrate (with symbolic): 1,698 tests; scirs2-neural (with symbolic): 1,763 tests
- All 4 physics examples build and run cleanly
- Cycle-prevention CI gate passes
- No-unwrap policy: PASS in production code paths

---

## v1.0.0 — PLANNED (Q4 2026)

### API Stability Guarantees
- [x] Semantic versioning commitment: backward-compatible across 1.x series
- [x] Deprecation timeline policy: 2-release warning cycle before removal
- [x] Long-term support (LTS) branch with security patches
- [x] Public API stability tests: compile-fail tests for removed APIs

### Comprehensive Testing and Validation
- [x] 95%+ code coverage across all primary modules — `.github/workflows/coverage.yml` enabled (was .disabled); gate at 70% enforced, 95% target documented; cargo-llvm-cov + codecov + per-crate breakdown
- [x] Statistical validation for all 40+ distributions against NumPy/SciPy reference
- [x] Numerical benchmark comparisons: LAPACK, FFTW, SciPy for all algorithms
- [x] Performance regression tests in CI (nightly benchmarks with Bencher.dev integration)
- [x] Fuzzing coverage for all parsing code (io, text tokenizers)
- [x] Cross-platform compatibility tests on Windows, macOS, Linux (x86_64, ARM64)

### Documentation Excellence
- [x] Complete tutorial series for all major modules (beginner to advanced)
- [x] Migration guide from SciPy/NumPy/scikit-learn with automated conversion hints
- [x] API reference with full examples and mathematical references
- [x] Video tutorials for key workflows (tutorial examples serve as foundation)
- [x] Multi-language documentation (EN, JP)

### Enterprise Features
- [x] Security audit and supply chain verification (SBOM, cargo-audit CI)
- [x] Performance SLA guarantees with published benchmark baselines
- [x] Enterprise deployment guides (containerization, cloud)
- [x] Commercial support channel documentation

---

## Quality Gates and CI Enhancements

### Current CI Infrastructure
- Pure Rust toolchain with cargo-nextest
- Zero warnings enforcement (clippy + rustc)
- Comprehensive test coverage (34,883 tests)
- No-unwrap policy enforced in code review

### Planned CI Enhancements
- [x] Statistical validation in CI: automated correctness tests for all distributions vs NumPy/SciPy
- [x] cargo-scirs2-policy linter: detect `use rand::*`, `use ndarray::` in non-core crates
- [x] Performance regression detection: nightly benchmarks with automated alerts
- [x] Cross-platform testing: Linux (x86_64, ARM64), macOS (Intel, Apple Silicon), Windows (MSVC, GNU)
- [x] WebAssembly target testing: wasm32-unknown-unknown, wasm32-wasi
- [x] Fuzzing: cargo-fuzz integration for IO parsing code

---

## Ecosystem Collaboration

### Current Integrations
- **NumRS2**: Numerical computing ecosystem — 99%+ test pass rate with SciRS2-Core
- **OxiRS**: RDF/SPARQL graph database — 100% build success, removed 269-line compatibility shim
- **SkleaRS**: Scikit-learn compatibility layer — 100% build success
- **TrustformeRS**: Transformer models — active integration with scirs2-neural and scirs2-autograd
- **OptiRS**: Independent ML optimization project (former scirs2-optim)
- **OxiBLAS**: Pure Rust BLAS/LAPACK (used throughout workspace)
- **OxiFFT**: Pure Rust FFT (used in scirs2-fft, scirs2-signal)

### Future Collaborations
- [x] NumRS2: share statistical validation framework and distribution correctness tests
- [x] OxiRS: validate metrics API against SPARQL workloads using scirs2-metrics
- [x] SkleaRS: provide property-based test utilities for ML algorithm validation
- [x] TrustformeRS: enhance Transformer support in scirs2-neural with Flash Attention

---

## Policy References

All development must adhere to the following policies:

- **No unwrap() Policy**: No `unwrap()` or `expect()` in production code; use `?` and proper error handling
- **Pure Rust Policy**: Default feature set must be 100% Pure Rust (no C/Fortran); optional C-backed features must be feature-gated
- **COOLJAPAN Ecosystem Policy**: Use OxiBLAS (not OpenBLAS/MKL), OxiFFT (not rustfft/FFTW), oxiarc-* (not zip), oxicode (not bincode)
- **Workspace Policy**: All crate versions use `*.workspace = true`; no per-crate version declarations (except keywords/categories)
- **File Size Policy**: No single file > 2000 lines; use `splitrs` for refactoring (installed at `~/work/splitrs/`)
- **Naming Convention**: `snake_case` for all variables, functions, modules; `CamelCase` for types and traits
- **SciRS2 POLICY**: All non-core crates must use `scirs2-core` abstractions for rand/ndarray/num_complex access

### Reference Documents
- [SCIRS2_POLICY.md](SCIRS2_POLICY.md): Ecosystem architecture and core abstractions
- [CHANGELOG.md](CHANGELOG.md): Detailed changelog for each release
- [CLAUDE.md](CLAUDE.md): Development guidelines and best practices
- [README.md](README.md): Project overview and quick start

### External Resources
- GitHub Repository: https://github.com/cool-japan/scirs
- Documentation: https://docs.rs/scirs2
- OptiRS Project: https://github.com/cool-japan/optirs
- NumRS2: https://github.com/cool-japan/numrs
- ToRSh (PyTorch-compatible): https://github.com/cool-japan/torsh
- SkleaRS (Scikit-learn compatibility): https://github.com/cool-japan/sklears
- TrustformeRS (Transformers): https://github.com/cool-japan/trustformers
- OxiRS (RDF/SPARQL): https://github.com/cool-japan/oxirs

---

**Last Updated**: 2026-06-02
**Branch**: 0.5.0
**Status**: v0.5.0 (current) — Waves 53–77 complete. EML-IR CAS substrate (Phases 0–3 complete, Phase 4 ongoing); GPU dispatch real (BFS/SSSP/delta-stepping/RBF/CG/Newton/L-BFGS via wgpu); GpuNdarray<f32> with axis ops; NUMA par_map_chunks; ALiBi + Riemann/Weyl diffgeom + SR-as-prior + SymbolicPriorLoss; ~36,082 tests passing. See CHANGELOG.md `[0.5.0]` for full release notes.

---

## Wave 72 (May 2026)
- [x] scirs2-special: `erfc_batch_wgpu` + `ERFC_WGSL` (complementary error function, hard clamp |x|>6); `erfinv_batch_wgpu` + `ERFINV_WGSL` (Winitzki 2008 rational approximation); both re-exported from `gpu_kernels` and routed through `batch_erfc`/`batch_erfinv` in `gpu_dispatch.rs`; 2 new integration tests added to `tests/gpu_wgpu_dispatch.rs`
- [x] scirs2-fft: `fft_wgpu()` fully implemented — multi-pass Cooley-Tukey radix-2 DIT dispatch via wgpu; bit-reverse permutation, per-stage uniform-buffer update, graceful skip when no adapter; roundtrip and non-power-of-two tests
- [x] scirs2-stats: new `gpu` module (`src/gpu/mod.rs`) with batch WGSL compute shaders for Normal log-PDF, Normal CDF, Exponential log-PDF, Exponential CDF; `MIN_GPU_SIZE=1024` threshold ensures small arrays always use CPU (f64 precision); `gpu` and `gpu_wgpu` feature flags; all tests pass

---

## Wave 71 (May 2026)
- [x] scirs2-special: `wgpu_kernels` feature — real wgpu dispatch for `gamma_batch_wgpu`, `erf_batch_wgpu`, `bessel_j0_batch_wgpu`; new `lgamma_batch_wgpu` + `LGAMMA_WGSL` shader; integration test suite (5 tests, graceful skip when no adapter)
- [x] scirs2-core: WGSL compute shaders for `ElementwiseSub/Mul/Div/Pow/Sqrt/Exp/Log` kernels — registry-path wgpu now covers all common elementwise ops; new smoke tests verify shader compile and workgroup layout
- [x] scirs2-transform: `GpuPCA::fit/transform/fit_transform` now CPU-backed via `reduction::PCA` (SVD); 18 functional tests replace 5 stub tests; `GpuPCA` no longer returns `NotImplemented`

---

## /stub-check 2026-05-05 — Deferred items

These stubs were discovered by /stub-check but deferred because they require external SDKs, major new subsystems (>5000 LoC), or have dedicated planned waves.

### GPU backends (Pure Rust Policy: no CUDA/Metal/OpenCL/Vulkan SDKs)
- [x] `scirs2-core/src/gpu/kernels/elementwise.rs` — WGSL filled in for `sub`, `mul`, `div`, `pow`, `sqrt`, `exp`, `log` (Wave 71)
- [x] `scirs2-special/src/gpu_kernels/` — wgpu dispatch wired for gamma/erf/bessel_j0/lgamma; `wgpu_kernels` feature (Wave 71)
- [x] `scirs2-transform/src/gpu.rs` — `GpuPCA::fit/transform/fit_transform` now CPU-backed via `reduction::PCA` (Wave 71)
- [ ] `scirs2-core/src/gpu/` — remaining kernel stubs (large scope; partial wgpu coverage now in place)
- [x] `scirs2-stats/src/gpu/` — GPU-backed statistical operations (Normal/Exponential log-PDF and CDF batch; MIN_GPU_SIZE=1024 threshold; Wave 72)
- [x] `scirs2-fft/src/gpu_fft/wgpu_backend.rs` — `fft_wgpu()` multi-pass Cooley-Tukey DIT; graceful adapter-missing skip (Wave 72)
- [ ] `scirs2-fft/src/backends/{cuda,metal}.rs` — CUDA/Metal FFT backends
- [x] `scirs2-interpolate/src/gpu_accelerated/` — Real wgpu RBF kernel-matrix + evaluation dispatch (Wave 75)
  - Split into mod.rs (CPU types, GpuStats, struct, dispatch logic) + wgpu_rbf.rs (WGSL shaders, GPU pipeline)
  - `wgpu_rbf` feature gate; GPU threshold n_centers*n_queries>=4096; real GpuStats with measured cpu_time_ns/gpu_dispatch_ns/transfer_ns/used_gpu; f64→f32 cast documented; OnceLock<bool> GPU probe cache
  - 5 smoke tests in scirs2-interpolate/tests/wgpu_rbf_smoke.rs; all 1147 tests pass with wgpu_rbf feature
- [x] `scirs2-graph/src/gpu/algorithms.rs` — GPU graph algorithms (already CPU-parallel; wgpu dispatch future) (planned 2026-05-25)
  - **Goal:** Real wgpu BFS + Bellman-Ford SSSP + delta-stepping; CpuParallel branch fixed to call parallel_bfs_atomic
  - **Design:** WGSL bfs_frontier (level-sync, atomicCompareExchange on distances array<atomic<i32>>); sssp_bellman_ford (edge-parallel atomicMin, f32 weights); sssp_delta_stepping (dispatches Bellman-Ford GPU kernel; true bucket-based delta-stepping deferred to Wave 76); parallel_bellman_ford_atomic via AtomicU32 (f32::to_bits trick for non-negative weights); n_edges<4096 CPU threshold
  - **Files:** scirs2-graph/src/gpu/algorithms.rs, scirs2-graph/src/gpu/parallel.rs (new), scirs2-graph/src/gpu/wgpu_shaders.rs (new), scirs2-graph/Cargo.toml (extend gpu feature with dep:wgpu/pollster/bytemuck), scirs2-graph/tests/wgpu_graph_smoke.rs (new)
  - **Tests:** bfs_gpu_matches_cpu_or_skips, sssp_bellman_ford_gpu_matches_cpu_or_skips, cpu_parallel_bfs_actually_parallel, delta_stepping_matches_bellman_ford_or_skips (4 tests gated on cfg(feature = "gpu") with adapter skip) — all pass 1422/1422
  - **Result:** CpuParallel branches now call parallel_bfs_atomic/parallel_bellman_ford_atomic; GPU branches dispatch wgpu shaders with CPU fallback; 0 clippy warnings
- [ ] `scirs2-optimize/src/gpu/` + `distributed_gpu.rs` — GPU optimization

### Distributed computing (requires external MPI/RDMA runtime)
- [ ] `scirs2-linalg/src/distributed/communication.rs` — MPI-backed distributed communication
- [ ] `scirs2-autograd/src/distributed/communication.rs` — MPI-backed autograd communication

### Architectural / large scope
- [ ] `scirs2-core/src/advanced_cloud_storage/providers.rs` — Azure Blob + AWS-specific backends (requires cloud SDKs)
- [ ] `scirs2-core/src/memory_efficient/cross_device.rs` — TPU support (no Pure Rust path)
- [x] `scirs2-core/src/array_protocol/operations.rs` — 15 protocol stubs (matmul/svd/inverse/transpose/reshape etc.) — architectural, backend dispatch (planned 2026-05-25)
  - **Goal:** GpuNdarray<T: GpuScalar> implementing ArrayProtocol::array_function for 14 registered ops (10 wgpu, 4 CPU fallback with documented reason)
  - **Design:** Struct with Arc<wgpu::Buffer>+shape+strides+Arc<WebGPUContext>; GpuScalar sealed for f32; elementwise_binary.wgsl (op_id uniform, 256,1,1); matmul_tiled.wgsl (16×16 shared-memory); reduce_sum.wgsl (two-pass); transpose_tiled.wgsl (32×32 bank-conflict-padded); concatenate axis=0 via copy_buffer_to_buffer; reshape = zero-copy Arc clone; svd/inverse/apply_elementwise/axised-sum → CPU fallback with mandatory comment; operations.rs dispatch hooks at lines 82,224,356,488,620,680,743,780,850,927,984,1033,1084,1135
  - **Files:** scirs2-core/src/array_protocol/gpu_ndarray.rs (new ~1100 LoC), scirs2-core/src/array_protocol/operations.rs (extend ~200 LoC), scirs2-core/src/array_protocol/mod.rs (pub mod gpu_ndarray gated), scirs2-core/tests/gpu_ndarray_dispatch_smoke.rs (new)
  - **Tests:** gpu_ndarray_add_matches_cpu_or_skips, gpu_ndarray_matmul_matches_cpu_or_skips, gpu_ndarray_sum_full_reduction, gpu_ndarray_transpose_2d, gpu_ndarray_reshape_is_zero_copy, gpu_ndarray_concatenate_axis0, gpu_ndarray_svd_falls_back_to_cpu_with_note, gpu_dispatch_below_threshold_uses_cpu (8 tests gated on cfg(feature = "array_protocol_wgpu") with adapter skip)
  - **Risk:** Matmul f32 tolerance 1e-4; concatenate axis>0 deferred to Wave 76 with CPU fallback + doc note
- [ ] `scirs2-neural/src/serving.rs` — `generate_binary` / `generate_shared_library` (runtime codegen via rustc, oversized)

### Specialized algorithms (dedicated future waves)
- [x] ~~`scirs2-integrate/src/dae/solvers.rs:833` — Pantelides DAE index reduction~~ (DONE Wave 73, 2026-05-08)
- [x] ~~`scirs2-integrate/src/sde/runge_kutta_sde.rs:59` — SDE iterated stochastic integrals (Lévy-area approximation)~~ (DONE Wave 73, 2026-05-08)
- [x] **Distributed FFT for 4D+ arrays via slab + pencil + volumetric decomposition** (planned 2026-05-15) (DONE Wave 75, 2026-05-15 — slab/pencil/volumetric helpers in scirs2-fft/src/distributed.rs; 3 integration tests at scirs2-fft/tests/distributed_{4d_slab,5d_pencil,volumetric_nd}.rs)
  - **Goal:** Replace the three unimplemented stubs at `scirs2-fft/src/distributed.rs` (return `FFTError::DimensionError("Dimensions higher than 3 not yet implemented")`) with real slab (4D), pencil (5D), and volumetric (n-D) decompositions.
  - **Design:** Slab for 4D: split axis 0 into n_workers equal slabs; local `fftn_3d` on each slab; alltoall transpose; FFT axis 0. Pencil for 5D: 2D process grid on axes 0+1; local FFT axes 2-4; two alltoall rotations. Volumetric for n-D: recursive transpose-and-FFT splitting the first `ceil(log2(n_workers))` axes. Pure Rust (no MPI). API: `pub fn fftn_distributed<F, D: ndarray::Dimension>(input, axes, workers: Option<usize>) -> FftResult<...>`.
  - **Files:** `scirs2-fft/src/distributed.rs` (replace 3 unimplemented stubs); `scirs2-fft/src/lib.rs` (re-export); optional split-out submodules if > 2000 LoC.
  - **Tests:** `tests/distributed_4d_slab.rs`, `tests/distributed_5d_pencil.rs`, `tests/distributed_volumetric_nd.rs` (6D+7D, dims ≤ 8); each includes a round-trip test.
  - **Risk:** Memory (cap dim ≤ 8 in tests); uneven worker blocks (use truncated sizes, no padding).
- [x] **n-D Voronoi cell vertex + volume computation** (planned 2026-05-15) (DONE Wave 75, 2026-05-15 — vertices_nd/volume_nd/neighbours_nd on VoronoiCell; scirs2-spatial Bowyer-Watson dependency; 4 test files; 4D expensive tests gated #[ignore])
  - **Goal:** Extend `scirs2-interpolate/src/voronoi/voronoi_cell.rs` from 2D/3D-approximation to genuine n-D (n ≥ 2). Add cell-vertex enumeration, cell-volume computation, and neighbour-finding up to 6D.
  - **Design:** Vertices = circumcentres of Delaunay (n+1)-simplices adjacent to each site (solve (n+1)×(n+1) linear system per simplex). Cell volume = convex hull of vertices via Cayley-Menger determinant / QuickHull + simplex-volume sum. Re-use `scirs2-spatial::delaunay::DelaunayTriangulation` (nD Bowyer-Watson, f64, runtime-dim dispatch). API: `VoronoiCell<F>::vertices() -> Vec<Array1<F>>`, `volume() -> F`, `neighbours() -> Vec<usize>`.
  - **Files:** `scirs2-interpolate/src/voronoi/voronoi_cell.rs` (extend); `scirs2-interpolate/src/voronoi/mod.rs` (re-export). Splitrs if > 2000 LoC.
  - **Tests:** `tests/voronoi_3d_cube.rs`, `tests/voronoi_4d_hypercube.rs`, `tests/voronoi_5d_random_sites.rs` (32 pts, volumes sum = bbox volume ±1e-8), `tests/voronoi_neighbours_2d.rs` (backward-compat).
  - **Risk:** Degenerate (co-circular) sites → ε-perturbation; curse of dimensionality → cap 5D in CI.
- [x] `scirs2-linalg/src/autograd/tensor_algebra.rs:249,509` — General tensor contraction (Einstein summation engine) [implemented in `src/autograd/einsum.rs`; 13 tests pass]
- [x] **Gauss-Legendre quadrature orders 4-10 for volume-preserving integrators** (planned 2026-05-16) — *workspace `TODO.md` L864 (DG/spectral/volume-preserving sub-item)*
  - **Goal:** Replace the early-error path at `scirs2-integrate/src/geometric/volume_preserving.rs:590` (`n_quad ≥ 4 → return Err`) with a real implementation supplying Gauss-Legendre nodes + weights for orders n = 1..=10 with exactness for polynomials of degree 2n-1 on [-1, 1].
  - **Design:** Hardcoded double-precision tables (Abramowitz & Stegun / SciPy leggauss) for n ∈ {4..10}; single match dispatch; n=0 and n>10 return IntegrateError::InvalidArgument. No new cross-crate deps.
  - **Files:** `scirs2-integrate/src/geometric/volume_preserving.rs` (lines ~580-650); new `scirs2-integrate/tests/volume_preserving_high_order_quadrature.rs`.
  - **Tests:** 5 tests — exactness for k=0..2n-1, node antisymmetry, weight sum = 2, invalid order errors, IRK-Gauss pendulum Hamiltonian drift ≤ 1e-10 for 1000 steps.
  - **Risk:** transcription errors caught by polynomial-exactness test.

- [x] **Discontinuous Galerkin solver for 1D compressible Euler with HLLC Riemann flux + slope-limiter, validated against Sod shock tube** (completed 2026-05-16) — *workspace `TODO.md` L864*
  - **Goal:** New `scirs2-integrate/src/pde/dg_systems/` module; nodal DG with GLL nodes + HLLC flux + SSPRK3 time integrator + MinmodTvbLimiter; Sod shock tube L1 error ≤ 5e-2 at p=2/N=200.
  - **Design:** `euler_1d.rs` (EulerState/EulerFlux/primitives_to_conservative); `hllc_euler.rs` (Toro HLLC, Davis-Einfeldt wave speeds); `dg_system_solver.rs` (DgSystemConfig, DgSystemSolution, solve_1d_euler_dg, GLL nodes from dg_advanced::entropy_stable); `limiter.rs` (SlopeLimiter trait, MinmodTvbLimiter, StandardPerssonPeraire); SSPRK3/SSPRK4 added to ode/methods/explicit.rs. Sequential sub-steps: B0 SSPRK prerequisite → B1 state/flux → B2 limiter → B3 solver → B4 tests/example/wiring.
  - **Files:** 5 new files under `src/pde/dg_systems/`; `src/pde/mod.rs`; `src/lib.rs`; `src/ode/methods/explicit.rs`; `examples/sod_shock_tube_dg.rs`; 2 new test files.
  - **Tests:** 5 Sod-tube tests (L1 errors p=2/3/4, conservation, positivity); 9 convergence-order tests (3 orders × 3 mesh sizes); 16 in-module unit tests.
  - **Risk:** Gibbs oscillations without limiting (MinmodTvbLimiter default); CFL positivity failure (fallback cfl=0.3).
- [x] `scirs2-neural/src/models/architectures/mamba.rs` — Mamba SSM (implemented — 983 lines, full selective scan + ZOH discretization)
- [x] ~~`scirs2-neural/src/models/diffusion/dpm_solver.rs` — DPM-Solver (dedicated wave planned)~~ (DONE Wave 73, 2026-05-08)
- [x] ~~`scirs2-neural/src/layers/grouped_query_attention.rs` — GQA (dedicated wave planned)~~ (DONE Wave 73, 2026-05-08)
- [x] ~~`scirs2-neural/src/layers/multi_query_attention.rs` — MQA (dedicated wave planned)~~ (DONE Wave 73, 2026-05-08)

---

## v0.5.0 — IN PROGRESS (June 2, 2026)

**Focus**: GPU acceleration, advanced CAS, NUMA, and symbolic AI/ML integrations.

### Completed (Waves 73–77)

#### Wave 73 (2026-05-08)
- [x] `scirs2-core::parallel::numa::par_map_chunks` — typed-result NUMA-locality chunk map (Linux pthread affinity pin, rayon fallback Darwin/WASM); 8 tests
- [x] `scirs2-integrate` Pantelides full graph algorithms — Hopcroft-Karp O(E√V) bipartite matching + Tarjan iterative SCC in `index_reduction.rs`; 13 tests
- [x] `scirs2-integrate::sde` Wiktorsson 2001 truncated-series Lévy-area (`levy_area.rs`, `srk_strong_general`); 10 tests
- [x] `scirs2-spatial::hilbert` 2D+3D Hilbert sort (24-state Butz/Hamilton lookup for 3D; `hilbert_sort_{2d,3d}`); 8 tests
- [x] `scirs2-autograd` `ScalarMulOp` gradient fix; `MatmulEpilogue` + batched-matmul→reduction JIT fusion; 8 tests
- [x] `scirs2-symbolic::regression::discover::predict_parallel` NUMA wire-up via `scirs2_core::par_map_chunks`; 3 tests

#### Wave 74 (2026-05-13)
- [x] `scirs2-cluster/src/subspace_enhanced.rs` LRSC/SSC timeout fix — sign-aware delta convergence early-exit in `power_iter_eig_local`; `min_sigma_sq` parameter to skip sub-threshold modes; 3 LRSC 120s→2s, 2 SSC 120s→33s

#### Wave 75 (2026-05-25)
- [x] `scirs2-interpolate` RBF wgpu — real WGSL kernel-matrix+eval (kernel_id uniform, workgroup 16×16/64); OnceLock GPU probe cache; 5 smoke tests
- [x] `scirs2-graph` GPU BFS+SSSP — real WGSL BFS (level-sync, atomicCompareExchange), Bellman-Ford (edge-parallel, atomicMin f32-bits), delta-stepping; CpuParallel fixed; GPU threshold n_edges<4096; 4 smoke tests
- [x] `scirs2-core` `GpuNdarray<f32>` — 7 WGSL kernels (elementwise add/sub/mul/scalar, naive matmul, two-pass sum, 16×16 transpose); singleton WebGPUContext; CPU fallback for svd/inverse; `array_protocol_wgpu` feature; 8 smoke tests
- [x] `scirs2-symbolic` `eval_batch` real wgpu submission in `compile/gpu.rs`; f64→f32 boundary; GpuError variants; 4 smoke tests
- [x] Distributed FFT >3D (slab/pencil/volumetric in `scirs2-fft/src/distributed.rs`); Voronoi nD cell vertices/volume/neighbours; `cas_tutorial.md` (1488 lines, 13 compile tests)

#### Wave 76 (2026-05-25)
- [x] `scirs2-core/gpu/kernels/mod.rs` — 13 empty WGSL slots filled (Adam/SGD/RMSprop/Adagrad/LAMB/memcpy/fill/reduce_sum/reduce_max/rk4_stages/rk4_combine/error_estimate); 4 smoke tests
- [x] `scirs2-optimize` L-BFGS GPU — `lbfgs_gpu.rs` two-loop recursion via GpuNdarray; `GpuLbfgsState`; f64↔f32 boundary; GPU_LBFGS_THRESHOLD=4096; 2 smoke tests
- [x] `scirs2-symbolic/diffgeom` Riemann tensor `R^μ_{νρσ}` + Ricci trace + Weyl tensor full-n decomposition; 10 integration tests (Schwarzschild, Bianchi identity, trace-free, Kretschner)
- [x] `scirs2-symbolic/neural_priors.rs` `discover_series_prior`/`eval_series_prior`/`series_prior_regularization`; `scirs2-neural/losses/symbolic_prior_loss.rs` `SymbolicPriorLoss`; 8 tests

#### Wave 77 (2026-05-25)
- [x] `scirs2-symbolic/attention/symbolic_alibi.rs` — `alibi_slope`, `alibi_bias_expr`, `alibi_bias_matrix_symbolic`, `verify_symbolic_vs_numerical`; 6 tests
- [x] `scirs2-optimize` CG GPU (`cg_gpu.rs`) + Newton GPU (`newton_gpu.rs`) — dot/direction-update and Hessian-vector matmul via GpuNdarray; 3 smoke tests
- [x] `scirs2-graph` true delta-stepping WGSL — `DELTA_LIGHT_WGSL` + `DELTA_APPLY_WGSL` + `DELTA_HEAVY_WGSL` with changed_flag convergence fix; 2 new smoke tests
- [x] `scirs2-core` GpuNdarray advanced ops — `concat_axis.wgsl` (axis>0, uniform-based stride gather) + `reduce_sum_axis.wgsl` (rank≥3); `gpu_ndarray_shaders.rs` split; 3 smoke tests

### Planned
- [x] ~~Symbolic computing enhancements (Phase 4 diffgeom extensions)~~ (DONE Wave 72, 2026-05-07)
- [x] ~~NUMA scheduler public API (`scirs2-core::par_map_chunks` ecosystem-wide)~~ (DONE Wave 73, 2026-05-08)
- [x] ~~GPU WGSL JIT actual wgpu submission (deferred from v0.4.4)~~ (DONE Wave 75, 2026-05-25)
  - **Goal:** GpuKernel::eval_batch() actually dispatches to wgpu (replacing hardcoded Err("wired in v0.4.5")); to_jit_auto GPU_DISPATCH_THRESHOLD=100_000 becomes real dispatch
  - **Design:** Inline wgpu (adapter→device→pipeline→bind-group→dispatch→readback) in gpu.rs using scirs2-fft pattern; f64→f32 cast at upload, f32→f64 at readback; 2-binding layout (inputs read, outputs rw); dispatch ceil(n_rows/64) workgroups; GpuError extended with NoAdapter(String), EmptyInput, BufferError; GpuError::Unsupported removed from active path
  - **Files:** scirs2-symbolic/src/compile/gpu.rs (480→723 LoC), scirs2-symbolic/tests/wgpu_eval_batch_smoke.rs (new), scirs2-symbolic/tests/cas_tutorial_compile.rs (section_09 updated for Phase 2)
  - **Tests:** eval_batch_constant_kernel_or_skips, eval_batch_transcendental_kernel_or_skips, to_jit_auto_returns_gpu_above_threshold_or_skips, eval_batch_empty_input_is_always_error (4 tests, all pass)

- [x] **State-sync: flip Wave 75 `[~]` items to `[x]` + acknowledge finance pricing residue** (planned 2026-05-16)
  - **Goal:** Bring `TODO.md` in line with Wave 75 reality: flip the 3 Wave 75 `[~]` items (distributed FFT >3D, Voronoi nD, cas_tutorial) to `[x]`; add finance-pricing acceptance note.
  - **Design:** Find and flip the three `[~]` plan blocks for Wave 75 Slices 3/4/5 to `[x]` with DONE annotations. Append a note under a `## Notes` section (creating one if absent) about the Wave 75 finance-pricing code (+913 LoC SABR/Hull-White/Bates/LocalVolatility) accepted and retained.
  - **Files:** `/Users/kitasan/work/scirs/TODO.md` only.
  - **Tests:** none required.

---

## Proposed follow-ups

- [x] Wave 76 Refinement A: scirs2-core/gpu WGSL kernel stubs filled (Adam/SGD/RMSprop/Adagrad/LAMB/memcpy/fill/reduce_sum/reduce_max/rk4_stages/rk4_combine/error_estimate — 13 kernels, 4 smoke tests) (2026-05-25)
- [x] Wave 76 Refinement B: L-BFGS 2-loop recursion GPU-accelerated via GpuNdarray (dot products + vector ops; f64→f32 boundary; GPU_LBFGS_THRESHOLD=4096; 2 smoke tests; CPU path unchanged) (2026-05-25)
- [x] Wave 76 Refinement C(i): SR-as-prior time-series — discover_series_prior (sliding-window feature matrix → SR discovery), eval_series_prior, series_prior_regularization (min-over-formulas penalty); SymbolicPriorLoss in scirs2-neural; 8 tests (2026-05-25)

### Wave 75 Refinement A — Enumerate remaining scirs2-core/gpu stubs
~~Propose Wave 76 stub-check: scan scirs2-core/src/gpu/**/*.rs, classify into {wgpu-ready, needs-design, CPU-only-by-nature}, land wgpu-ready ones.~~ (Done — Wave 76 Refinement A above).

### Wave 75 Refinement B — Split scirs2-optimize GPU into single-GPU vs distributed
- gpu_optimize_single_device: unblocked after Wave 75 dispatch substrate; target L-BFGS hot loop
- gpu_optimize_distributed: blocked on MPI/nccl decisions; keep deferred

### Wave 75 Refinement C — Phase-3 remaining integration backlog (scirs2-symbolic)
Three concrete sub-items: (i) extended SR-as-prior for time-series; (ii) ALiBi positional bias for symbolic attention; (iii) Riemann+Weyl tensor in diffgeom (200-400 LoC each).

- [x] Wave 76 Refinement C(iii): scirs2-symbolic diffgeom Riemann tensor (R^μ_{νρσ}, full 4-index, anti-symmetry + Bianchi identity tests) + Weyl tensor (C_{μνρσ}, trace-free, conformally-flat tests) — 10 integration tests (2026-05-25)
- [x] Wave 77 Refinement C(ii): scirs2-symbolic ALiBi symbolic wrapper — alibi_slope, alibi_bias_expr (LoweredOp tree), alibi_bias_matrix_symbolic, verify_symbolic_vs_numerical; 6 tests (2026-05-25)
- [x] Wave 77: scirs2-graph true GPU delta-stepping SSSP — delta_light_kernel + delta_apply_kernel WGSL (light/heavy edge partition, atomicMin f32-bits, adaptive delta heuristic); 2 additional smoke tests (2026-05-25)
- [x] Wave 77: scirs2-optimize CG + Newton GPU — cg_gpu.rs (dot/direction-update via GpuNdarray, GPU_CG_THRESHOLD=4096), newton_gpu.rs (Hessian-vector matmul GPU); use_gpu fields in options; 3 smoke tests (2026-05-25)
- [x] Wave 77: GpuNdarray advanced ops — concatenate(axis>0) WGSL (uniform-based stride computation, per-element gather) + sum(axis=Some) rank≥3 WGSL (per-output-element axis reduction); 3 new smoke tests (2026-05-25)

---

## Notes

- 2026-05-15 (Wave 75 acceptance): scirs2-integrate/src/specialized/finance/pricing/{monte_carlo,finite_difference,fourier}.rs gained +913 LoC (SABR, Hull-White, Bates, LocalVolatility) outside the approved Wave 75 scope. Code is correct, tests pass, clippy clean — accepted and retained. No revert.
