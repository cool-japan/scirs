# TODO List for scirs2-transform

This module provides data transformation utilities for machine learning pre-processing and dimensionality reduction.

## Current Progress
- ✅ Basic data normalization and standardization (min-max, z-score, l1/l2 norm)
- ✅ Normalizer class for fit-transform workflow
- ✅ Feature engineering utilities (polynomial features, binarization, discretization)
- ✅ Power transformers (Box-Cox, Yeo-Johnson)
- ✅ Dimensionality reduction (PCA, TruncatedSVD, LDA)
- ✅ Enhanced scaling transformers (RobustScaler, QuantileTransformer, MaxAbsScaler)
- ✅ Missing value handling (SimpleImputer, MissingIndicator)
- ✅ Categorical encoding (OneHotEncoder, OrdinalEncoder)
- ✅ Feature selection (VarianceThreshold)
- ✅ Unit tests for all functionality

## Data Preprocessing and Scaling

- [x] Enhanced scaling transformers
  - [x] RobustScaler (median and quantile-based scaling)
  - [x] QuantileTransformer (non-linear transformations)
  - [x] PowerTransformer optimizations
  - [x] MaxAbsScaler implementation
  - [ ] Unit scaling with customizable norms
- [ ] Outlier handling
  - [ ] Winsorization
  - [ ] Clipping transformers
  - [ ] Outlier detection integration
  - [ ] Automatic outlier treatment
- [x] Missing value handling
  - [x] SimpleImputer implementation
  - [x] KNNImputer for nearest neighbor imputation
  - [x] IterativeImputer (MICE algorithm)
  - [x] MissingIndicator functionality
- [x] Categorical encoding
  - [x] OneHotEncoder implementation
  - [x] OrdinalEncoder implementation
  - [x] TargetEncoder for supervised encoding
  - [x] BinaryEncoder implementation
  - [ ] BaseNEncoder for compact representations
  - [ ] HashingEncoder implementation

## Feature Engineering

- [ ] Feature generation
  - [ ] Expand polynomial features
  - [ ] Spline transformers
  - [ ] Cyclical feature encoding
  - [ ] Interaction feature generation
  - [ ] Automatic feature crossing
- [x] Feature selection
  - [x] Variance threshold filtering
  - [ ] Univariate feature selection
    - [ ] Chi-squared, F-test, mutual information
    - [ ] ANOVA F-value
    - [ ] Correlation-based filtering
  - [ ] Recursive feature elimination
  - [ ] Model-based selection
    - [ ] Lasso-based selection
    - [ ] Tree-based importance
    - [ ] Permutation importance

## Dimensionality Reduction

- [ ] Linear techniques
  - [ ] Enhanced PCA implementation
    - [ ] Incremental PCA
    - [ ] Sparse PCA
    - [ ] Kernel PCA
    - [ ] Randomized PCA
  - [ ] Factor Analysis
  - [ ] Independent Component Analysis (ICA)
  - [ ] Random projection methods
    - [ ] Gaussian random projection
    - [ ] Sparse random projection
  - [ ] Linear Discriminant Analysis enhancements
    - [ ] Quadratic Discriminant Analysis
    - [ ] Regularized Discriminant Analysis

- [ ] Non-linear techniques
  - [x] t-SNE implementation 
    - [x] Barnes-Hut approximation
    - [x] Multicore implementation
  - [ ] UMAP implementation
    - [ ] Supervised UMAP
    - [ ] Semi-supervised UMAP
  - [ ] Manifold learning methods
    - [ ] Isomap
    - [ ] Locally Linear Embedding (LLE)
    - [ ] Modified LLE
    - [ ] Hessian Eigenmapping
    - [ ] Spectral Embedding
    - [ ] Multi-dimensional Scaling (MDS)
  - [ ] Autoencoders integration

## Text and Image Transformations

- [ ] Text vectorization
  - [ ] CountVectorizer implementation
  - [ ] TfidfVectorizer implementation
  - [ ] HashingVectorizer for large vocabularies
  - [ ] Word embedding transformers
  - [ ] n-gram support
- [ ] Image preprocessing
  - [ ] Patch extraction
  - [ ] HOG feature extraction
  - [ ] SIFT/SURF descriptor integration
  - [ ] Image normalization utilities
  - [ ] Augmentation transformers

## Time Series Transformations

- [ ] Temporal feature extraction
  - [ ] Time series decomposition
  - [ ] Fourier features
  - [ ] Wavelet features
  - [ ] Lag features generation
  - [ ] Rolling statistics
- [ ] Signal transformations
  - [ ] Waveform feature extraction
  - [ ] Spectrogram features
  - [ ] Mel-frequency cepstral coefficients (MFCC)
  - [ ] Constant-Q transform features

## Matrix Decomposition

- [ ] Advanced matrix factorization
  - [ ] Non-negative Matrix Factorization (NMF)
    - [ ] Multiplicative update rules
    - [ ] Coordinate descent solver
    - [ ] Beta-divergence objectives
  - [ ] Dictionary learning
    - [ ] Sparse coding
    - [ ] Mini-batch dictionary learning
  - [ ] Low-rank decompositions
    - [ ] Robust PCA
    - [ ] Matrix completion
  - [ ] Tensor decompositions
    - [ ] CANDECOMP/PARAFAC
    - [ ] Tucker decomposition

## Transform Pipeline

- [ ] Pipeline API
  - [ ] Sequential transformation chains
  - [ ] Parallel feature processing
  - [ ] ColumnTransformer implementation
  - [ ] FeatureUnion implementation
  - [ ] Pipeline persistence
- [ ] Feature selection in pipelines
  - [ ] Integrated feature selection
  - [ ] Sequential feature selection
  - [ ] Recursive feature elimination with cross-validation
- [ ] Hyperparameter optimization
  - [ ] Grid search integration
  - [ ] Pipeline parameter naming convention
  - [ ] Cross-validation utilities

## Data Streaming and Scalability

- [ ] Online learning transformers
  - [ ] Incremental fitting capabilities
  - [ ] Mini-batch transformers
  - [ ] Partial_fit API for all transformers
- [ ] Memory optimization
  - [ ] Memory-mapped transformations
  - [ ] Out-of-core processing
  - [ ] Chunked transformation utilities
- [ ] Parallel processing
  - [ ] Feature-wise parallelism
  - [ ] Sample-wise parallelism
  - [ ] Thread pool management
  - [ ] Parallel decompositions
  - [ ] Joblib integration

## Performance Improvements

- [ ] SIMD acceleration
  - [ ] Vectorized operations for transforms
  - [ ] SIMD-optimized matrix operations
  - [ ] Cache-efficient algorithms
- [ ] GPU acceleration
  - [ ] CUDA support for dimensionality reduction
  - [ ] GPU matrix decompositions
  - [ ] Mixed CPU/GPU pipelines
- [ ] Sparse matrix support
  - [ ] Sparse-aware transformers
  - [ ] Efficient sparse operations
  - [ ] Implicit zero handling
- [ ] Benchmarking infrastructure
  - [ ] Performance tracking
  - [ ] Comparison with scikit-learn
  - [ ] Memory profiling

## API and Usability

- [ ] Rust-idiomatic API design
  - [ ] Trait-based transformer architecture
  - [ ] Generic support for different array types
  - [ ] Error handling best practices
  - [ ] Builder pattern for complex transformers
- [ ] Extended documentation
  - [ ] Algorithm descriptions
  - [ ] Mathematical foundations
  - [ ] Usage examples
  - [ ] Performance considerations
- [ ] Interactive tutorials
  - [ ] Transformation guides
  - [ ] Visualization utilities
  - [ ] Common preprocessing patterns
  - [ ] Dimensionality reduction selection guide

## Long-term Goals

- [ ] Full feature parity with scikit-learn's preprocessing module
- [ ] Advanced integration with deep learning preprocessing
- [ ] Custom transformer framework
- [ ] Automated feature engineering
- [ ] Transfer learning for transformations
- [ ] Self-supervised feature learning
- [ ] Hardware-specific optimizations
- [ ] Integration with data validation frameworks