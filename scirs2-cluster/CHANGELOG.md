# Changelog

## Recent Enhancements

### Added

- **SciPy-compatible kmeans2 function**: Enhanced K-means implementation with multiple initialization strategies
  - `Random`: Generate centroids from Gaussian distribution based on data statistics
  - `Points`: Select random data points as initial centroids
  - `PlusPlus`: K-means++ smart initialization for better convergence

- **Data preprocessing utilities**:
  - `whiten`: Normalize features by dividing by standard deviation (SciPy-compatible)

- **Improved error handling**:
  - `EmptyCluster` error variant for better empty cluster management
  - Configurable empty cluster behavior: `MissingMethod::Warn` vs `MissingMethod::Raise`

- **Enhanced evaluation metrics**:
  - Updated silhouette coefficient implementation with proper i32 label handling
  - Added support for noise points (label -1) in evaluation metrics

### Documentation

- Comprehensive README with examples for all new features
- Added example demonstrating kmeans2 SciPy compatibility
- Added example showing clustering evaluation metrics usage
- Detailed API documentation for all new functions

### Implementation Notes

- Maintained backward compatibility with existing K-means interface
- Added proper random seed handling for reproducible results
- Improved integration with existing metrics module
- Created comprehensive test suite for new functionality

## Future Work

- Add support for custom distance metrics (Mahalanobis, etc.)
- Implement additional initialization methods
- Add GPU acceleration for large datasets
- Enhance parallel processing capabilities