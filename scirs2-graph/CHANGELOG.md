# Changelog

All notable changes to scirs2-graph will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `CommunityResult` type for standardized community detection return values
- Stability attributes infrastructure for tracking API stability
- Comprehensive deprecation warnings for APIs changing in v1.0
- Performance benchmarks against NetworkX and igraph
- Large graph stress testing suite (>1M nodes)
- Memory profiling and optimization tools
- Cross-platform build verification in CI

### Changed
- Community detection algorithms now support unified return type via `CommunityResult`
- Improved documentation with algorithm complexity analysis
- Enhanced examples showing common workflows

### Deprecated
- `shortest_path` and `shortest_path_digraph` functions - use `dijkstra_path` variants for future compatibility
- Direct usage of `CommunityStructure` - use `CommunityResult` for new code

### Fixed
- Memory efficiency improvements for large graphs
- Numerical accuracy issues in certain edge cases

## [0.1.0-beta.2] - 2024-01-26

### Added
- Complete graph data structures:
  - Undirected and directed graphs
  - Multi-graphs with parallel edges
  - Bipartite graphs
  - Hypergraphs
  - Temporal graphs
  - Attributed graphs
- Core algorithms:
  - Traversal: BFS, DFS, bidirectional search
  - Shortest paths: Dijkstra, A*, Floyd-Warshall, k-shortest paths
  - Connectivity: Components, articulation points, bridges
  - Spanning trees: Kruskal, Prim
  - Flow: Ford-Fulkerson, Dinic, push-relabel
  - Matching: Bipartite, maximum cardinality, stable marriage
  - Centrality: Degree, betweenness, closeness, eigenvector, PageRank, HITS
- Advanced analytics:
  - Community detection: Louvain, label propagation, Infomap, fluid communities
  - Spectral methods: Laplacian, spectral clustering
  - Graph isomorphism: VF2 algorithm
  - Motif finding
- Performance features:
  - Multi-threaded algorithms via Rayon
  - SIMD acceleration where applicable
  - Memory-efficient operations
  - Streaming graph processing
- I/O support:
  - GraphML, GML, DOT, JSON formats
  - Edge list and adjacency list
  - Matrix Market format

### Performance
- Optimized for graphs with millions of nodes
- Parallel processing for compute-intensive algorithms
- Cache-friendly data structures

### Documentation
- Comprehensive API documentation
- Usage examples for all major features
- Migration guide from NetworkX
- Algorithm complexity reference

## [0.0.1] - 2023-12-01

### Added
- Initial project structure
- Basic graph types
- Simple traversal algorithms