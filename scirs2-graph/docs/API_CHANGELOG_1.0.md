# API Changelog for scirs2-graph v1.0

This document tracks all API changes made for the 1.0 release and provides migration guidance.

## Summary

The v1.0 release introduces important API improvements focused on:

1. **Standardized return types** for community detection algorithms
2. **Explicit deprecation warnings** for legacy APIs
3. **Stability attributes** for all public interfaces
4. **Consistent naming conventions** across algorithms
5. **Enhanced documentation** with complexity analysis

## Breaking Changes (Deprecated APIs)

### 1. Community Detection Algorithms

#### Louvain Communities
**Legacy API (Deprecated)**:
```rust
pub fn louvain_communities<N, E, Ix>(graph: &Graph<N, E, Ix>) -> CommunityStructure<N>
```

**New API (Stable)**:
```rust
pub fn louvain_communities_result<N, E, Ix>(graph: &Graph<N, E, Ix>) -> CommunityResult<N>
```

**Migration**:
```rust
// Old way
let communities = louvain_communities(&graph);
let modularity = communities.modularity;

// New way  
let result = louvain_communities_result(&graph);
let modularity = result.quality_score.unwrap_or(0.0);
let communities = result.as_community_sets();
```

#### Label Propagation
**Legacy API (Deprecated)**:
```rust
pub fn label_propagation<N, E, Ix>(graph: &Graph<N, E, Ix>, max_iter: usize) -> HashMap<N, usize>
```

**New API (Stable)**:
```rust
pub fn label_propagation_result<N, E, Ix>(graph: &Graph<N, E, Ix>, max_iter: usize) -> CommunityResult<N>
```

**Migration**:
```rust
// Old way
let node_communities = label_propagation(&graph, 100);

// New way
let result = label_propagation_result(&graph, 100);
let node_communities = &result.node_communities;
let communities = &result.communities;
```

### 2. Shortest Path Algorithms

#### Shortest Path Function
**Legacy API (Deprecated)**:
```rust
pub fn shortest_path<N, E, Ix>(graph: &Graph<N, E, Ix>, source: &N, target: &N) -> Result<Option<Path<N, E>>>
```

**New API (Stable)**:
```rust
pub fn dijkstra_path<N, E, Ix>(graph: &Graph<N, E, Ix>, source: &N, target: &N) -> Result<Option<Path<N, E>>>
```

**Migration**:
```rust
// Old way
let path = shortest_path(&graph, &start, &end)?;

// New way
let path = dijkstra_path(&graph, &start, &end)?;
```

## New Features

### 1. Standardized Community Result Type

```rust
pub struct CommunityResult<N: Node> {
    /// Map from node to community ID
    pub node_communities: HashMap<N, usize>,
    /// The communities as sets of nodes
    pub communities: Vec<HashSet<N>>,
    /// Number of communities found
    pub num_communities: usize,
    /// Quality metric for the community structure (e.g., modularity)
    pub quality_score: Option<f64>,
    /// Additional metadata about the communities
    pub metadata: HashMap<String, f64>,
}
```

### 2. Stability Attributes

All public APIs now include stability attributes:

```rust
// Stable APIs
stable!("graph_core", "0.1.0-beta.2");
pub use algorithms::{
    dijkstra_path, louvain_communities_result, label_propagation_result,
    // ... other stable functions
};

// Experimental APIs
unstable!("graph_embeddings", "none");
pub use embeddings::{
    DeepWalk, Node2Vec, 
    // ... experimental features
};
```

### 3. Enhanced Documentation

All algorithms now include:
- **Time complexity** analysis
- **Space complexity** analysis
- **Usage examples** in documentation
- **Performance characteristics**

Example:
```rust
/// Finds the shortest path between source and target nodes using Dijkstra's algorithm
///
/// # Time Complexity
/// O((V + E) log V) where V is the number of vertices and E is the number of edges.
///
/// # Space Complexity
/// O(V) for the distance array and predecessor tracking.
///
/// # Example
/// ```rust
/// use scirs2_graph::{Graph, dijkstra_path};
/// 
/// let mut graph = Graph::new();
/// // ... add nodes and edges ...
/// let path = dijkstra_path(&graph, &start, &end)?;
/// ```
pub fn dijkstra_path<N, E, Ix>(...) -> Result<Option<Path<N, E>>>
```

## Deprecation Timeline

### Phase 1: v0.1.0-beta.2 (Current)
- ✅ Add deprecation warnings to legacy APIs
- ✅ Introduce new standardized APIs
- ✅ Update documentation with migration examples

### Phase 2: v1.0.0-rc.1 (Upcoming)
- 🔄 Migration tool for automated code updates
- 🔄 Comprehensive testing of migration paths
- 🔄 Final documentation review

### Phase 3: v1.0.0 (Stable Release)
- ✅ Legacy APIs remain with deprecation warnings
- ✅ Clear timeline for removal in v2.0
- ✅ Stability guarantees for new APIs

### Phase 4: v2.0.0 (Future)
- 🔮 Remove deprecated APIs
- 🔮 Additional breaking changes if needed
- 🔮 New major features

## API Organization

### Stable APIs (1.0 Guarantee)
- **Core Graph Types**: `Graph`, `DiGraph`, `MultiGraph`, etc.
- **Basic Algorithms**: BFS, DFS, Dijkstra, Floyd-Warshall
- **Centrality Measures**: Betweenness, closeness, eigenvector, PageRank
- **Community Detection**: Louvain, label propagation (new APIs)
- **Flow Algorithms**: Max flow, min cut
- **Graph Generators**: Erdős-Rényi, Barabási-Albert, etc.
- **Memory Optimization**: Compressed representations, profiling
- **Performance Monitoring**: Metrics, profiling tools

### Experimental APIs (Subject to Change)
- **Graph Embeddings**: Node2Vec, DeepWalk
- **Layout Algorithms**: Spring layout, spectral layout
- **Temporal Graphs**: Time-based operations
- **Advanced Analytics**: Some specialized algorithms

## Migration Tools

### Automated Migration Script
A migration tool will be provided to automatically update code:

```bash
# Install migration tool
cargo install scirs2-graph-migrate

# Run migration on your codebase
scirs2-graph-migrate --input src/ --output src_migrated/

# Or in-place migration
scirs2-graph-migrate --input src/ --in-place
```

### Manual Migration Patterns

#### Pattern 1: Community Detection Results
```rust
// Before
let communities = louvain_communities(&graph);
for (node, community_id) in &communities.node_communities {
    println!("Node {:?} in community {}", node, community_id);
}

// After
let result = louvain_communities_result(&graph);
for (node, community_id) in &result.node_communities {
    println!("Node {:?} in community {}", node, community_id);
}
// Plus access to additional information:
println!("Found {} communities", result.num_communities);
println!("Modularity: {:.3}", result.quality_score.unwrap_or(0.0));
```

#### Pattern 2: Function Renaming
```rust
// Before
use scirs2_graph::shortest_path;
let path = shortest_path(&graph, &start, &end)?;

// After
use scirs2_graph::dijkstra_path;
let path = dijkstra_path(&graph, &start, &end)?;
```

## Compatibility Guarantees

### Semantic Versioning
- **1.x.y releases**: Only backward-compatible additions and bug fixes
- **2.0.0 release**: May include breaking changes with clear migration path

### API Stability Levels
1. **Stable**: Guaranteed API compatibility until next major version
2. **Unstable**: May change in minor releases, clearly marked
3. **Deprecated**: Will be removed in next major version

### Performance Guarantees
- Performance improvements are always allowed
- Performance regressions are considered bugs
- Algorithm complexity guarantees remain stable

## Getting Help

### Resources
- **Migration Guide**: `docs/NETWORKX_MIGRATION_GUIDE.md`
- **Performance Benchmarks**: `docs/PERFORMANCE_BENCHMARKS.md`
- **Usage Examples**: `docs/USAGE_EXAMPLES.md`
- **API Documentation**: Generated docs with examples

### Support Channels
- GitHub Issues for bug reports and questions
- Documentation feedback via pull requests
- Community discussions in project forums

## Conclusion

The v1.0 release provides a stable, high-performance foundation for graph processing in Rust while maintaining clear upgrade paths for existing code. The new APIs offer improved consistency, better type safety, and enhanced functionality while preserving the performance characteristics that make scirs2-graph an excellent choice for scientific computing applications.