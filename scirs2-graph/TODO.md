# scirs2-graph TODO

## Status: v0.3.4 Released (March 18, 2026) — v0.4.3 in progress (2026-05-03)

## v0.3.3 Completed

### Community Detection
- Louvain method (modularity optimization)
- Girvan-Newman algorithm (edge betweenness)
- Label propagation
- Infomap algorithm
- Fluid communities
- Hierarchical clustering

### Graph Neural Networks
- Graph Convolutional Network (GCN)
- Graph Attention Network (GAT)
- GraphSAGE (inductive representation learning)
- Graph Isomorphism Network (GIN)
- Message-passing framework

### Graph Embeddings
- Node2Vec random walk embeddings
- DeepWalk
- Spectral embeddings
- Diffusion-based embeddings

### Graph Isomorphism
- VF2 algorithm for graph/subgraph isomorphism
- Subgraph matching with label/attribute constraints

### Spectral Graph Theory
- Graph Laplacian and normalized Laplacian
- Spectral clustering (including algebraic connectivity)
- Graph Fourier transform
- Graph wavelets
- Graph filtering in spectral domain

### Network Flow
- Ford-Fulkerson, Dinic's algorithm, push-relabel
- Minimum-cost flow
- Maximum matching extensions

### Graph Visualization
- SVG output with customizable layouts
- DOT format for Graphviz
- Force-directed, circular, hierarchical layout algorithms

### Heterogeneous and Knowledge Graphs
- Heterogeneous graph representation
- Knowledge graph modeling with entity/relation types
- Type-aware traversal and queries

### Temporal Graphs
- TemporalGraph data structure
- Snapshot queries at specific timestamps
- Temporal path finding
- Dynamic graph algorithms

### Hypergraphs
- Hypergraph representation
- Hypergraph algorithms (hyperedge connectivity, centrality)

### Social Network Analysis
- Influence propagation models (independent cascade, linear threshold)
- Role detection
- Reciprocity metrics

### Additional Graph Algorithms
- Domination problems (dominating sets, independent sets)
- Planarity testing (LR-planarity)
- Algebraic graph theory (characteristic polynomial, graph spectrum)
- Graph reliability and robustness analysis (percolation, connectivity)
- Network sampling algorithms (snowball, forest-fire, random walk sampling)

### Scale and Performance
- CSR representation for cache-efficient traversal
- Rayon parallel processing for core algorithms
- Memory profiling tooling
- Streaming graph processing for large datasets

## v0.4.0 Roadmap

### Temporal Graph Neural Networks — Implemented in v0.4.0
- [x] Temporal GNNs for dynamic graph learning
- [x] Time-aware node embeddings
- [x] Continuous-time dynamic graph models (JODIE, TGN)

### Graph Transformers — Implemented in v0.4.0
- [x] Graph Transformer layers with positional encodings
- [x] Expressive power beyond WL-1 test
- [x] Long-range graph dependencies

### Large-Scale Graph Partitioning
- Balanced k-way partitioning for distributed processing (target: 10B+ edges)
- METIS-style multilevel partitioning
- Streaming partitioning for dynamic graphs
- Distributed graph storage (partitioned adjacency lists)

### GPU-Accelerated Graph Algorithms
- GPU BFS and SSSP (single-source shortest paths)
- GPU PageRank and betweenness centrality
- GPU sparse matrix operations for spectral methods

### Advanced GNN Architectures — Partially implemented in v0.4.0
- [x] Graph Transformers (Graphormer, GPS) — Implemented in v0.4.0
- Equivariant GNNs (E(n)-GNN) for molecular applications
- Heterogeneous GNNs for knowledge graph completion

### Graph Self-Supervised Learning
- Contrastive graph learning (GraphCL, SimGRACE)
- Graph masked autoencoders (GraphMAE)
- Pre-training strategies for downstream tasks

### Hypergraph Neural Networks
- Hypergraph convolution layers
- Hypergraph attention mechanisms
- Hyperedge prediction

## Known Issues

- VF2 subgraph isomorphism may be slow for dense graphs with many automorphisms
- Louvain community detection is non-deterministic; seed control recommended for reproducibility
- Some spectral methods require the `parallel` feature for acceptable performance on graphs larger than 100K nodes
