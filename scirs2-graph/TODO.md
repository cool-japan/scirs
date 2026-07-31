# scirs2-graph TODO

## Status: v0.6.5 Released (July 31, 2026)

Several correctness bugs fixed this cycle, surfaced by a workspace-wide `#[ignore]`-legitimacy audit
followed to ground rather than just re-read:

- `spectral::spectral_clustering` / `spectral_graph::spectral_clustering` now compute a real
  spectral embedding (smallest eigenvectors of the normalized Laplacian) followed by genuine
  Lloyd's-algorithm k-means, instead of a random/stand-in result.
- `algorithms::matching::hungarian_algorithm` now solves the assignment problem exactly via the
  real O(n^3) Hungarian (Kuhn-Munkres) algorithm instead of a random/stand-in result; cross-checked
  against a brute-force solver on random instances up to n=6
  (`test_hungarian_matches_bruteforce_on_random_instances`).
- `advanced::AdvancedProcessor` (`advanced/mod.rs`) now accumulates genuine wall-clock timing
  (`SimplePerformanceMonitor`) and structural/RSS-sampled memory statistics (`execute` /
  `execute_profiled`) instead of fabricated constants.
- `generators::watts_strogatz_graph`'s rewiring step (`src/generators/mod.rs`) checked
  `new_graph.has_node(...)` — always `true` once all `n` nodes are present — instead of
  `has_edge(...)`, so the "already an edge / avoid duplicate" guard could never trigger and the
  rewiring loop hung on ~99.8% of seeds at `p > 0`. Now checks `has_edge`.

See `CHANGELOG.md` `[0.6.5]` for full detail.

The audit below — last performed against source on 2026-07-15 — remains accurate for everything
other than the four items above (the fixes did not add or remove any public API surface).

Note: this file was last audited against `src/` on 2026-07-15. Almost all of the
former "v0.4.0 Roadmap" below has since been implemented (verified by grepping
for the relevant structs/functions); remaining gaps are called out explicitly.

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
- Message-passing framework

Note (2026-07-15 audit): "Graph Isomorphism Network (GIN)" was listed here previously
but no standalone public GIN layer was found in `src/gnn/` — only a private
`GinLocal` helper used internally by the GPS graph-transformer aggregation
(`src/gnn/transformers/gps.rs`). Removed from this list pending a real public
GIN layer.

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
- Influence propagation models (independent cascade, linear threshold) — `influence_maximization`, `simulate_spread` in `src/social.rs`
- Role detection (`role_detection` in `src/social.rs`)
- Echo chamber detection and polarization index (`echo_chamber_detection`, `polarization_index` in `src/social.rs`)

Note (2026-07-15 audit): "Reciprocity metrics" was listed here previously but no
directed-graph reciprocity function (fraction of mutual edges) was found anywhere
in `src/`; every `reciproc*` hit found is unrelated (Mean Reciprocal Rank in
`gnn/kg_completion.rs`, SIMD reciprocal-sqrt in `simd_ops.rs`, "reciprocal of"
doc-comment phrasing). Removed pending a real implementation.

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
- [x] Temporal GNNs for dynamic graph learning (`TemporalGnnModel` in `src/gnn/transformers/temporal_gnn.rs`)
- [x] Time-aware node embeddings (`TimeEncoding` in `src/gnn/transformers/temporal_gnn.rs`)
- [x] Continuous-time dynamic graph models — TGN-style (`MemoryModule` + `TemporalAttention` in `src/gnn/transformers/temporal_gnn.rs`); note (2026-07-15 audit): "JODIE" was listed alongside TGN previously but no JODIE-specific implementation (or any mention of it) was found anywhere in `src/` — only the TGN architecture is present

### Graph Transformers — Implemented in v0.4.0
- [x] Graph Transformer layers with positional encodings
- [x] Expressive power beyond WL-1 test
- [x] Long-range graph dependencies

### Large-Scale Graph Partitioning — Implemented
- [x] Balanced k-way partitioning for distributed processing (`multilevel_kway`, `KwayPartitionResult` in `src/partitioning/multilevel.rs`; "target: 10B+ edges" is the original design goal, not a benchmarked figure)
- [x] METIS-style multilevel partitioning (`multilevel_partition`, `CoarseningStrategy`, `RefinementStrategy` in `src/partitioning/multilevel.rs`)
- [x] Streaming partitioning for dynamic graphs (`streaming_partition`, `StreamingPartitioner` in `src/partitioning/streaming.rs`)
- [x] Distributed graph storage (partitioned adjacency lists) (`DistributedGraph`, `GraphShard`, `build_distributed_graph` in `src/distributed/mod.rs`)

### GPU-Accelerated Graph Algorithms — Partially implemented
- [x] GPU BFS and SSSP (single-source shortest paths) — real wgpu dispatch (`gpu_bfs`, `gpu_sssp_bellman_ford`, `gpu_sssp_delta_stepping` in `src/gpu/algorithms.rs`), CPU-parallel atomic fallback otherwise
- GPU PageRank and betweenness centrality — not implemented (no GPU-specific pagerank/betweenness found under `src/gpu/`)
- GPU sparse matrix operations for spectral methods — partial: a standalone CUDA CSR SpMV primitive exists (`cuda_spmv_csr`, `src/gpu_cuda.rs`, off-by-default `cuda` feature, oxicuda-* backed) but it is not yet wired into the spectral/PageRank algorithm entry points (`spectral.rs`, `measures.rs`)

### Advanced GNN Architectures — Implemented
- [x] Graph Transformers (Graphormer, GPS) — Implemented in v0.4.0
- [x] Equivariant GNNs (E(n)-GNN) for molecular applications (`EgnnLayer` in `src/gnn/equivariant/egnn.rs`; SE(3)-Transformer also present)
- [x] Heterogeneous GNNs for knowledge graph completion (`HgtLayer` in `src/gnn/hgt.rs`, `RgcnLayer` in `src/gnn/rgcn.rs`, `src/gnn/kg_completion.rs`)

### Graph Self-Supervised Learning — Implemented
- [x] Contrastive graph learning (GraphCL, SimGRACE) (`src/ssl/contrastive.rs`: `GraphClConfig`, `simgrace_perturb`, `nt_xent_loss`)
- [x] Graph masked autoencoders (GraphMAE) (`GraphMae`, `GraphMaeConfig` in `src/ssl/masked_autoencoder.rs`)
- [x] Pre-training strategies for downstream tasks (`NodeMaskingPretrainer`, `GraphContextPretrainer`, `AttributeReconstructionObjective` in `src/ssl/pretrain.rs`)

### Hypergraph Neural Networks — Implemented
- [x] Hypergraph convolution layers (`HgnnLayer`, `HgnnNetwork` in `src/hypergraph/neural.rs`)
- [x] Hypergraph attention mechanisms (`HypergraphAttentionLayer`, `HypergraphAttentionNetwork` in `src/hypergraph/attention.rs`)
- [x] Hyperedge prediction (`HyperedgePredictor` in `src/hypergraph/edge_prediction.rs`)

## Newer Modules Not Yet Reflected Above (added since v0.4.0, verified present in `src/`)
- Network alignment: IsoRank and GRASP (`src/alignment/`)
- Signed/directed graph learning: signed spectral (SPONGE), Signed GCN, HOPE/APP directed embeddings (`src/signed_directed/`)
- Graph condensation / dataset distillation: coreset selection, gradient-matching distillation (`src/condensation/`)
- Dynamic graph streams: evolving graphs, link streams, temporal snapshots, temporal Dijkstra (`src/dynamic/`)

## Known Issues

- VF2 subgraph isomorphism may be slow for dense graphs with many automorphisms
- Louvain community detection (`louvain_communities_result`) takes no seed parameter and no RNG usage was found in `src/algorithms/community/louvain.rs`; if run-to-run variation is observed it would stem from `HashMap`/`HashSet` iteration order, not a configurable seed (the previous "seed control recommended" advice was inaccurate — there is no seed to control)
- Some spectral methods require the `parallel` feature for acceptable performance on graphs larger than 100K nodes
