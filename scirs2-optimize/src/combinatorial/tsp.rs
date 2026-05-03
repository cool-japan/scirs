//! Traveling Salesman Problem (TSP) solvers and heuristics.
//!
//! Provides nearest-neighbor construction, 2-opt and 3-opt local search,
//! Or-opt segment relocation, and a Christofides-style MST lower bound.

use scirs2_core::ndarray::Array2;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::error::OptimizeError;

/// Result type for TSP operations.
pub type TspResult<T> = Result<T, OptimizeError>;

// ── Internal priority queue entry for Prim's MST ─────────────────────────────

#[derive(Clone, PartialEq)]
struct PrimEntry {
    cost: f64,
    vertex: usize,
}

impl Eq for PrimEntry {}

impl PartialOrd for PrimEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PrimEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // min-heap: reverse comparison
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
            .then(self.vertex.cmp(&other.vertex))
    }
}

// ── Public helpers ────────────────────────────────────────────────────────────

/// Compute the total length of a tour given a distance matrix.
///
/// The tour is assumed to be a permutation of vertices; the last vertex
/// connects back to the first.
pub fn tour_length(tour: &[usize], dist: &Array2<f64>) -> f64 {
    let n = tour.len();
    if n == 0 {
        return 0.0;
    }
    let mut total = 0.0;
    for i in 0..n {
        let from = tour[i];
        let to = tour[(i + 1) % n];
        total += dist[[from, to]];
    }
    total
}

/// Greedy nearest-neighbour construction heuristic.
///
/// Starting from `start`, repeatedly visit the closest unvisited city.
/// Returns `(tour, length)`.
pub fn nearest_neighbor_heuristic(
    dist: &Array2<f64>,
    start: usize,
) -> TspResult<(Vec<usize>, f64)> {
    let n = dist.nrows();
    if n == 0 {
        return Ok((vec![], 0.0));
    }
    if start >= n {
        return Err(OptimizeError::InvalidInput(format!(
            "start index {start} out of range for {n} cities"
        )));
    }

    let mut visited = vec![false; n];
    let mut tour = Vec::with_capacity(n);
    let mut current = start;
    visited[current] = true;
    tour.push(current);

    for _ in 1..n {
        let mut best_next = None;
        let mut best_dist = f64::INFINITY;
        for j in 0..n {
            if !visited[j] {
                let d = dist[[current, j]];
                if d < best_dist {
                    best_dist = d;
                    best_next = Some(j);
                }
            }
        }
        match best_next {
            Some(next) => {
                visited[next] = true;
                tour.push(next);
                current = next;
            }
            None => break,
        }
    }

    let length = tour_length(&tour, dist);
    Ok((tour, length))
}

/// 2-opt local search.
///
/// Iteratively reverses sub-sequences of the tour whenever doing so reduces
/// the total length.  Returns the improved tour length.
pub fn two_opt(tour: &mut Vec<usize>, dist: &Array2<f64>) -> f64 {
    let n = tour.len();
    if n < 4 {
        return tour_length(tour, dist);
    }

    let mut improved = true;
    while improved {
        improved = false;
        for i in 0..n - 1 {
            for j in i + 2..n {
                // Skip the wrap-around edge (n-1, 0) when j == n-1 and i == 0
                if i == 0 && j == n - 1 {
                    continue;
                }
                let a = tour[i];
                let b = tour[i + 1];
                let c = tour[j];
                let d = tour[(j + 1) % n];
                let current_cost = dist[[a, b]] + dist[[c, d]];
                let new_cost = dist[[a, c]] + dist[[b, d]];
                if new_cost < current_cost - 1e-10 {
                    // Reverse the segment [i+1 .. j]
                    tour[i + 1..=j].reverse();
                    improved = true;
                }
            }
        }
    }

    tour_length(tour, dist)
}

/// Evaluate all 3-opt reconnection types for the three edges defined by
/// positions `i`, `j`, `k` in the tour.
///
/// Returns `Some(new_tour)` if a strictly improving reconnection exists,
/// `None` otherwise.
///
/// The three edges removed are:
///   (tour[i], tour[i+1]),  (tour[j], tour[j+1]),  (tour[k], tour[(k+1)%n])
pub fn three_opt_move(
    dist: &Array2<f64>,
    i: usize,
    j: usize,
    k: usize,
    tour: &[usize],
) -> Option<Vec<usize>> {
    let n = tour.len();
    if n < 6 {
        return None;
    }
    // Validate ordering
    if !(i < j && j < k && k < n) {
        return None;
    }

    let a = tour[i];
    let b = tour[i + 1];
    let c = tour[j];
    let d = tour[j + 1];
    let e = tour[k];
    let f = tour[(k + 1) % n];

    let d0 = dist[[a, b]] + dist[[c, d]] + dist[[e, f]];

    // Segment definitions:
    //   seg1 = tour[0..=i]
    //   seg2 = tour[i+1..=j]
    //   seg3 = tour[j+1..=k]
    //   seg4 = tour[k+1..]

    // We test all 7 non-trivial reconnections (the 8th is the original).
    let candidates: [(f64, u8); 7] = [
        // 1: reverse seg2
        (dist[[a, c]] + dist[[b, d]] + dist[[e, f]], 1),
        // 2: reverse seg3
        (dist[[a, b]] + dist[[c, e]] + dist[[d, f]], 2),
        // 3: reverse seg2 and seg3
        (dist[[a, c]] + dist[[b, e]] + dist[[d, f]], 3),
        // 4: move seg3 between seg1 and seg2
        (dist[[a, d]] + dist[[e, b]] + dist[[c, f]], 4),
        // 5: move seg2 between seg3 and seg4 (reverse of case 4)
        (dist[[a, d]] + dist[[e, c]] + dist[[b, f]], 5),
        // 6: seg1-seg3-seg2-seg4
        (dist[[a, e]] + dist[[d, b]] + dist[[c, f]], 6),
        // 7: reverse all three combined
        (dist[[a, e]] + dist[[d, c]] + dist[[b, f]], 7),
    ];

    let best = candidates
        .iter()
        .min_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(Ordering::Equal));

    let (best_cost, reconnect_type) = match best {
        Some(&(c, t)) => (c, t),
        None => return None,
    };

    if best_cost >= d0 - 1e-10 {
        return None;
    }

    // Build the new tour based on reconnect_type
    let seg1: Vec<usize> = tour[..=i].to_vec();
    let seg2: Vec<usize> = tour[i + 1..=j].to_vec();
    let seg3: Vec<usize> = tour[j + 1..=k].to_vec();
    let seg4: Vec<usize> = if k + 1 < n {
        tour[k + 1..].to_vec()
    } else {
        vec![]
    };

    let mut new_tour = seg1;
    match reconnect_type {
        1 => {
            new_tour.extend(seg2.iter().rev());
            new_tour.extend_from_slice(&seg3);
        }
        2 => {
            new_tour.extend_from_slice(&seg2);
            new_tour.extend(seg3.iter().rev());
        }
        3 => {
            new_tour.extend(seg2.iter().rev());
            new_tour.extend(seg3.iter().rev());
        }
        4 => {
            new_tour.extend_from_slice(&seg3);
            new_tour.extend_from_slice(&seg2);
        }
        5 => {
            new_tour.extend_from_slice(&seg3);
            new_tour.extend(seg2.iter().rev());
        }
        6 => {
            new_tour.extend(seg3.iter().rev());
            new_tour.extend_from_slice(&seg2);
        }
        7 => {
            new_tour.extend(seg3.iter().rev());
            new_tour.extend(seg2.iter().rev());
        }
        _ => unreachable!(),
    }
    new_tour.extend_from_slice(&seg4);
    Some(new_tour)
}

/// Or-opt local search: relocate segments of length 1, 2, or 3.
///
/// For each segment of the given length, try inserting it at every other
/// position in the tour.  Accepts the best improving move found.
/// Returns the improved tour length after convergence.
pub fn or_opt(tour: &mut Vec<usize>, dist: &Array2<f64>) -> f64 {
    let n = tour.len();
    if n < 4 {
        return tour_length(tour, dist);
    }

    let mut improved = true;
    while improved {
        improved = false;
        for seg_len in 1..=3_usize {
            if n < seg_len + 2 {
                continue;
            }
            'outer: for seg_start in 0..n {
                let seg_end = (seg_start + seg_len - 1) % n;
                // Compute cost of removing the segment from its current position
                let prev = if seg_start == 0 { n - 1 } else { seg_start - 1 };
                let after = (seg_end + 1) % n;
                // Skip if wrap-around overlap
                if prev == seg_end || after == seg_start {
                    continue;
                }

                // Removal gain
                let first_city = tour[seg_start];
                let last_city = tour[seg_end];
                let prev_city = tour[prev];
                let after_city = tour[after];

                let remove_cost = dist[[prev_city, first_city]] + dist[[last_city, after_city]]
                    - dist[[prev_city, after_city]];

                // Try inserting after position `ins` (not inside the segment)
                let mut best_gain = 1e-10; // must improve by at least this
                let mut best_ins = None;
                let mut best_reverse = false;

                for ins in 0..n {
                    // Skip positions within or adjacent to segment
                    let in_seg = if seg_start <= seg_end {
                        ins >= seg_start && ins <= seg_end
                    } else {
                        ins >= seg_start || ins <= seg_end
                    };
                    if in_seg || ins == prev {
                        continue;
                    }
                    let ins_next = (ins + 1) % n;
                    let ins_city = tour[ins];
                    let ins_next_city = tour[ins_next];

                    // Forward insertion cost delta
                    let fwd = dist[[ins_city, first_city]] + dist[[last_city, ins_next_city]]
                        - dist[[ins_city, ins_next_city]];
                    let gain_fwd = remove_cost - fwd;
                    if gain_fwd > best_gain {
                        best_gain = gain_fwd;
                        best_ins = Some(ins);
                        best_reverse = false;
                    }

                    // Reversed insertion
                    if seg_len > 1 {
                        let rev = dist[[ins_city, last_city]] + dist[[first_city, ins_next_city]]
                            - dist[[ins_city, ins_next_city]];
                        let gain_rev = remove_cost - rev;
                        if gain_rev > best_gain {
                            best_gain = gain_rev;
                            best_ins = Some(ins);
                            best_reverse = true;
                        }
                    }
                }

                if let Some(ins) = best_ins {
                    // Extract the segment
                    let segment: Vec<usize> =
                        (0..seg_len).map(|k| tour[(seg_start + k) % n]).collect();
                    let seg_set: std::collections::HashSet<usize> =
                        segment.iter().cloned().collect();

                    // Build new tour without the segment, then insert
                    let remaining: Vec<usize> = tour
                        .iter()
                        .cloned()
                        .filter(|v| !seg_set.contains(v))
                        .collect();

                    // Find insertion position in remaining
                    let ins_city = tour[ins];
                    let ins_pos = remaining.iter().position(|&v| v == ins_city).unwrap_or(0);

                    let mut new_tour: Vec<usize> = Vec::with_capacity(n);
                    new_tour.extend_from_slice(&remaining[..=ins_pos]);
                    if best_reverse {
                        new_tour.extend(segment.iter().rev());
                    } else {
                        new_tour.extend_from_slice(&segment);
                    }
                    if ins_pos + 1 < remaining.len() {
                        new_tour.extend_from_slice(&remaining[ins_pos + 1..]);
                    }

                    if new_tour.len() == n {
                        *tour = new_tour;
                        improved = true;
                        break 'outer;
                    }
                }
            }
        }
    }

    tour_length(tour, dist)
}

/// Compute a minimum spanning tree lower bound using Prim's algorithm.
///
/// The MST weight is a classical lower bound for TSP on metric instances.
pub fn mst_lower_bound(dist: &Array2<f64>) -> f64 {
    let n = dist.nrows();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return 0.0;
    }

    let mut in_mst = vec![false; n];
    let mut min_edge = vec![f64::INFINITY; n];
    min_edge[0] = 0.0;

    let mut heap: BinaryHeap<PrimEntry> = BinaryHeap::new();
    heap.push(PrimEntry {
        cost: 0.0,
        vertex: 0,
    });

    let mut mst_weight = 0.0;

    while let Some(PrimEntry { cost, vertex }) = heap.pop() {
        if in_mst[vertex] {
            continue;
        }
        in_mst[vertex] = true;
        mst_weight += cost;

        for j in 0..n {
            if !in_mst[j] {
                let d = dist[[vertex, j]];
                if d < min_edge[j] {
                    min_edge[j] = d;
                    heap.push(PrimEntry { cost: d, vertex: j });
                }
            }
        }
    }

    mst_weight
}

/// High-level TSP solver that chains NN heuristic → 2-opt → Or-opt.
pub struct TspSolver {
    dist: Array2<f64>,
}

impl TspSolver {
    /// Create a new solver with the given distance matrix.
    ///
    /// # Errors
    /// Returns an error if the matrix is not square.
    pub fn new(dist: Array2<f64>) -> TspResult<Self> {
        if dist.nrows() != dist.ncols() {
            return Err(OptimizeError::InvalidInput(
                "Distance matrix must be square".to_string(),
            ));
        }
        Ok(Self { dist })
    }

    /// Solve using nearest-neighbour construction followed by 2-opt and Or-opt.
    ///
    /// Tries every city as a starting vertex for NN and keeps the best result.
    pub fn solve(&self) -> TspResult<(Vec<usize>, f64)> {
        let n = self.dist.nrows();
        if n == 0 {
            return Ok((vec![], 0.0));
        }

        let mut best_tour = vec![];
        let mut best_len = f64::INFINITY;

        for start in 0..n {
            let (mut tour, _) = nearest_neighbor_heuristic(&self.dist, start)?;
            two_opt(&mut tour, &self.dist);
            or_opt(&mut tour, &self.dist);
            let len = tour_length(&tour, &self.dist);
            if len < best_len {
                best_len = len;
                best_tour = tour;
            }
        }

        Ok((best_tour, best_len))
    }

    /// Return the MST-based lower bound on the optimal tour length.
    pub fn lower_bound(&self) -> f64 {
        mst_lower_bound(&self.dist)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn square_dist() -> Array2<f64> {
        // 4-city square: optimal tour length = 4.0
        array![
            [0.0, 1.0, 1.414, 1.0],
            [1.0, 0.0, 1.0, 1.414],
            [1.414, 1.0, 0.0, 1.0],
            [1.0, 1.414, 1.0, 0.0]
        ]
    }

    #[test]
    fn test_tour_length() {
        let dist = square_dist();
        let tour = vec![0, 1, 2, 3];
        let len = tour_length(&tour, &dist);
        // 0→1 + 1→2 + 2→3 + 3→0 = 1+1+1+1 = 4
        assert!((len - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_nearest_neighbor() {
        let dist = square_dist();
        let (tour, len) = nearest_neighbor_heuristic(&dist, 0).expect("unexpected None or Err");
        assert_eq!(tour.len(), 4);
        assert!(len > 0.0);
    }

    #[test]
    fn test_two_opt_improves() {
        let dist = square_dist();
        // A suboptimal tour: 0→2→1→3 has length 1.414+1+1.414+1 = 4.828
        let mut tour = vec![0, 2, 1, 3];
        let original_len = tour_length(&tour, &dist);
        let new_len = two_opt(&mut tour, &dist);
        assert!(new_len <= original_len + 1e-9);
    }

    #[test]
    fn test_or_opt() {
        let dist = square_dist();
        let mut tour = vec![0, 1, 2, 3];
        let len = or_opt(&mut tour, &dist);
        assert!(len > 0.0);
        assert_eq!(tour.len(), 4);
    }

    #[test]
    fn test_mst_lower_bound() {
        let dist = square_dist();
        let lb = mst_lower_bound(&dist);
        // MST of the square has weight 3.0 (three unit edges)
        assert!(lb > 0.0);
        assert!(lb <= 4.0 + 1e-6); // must be ≤ optimal tour length
    }

    #[test]
    fn test_solver_small() {
        let dist = square_dist();
        let solver = TspSolver::new(dist).expect("failed to create solver");
        let (tour, len) = solver.solve().expect("unexpected None or Err");
        assert_eq!(tour.len(), 4);
        // Optimal is 4.0
        assert!(len <= 4.5);
    }

    #[test]
    fn test_three_opt_move() {
        let dist = square_dist();
        // With only 4 nodes any 3-opt call would require i<j<k<4
        // Use a larger tour to exercise the logic
        let n = 6;
        let mut big_dist = Array2::<f64>::zeros((n, n));
        for r in 0..n {
            for c in 0..n {
                if r != c {
                    let dx = (r as f64) - (c as f64);
                    big_dist[[r, c]] = dx.abs();
                }
            }
        }
        let tour: Vec<usize> = vec![0, 1, 2, 3, 4, 5];
        // Just test it runs without panic
        let _ = three_opt_move(&big_dist, 0, 2, 4, &tour);
    }

    #[test]
    fn test_invalid_start() {
        let dist = square_dist();
        assert!(nearest_neighbor_heuristic(&dist, 10).is_err());
    }

    #[test]
    fn test_empty_tour() {
        let dist: Array2<f64> = Array2::zeros((0, 0));
        let (tour, len) = nearest_neighbor_heuristic(&dist, 0).expect("unexpected None or Err");
        assert!(tour.is_empty());
        assert_eq!(len, 0.0);
    }
}
