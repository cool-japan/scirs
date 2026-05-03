//! 0/1 Knapsack and fractional knapsack solvers.
//!
//! Provides exact DP, greedy heuristic, branch-and-bound, and the classical
//! fractional (continuous) knapsack.  Multi-dimensional knapsack is also
//! supported via `MultiKnapsackProblem`.

use crate::error::OptimizeError;

/// Result type for knapsack operations.
pub type KnapsackResult<T> = Result<T, OptimizeError>;

// ── Item type ─────────────────────────────────────────────────────────────────

/// A single knapsack item with integer weight and value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KnapsackItem {
    /// Non-negative weight of this item.
    pub weight: u64,
    /// Non-negative value of this item.
    pub value: u64,
}

// ── Exact DP ─────────────────────────────────────────────────────────────────

/// Solve the 0/1 knapsack exactly using bottom-up DP in O(n · W) time and space.
///
/// Returns `(total_value, selected_indices)`.
pub fn knapsack_dp(items: &[KnapsackItem], capacity: u64) -> KnapsackResult<(u64, Vec<usize>)> {
    let n = items.len();
    let w = capacity as usize;

    // dp[i][c] = max value using first i items with capacity c
    // Use rolling array to reduce memory: dp_prev and dp_curr
    // But we need backtracking, so store the full table.
    // For very large W we guard against allocation failure.
    let table_size = (n + 1).saturating_mul(w + 1);
    if table_size > 500_000_000 {
        return Err(OptimizeError::InvalidInput(format!(
            "DP table size {table_size} exceeds 500M; use branch-and-bound for large capacities"
        )));
    }

    // Flat row-major table: dp[i*(w+1) + c]
    let mut dp = vec![0u64; (n + 1) * (w + 1)];

    for i in 1..=n {
        let iw = items[i - 1].weight as usize;
        let iv = items[i - 1].value;
        for c in 0..=w {
            let without = dp[(i - 1) * (w + 1) + c];
            let with_item = if iw <= c {
                dp[(i - 1) * (w + 1) + c - iw].saturating_add(iv)
            } else {
                0
            };
            dp[i * (w + 1) + c] = without.max(with_item);
        }
    }

    // Backtrack to find selected items
    let mut selected = Vec::new();
    let mut remaining = w;
    for i in (1..=n).rev() {
        if dp[i * (w + 1) + remaining] != dp[(i - 1) * (w + 1) + remaining] {
            selected.push(i - 1);
            let iw = items[i - 1].weight as usize;
            remaining = remaining.saturating_sub(iw);
        }
    }
    selected.reverse();

    let total = dp[n * (w + 1) + w];
    Ok((total, selected))
}

// ── Fractional knapsack ───────────────────────────────────────────────────────

/// Solve the fractional knapsack in O(n log n) via greedy value-density sort.
///
/// Items may be taken in fractions.  Returns the maximum achievable value.
pub fn fractional_knapsack(items: &[KnapsackItem], capacity: u64) -> f64 {
    if capacity == 0 || items.is_empty() {
        return 0.0;
    }

    // Sort by value/weight ratio descending; items with weight 0 come first
    let mut indexed: Vec<(usize, f64)> = items
        .iter()
        .enumerate()
        .map(|(i, it)| {
            let ratio = if it.weight == 0 {
                f64::INFINITY
            } else {
                it.value as f64 / it.weight as f64
            };
            (i, ratio)
        })
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut remaining = capacity as f64;
    let mut total_value = 0.0;

    for (idx, _ratio) in &indexed {
        let item = &items[*idx];
        if item.weight == 0 {
            total_value += item.value as f64;
            continue;
        }
        let take = (remaining / item.weight as f64).min(1.0);
        total_value += take * item.value as f64;
        remaining -= take * item.weight as f64;
        if remaining <= 0.0 {
            break;
        }
    }

    total_value
}

// ── Greedy heuristic ─────────────────────────────────────────────────────────

/// Greedy 0/1 knapsack: sort by value/weight ratio and greedily include items.
///
/// This is the natural integer rounding of the fractional solution and
/// provides a simple O(n log n) heuristic.  Returns `(total_value, selected_indices)`.
pub fn knapsack_greedy(items: &[KnapsackItem], capacity: u64) -> (u64, Vec<usize>) {
    if capacity == 0 || items.is_empty() {
        return (0, vec![]);
    }

    let mut indexed: Vec<(usize, f64)> = items
        .iter()
        .enumerate()
        .map(|(i, it)| {
            let ratio = if it.weight == 0 {
                f64::INFINITY
            } else {
                it.value as f64 / it.weight as f64
            };
            (i, ratio)
        })
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut remaining = capacity;
    let mut total = 0u64;
    let mut selected = Vec::new();

    for (idx, _) in &indexed {
        let item = &items[*idx];
        if item.weight <= remaining {
            selected.push(*idx);
            remaining -= item.weight;
            total += item.value;
        }
    }

    selected.sort_unstable();
    (total, selected)
}

// ── Branch-and-bound ─────────────────────────────────────────────────────────

/// Node in the B&B search tree.
#[derive(Debug, Clone)]
struct BbNode {
    level: usize,
    value: u64,
    weight: u64,
    bound: f64,
    taken: Vec<bool>,
}

/// Compute the LP-relaxation upper bound from node `level` onwards.
fn lp_bound(
    items: &[KnapsackItem],
    sorted_indices: &[usize],
    level: usize,
    value: u64,
    weight: u64,
    capacity: u64,
) -> f64 {
    if weight > capacity {
        return 0.0;
    }
    let mut remaining = (capacity - weight) as f64;
    let mut bound = value as f64;

    for &idx in sorted_indices.iter().skip(level) {
        let item = &items[idx];
        if item.weight as f64 <= remaining {
            bound += item.value as f64;
            remaining -= item.weight as f64;
        } else {
            // Take fractional part
            if item.weight > 0 {
                bound += remaining * (item.value as f64 / item.weight as f64);
            }
            break;
        }
    }
    bound
}

/// Exact branch-and-bound for 0/1 knapsack with LP-relaxation bounding.
///
/// Returns `(total_value, selected_indices)`.
pub fn knapsack_branch_bound(
    items: &[KnapsackItem],
    capacity: u64,
) -> KnapsackResult<(u64, Vec<usize>)> {
    let n = items.len();
    if n == 0 || capacity == 0 {
        return Ok((0, vec![]));
    }

    // Sort by value/weight ratio descending for tighter bounds
    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| {
        let ra = if items[a].weight == 0 {
            f64::INFINITY
        } else {
            items[a].value as f64 / items[a].weight as f64
        };
        let rb = if items[b].weight == 0 {
            f64::INFINITY
        } else {
            items[b].value as f64 / items[b].weight as f64
        };
        rb.partial_cmp(&ra).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut best_value = 0u64;
    let mut best_taken = vec![false; n];

    // Initial greedy solution as lower bound
    {
        let (gv, gi) = knapsack_greedy(items, capacity);
        best_value = gv;
        for idx in gi {
            best_taken[idx] = true;
        }
    }

    // Iterative DFS using an explicit stack
    let root = BbNode {
        level: 0,
        value: 0,
        weight: 0,
        bound: lp_bound(items, &sorted_indices, 0, 0, 0, capacity),
        taken: vec![false; n],
    };

    let mut stack: Vec<BbNode> = vec![root];

    while let Some(node) = stack.pop() {
        if node.level == n {
            if node.value > best_value {
                best_value = node.value;
                best_taken = node.taken.clone();
            }
            continue;
        }

        if node.bound <= best_value as f64 {
            continue;
        }

        let item_idx = sorted_indices[node.level];
        let item = &items[item_idx];

        // Branch: include item
        if node.weight + item.weight <= capacity {
            let mut taken_with = node.taken.clone();
            taken_with[item_idx] = true;
            let new_value = node.value + item.value;
            let new_weight = node.weight + item.weight;
            let new_bound = lp_bound(
                items,
                &sorted_indices,
                node.level + 1,
                new_value,
                new_weight,
                capacity,
            );
            if new_bound > best_value as f64 {
                stack.push(BbNode {
                    level: node.level + 1,
                    value: new_value,
                    weight: new_weight,
                    bound: new_bound,
                    taken: taken_with,
                });
            }
        }

        // Branch: exclude item
        let excl_bound = lp_bound(
            items,
            &sorted_indices,
            node.level + 1,
            node.value,
            node.weight,
            capacity,
        );
        if excl_bound > best_value as f64 {
            stack.push(BbNode {
                level: node.level + 1,
                value: node.value,
                weight: node.weight,
                bound: excl_bound,
                taken: node.taken.clone(),
            });
        }
    }

    let selected: Vec<usize> = (0..n).filter(|&i| best_taken[i]).collect();
    Ok((best_value, selected))
}

// ── Multi-dimensional knapsack ────────────────────────────────────────────────

/// An item for the multi-dimensional knapsack.
#[derive(Debug, Clone)]
pub struct MultiKnapsackItem {
    /// Weight in each dimension.
    pub weights: Vec<u64>,
    /// Value of the item.
    pub value: u64,
}

/// Multi-dimensional 0/1 knapsack solved by greedy + local search.
///
/// This is NP-hard and no polynomial exact algorithm is known for d>1.
/// We use a greedy value/weight-norm ratio followed by 1-swap improvement.
///
/// Returns `(total_value, selected_indices)`.
pub fn multi_knapsack_greedy(
    items: &[MultiKnapsackItem],
    capacities: &[u64],
) -> KnapsackResult<(u64, Vec<usize>)> {
    let n = items.len();
    let d = capacities.len();
    if n == 0 || d == 0 {
        return Ok((0, vec![]));
    }

    // Validate dimensions
    for (i, item) in items.iter().enumerate() {
        if item.weights.len() != d {
            return Err(OptimizeError::InvalidInput(format!(
                "Item {i} has {} weight dimensions but capacities has {d}",
                item.weights.len()
            )));
        }
    }

    // Score: value / L2-norm of weight vector
    let mut indexed: Vec<(usize, f64)> = items
        .iter()
        .enumerate()
        .map(|(i, it)| {
            let norm_sq: f64 = it.weights.iter().map(|&w| (w as f64).powi(2)).sum();
            let norm = norm_sq.sqrt().max(1e-12);
            (i, it.value as f64 / norm)
        })
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut remaining = capacities.to_vec();
    let mut selected = vec![false; n];
    let mut total = 0u64;

    for (idx, _) in &indexed {
        let item = &items[*idx];
        if item
            .weights
            .iter()
            .enumerate()
            .all(|(dim, &w)| w <= remaining[dim])
        {
            selected[*idx] = true;
            total += item.value;
            for (dim, &w) in item.weights.iter().enumerate() {
                remaining[dim] -= w;
            }
        }
    }

    // 1-swap local improvement: try swapping out one selected item for one not selected
    let mut improved = true;
    while improved {
        improved = false;
        for out_idx in 0..n {
            if !selected[out_idx] {
                continue;
            }
            for in_idx in 0..n {
                if selected[in_idx] {
                    continue;
                }
                // Check if swapping out→in is feasible and improving
                let delta_v = items[in_idx].value as i64 - items[out_idx].value as i64;
                if delta_v <= 0 {
                    continue;
                }
                let feasible = items[in_idx].weights.iter().enumerate().all(|(dim, &w)| {
                    let freed = items[out_idx].weights[dim];
                    freed + remaining[dim] >= w
                });
                if feasible {
                    for dim in 0..d {
                        remaining[dim] += items[out_idx].weights[dim];
                        remaining[dim] -= items[in_idx].weights[dim];
                    }
                    total = total
                        .saturating_sub(items[out_idx].value)
                        .saturating_add(items[in_idx].value);
                    selected[out_idx] = false;
                    selected[in_idx] = true;
                    improved = true;
                    break;
                }
            }
            if improved {
                break;
            }
        }
    }

    let result: Vec<usize> = (0..n).filter(|&i| selected[i]).collect();
    Ok((total, result))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn classic_items() -> Vec<KnapsackItem> {
        // Classic 4-item problem, capacity 5 → optimal value 8 (items 1 and 3)
        vec![
            KnapsackItem {
                weight: 2,
                value: 3,
            },
            KnapsackItem {
                weight: 3,
                value: 4,
            },
            KnapsackItem {
                weight: 2,
                value: 5,
            },
            KnapsackItem {
                weight: 3,
                value: 6,
            },
        ]
    }

    #[test]
    fn test_dp_classic() {
        let items = classic_items();
        let (val, sel) = knapsack_dp(&items, 5).expect("unexpected None or Err");
        // Optimal: item2(w=2,v=5) + item3(w=3,v=6) = weight 5, value 11
        assert_eq!(val, 11, "expected value 11, got {val}");
        let total_weight: u64 = sel.iter().map(|&i| items[i].weight).sum();
        assert!(total_weight <= 5);
        let total_val: u64 = sel.iter().map(|&i| items[i].value).sum();
        assert_eq!(total_val, val);
    }

    #[test]
    fn test_dp_empty() {
        let (val, sel) = knapsack_dp(&[], 10).expect("unexpected None or Err");
        assert_eq!(val, 0);
        assert!(sel.is_empty());
    }

    #[test]
    fn test_dp_zero_capacity() {
        let items = classic_items();
        let (val, sel) = knapsack_dp(&items, 0).expect("unexpected None or Err");
        assert_eq!(val, 0);
        assert!(sel.is_empty());
    }

    #[test]
    fn test_fractional_knapsack() {
        let items = classic_items();
        let val = fractional_knapsack(&items, 5);
        // Fractional must be ≥ integer optimum
        assert!(val >= 9.0 - 1e-9);
    }

    #[test]
    fn test_greedy_knapsack() {
        let items = classic_items();
        let (val, sel) = knapsack_greedy(&items, 5);
        assert!(val > 0);
        let total_weight: u64 = sel.iter().map(|&i| items[i].weight).sum();
        assert!(total_weight <= 5);
    }

    #[test]
    fn test_branch_bound_classic() {
        let items = classic_items();
        let (val, sel) = knapsack_branch_bound(&items, 5).expect("unexpected None or Err");
        // Optimal: item2(w=2,v=5) + item3(w=3,v=6) = weight 5, value 11
        assert_eq!(val, 11);
        let total_weight: u64 = sel.iter().map(|&i| items[i].weight).sum();
        assert!(total_weight <= 5);
    }

    #[test]
    fn test_bb_equals_dp() {
        let items = vec![
            KnapsackItem {
                weight: 1,
                value: 6,
            },
            KnapsackItem {
                weight: 2,
                value: 10,
            },
            KnapsackItem {
                weight: 3,
                value: 12,
            },
        ];
        let cap = 5;
        let (dp_val, _) = knapsack_dp(&items, cap).expect("unexpected None or Err");
        let (bb_val, _) = knapsack_branch_bound(&items, cap).expect("unexpected None or Err");
        assert_eq!(dp_val, bb_val, "DP and B&B should agree");
    }

    #[test]
    fn test_multi_knapsack() {
        let items = vec![
            MultiKnapsackItem {
                weights: vec![2, 1],
                value: 5,
            },
            MultiKnapsackItem {
                weights: vec![1, 2],
                value: 5,
            },
            MultiKnapsackItem {
                weights: vec![3, 3],
                value: 8,
            },
        ];
        let caps = vec![4, 4];
        let (val, sel) = multi_knapsack_greedy(&items, &caps).expect("unexpected None or Err");
        assert!(val > 0);
        // Verify feasibility
        for dim in 0..2 {
            let used: u64 = sel.iter().map(|&i| items[i].weights[dim]).sum();
            assert!(used <= caps[dim]);
        }
    }

    #[test]
    fn test_fractional_zero_capacity() {
        let items = classic_items();
        assert_eq!(fractional_knapsack(&items, 0), 0.0);
    }

    #[test]
    fn test_all_items_fit() {
        let items = classic_items();
        let total_weight: u64 = items.iter().map(|i| i.weight).sum();
        let total_value: u64 = items.iter().map(|i| i.value).sum();
        let (val, sel) = knapsack_dp(&items, total_weight + 100).expect("unexpected None or Err");
        assert_eq!(val, total_value);
        assert_eq!(sel.len(), items.len());
    }
}
