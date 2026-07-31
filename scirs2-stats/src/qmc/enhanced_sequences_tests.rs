// Tests for `enhanced_sequences.rs`, split into a separate file (via
// `#[path = ...]`) to keep the main implementation file under the
// workspace's 2000-line guideline; this file's top-level content *is* the
// `tests` module body (same pattern used by `regularized_tests.rs` /
// `robust_tests.rs`).

use super::*;

/// Wave-1 finding: `compute_sobol_advanced` derived its result from
/// `index` alone with no per-dimension direction numbers, so (absent
/// scrambling) every dimension of a point was numerically identical --
/// a degenerate sequence that only ever visits the hyperplane
/// x_0 = x_1 = ... = x_{d-1}, covering none of the actual hypercube
/// volume off that diagonal. This is the assertion that would have
/// FAILED against the old code: it checks that at least one non-trivial
/// point (index > 0; index 0 is trivially all-zeros regardless of the
/// fix) has DIFFERING coordinates across dimensions.
#[test]
fn test_sobol_advanced_dimensions_differ() {
    let sequence_type = EnhancedSequenceType::SobolAdvanced {
        owen_scrambling: false,
        digital_shift: false,
        nested_scrambling: false,
    };
    let mut generator: EnhancedQMCGenerator<f64> =
        EnhancedQMCGenerator::new(sequence_type, 4, EnhancedQMCConfig::default())
            .expect("generator should construct");

    let points = generator.generate(16).expect("generation should succeed");

    let mut saw_differing_point = false;
    for i in 1..points.nrows() {
        let row = points.row(i);
        let first = row[0];
        if row.iter().any(|&v| (v - first).abs() > 1e-12) {
            saw_differing_point = true;
            break;
        }
    }
    assert!(
        saw_differing_point,
        "every dimension was identical for every point (index>0): {points:?}"
    );
}

/// Direct dimension-pair check: the previous implementation made every
/// dimension of a point IDENTICAL, so the Pearson correlation between
/// any two dimensions of the generated sequence was exactly 1.0. A
/// real per-dimension digital-net construction should not exhibit
/// that degenerate perfect correlation.
#[test]
fn test_sobol_advanced_dimensions_not_perfectly_correlated() {
    let sequence_type = EnhancedSequenceType::SobolAdvanced {
        owen_scrambling: false,
        digital_shift: false,
        nested_scrambling: false,
    };
    let mut generator: EnhancedQMCGenerator<f64> =
        EnhancedQMCGenerator::new(sequence_type, 3, EnhancedQMCConfig::default())
            .expect("generator should construct");

    let n = 64;
    let points = generator.generate(n).expect("generation should succeed");

    let col0: Vec<f64> = points.column(0).to_vec();
    let col1: Vec<f64> = points.column(1).to_vec();
    let corr = pearson_correlation(&col0, &col1);

    assert!(
        corr.abs() < 0.5,
        "dimensions 0 and 1 should not be near-perfectly correlated, got r={corr}"
    );
}

/// Coarse low-discrepancy sanity check: partition [0,1)^2 (dimensions
/// 0 and 1) into a 4x4 grid and check that no single cell is wildly
/// over-represented. Under the old bug every point satisfied
/// x_0 == x_1 exactly, so all 64 points would fall into only the 4
/// diagonal cells (up to 16 each) while every off-diagonal cell stayed
/// completely empty -- nowhere close to the ~4-per-cell a
/// space-filling sequence should achieve.
#[test]
fn test_sobol_advanced_low_discrepancy_grid_coverage() {
    let sequence_type = EnhancedSequenceType::SobolAdvanced {
        owen_scrambling: false,
        digital_shift: false,
        nested_scrambling: false,
    };
    let mut generator: EnhancedQMCGenerator<f64> =
        EnhancedQMCGenerator::new(sequence_type, 2, EnhancedQMCConfig::default())
            .expect("generator should construct");

    let n = 64;
    let points = generator.generate(n).expect("generation should succeed");

    let grid = 4usize;
    let mut counts = [[0usize; 4]; 4];
    for row in points.rows() {
        let cx = ((row[0] * grid as f64) as usize).min(grid - 1);
        let cy = ((row[1] * grid as f64) as usize).min(grid - 1);
        counts[cx][cy] += 1;
    }

    let expected_per_cell = n / (grid * grid);
    let max_count = counts.iter().flatten().copied().max().expect("non-empty");
    let empty_cells = counts.iter().flatten().filter(|&&c| c == 0).count();

    assert!(
        max_count <= expected_per_cell * 4 + 2,
        "one grid cell is wildly over-represented (max={max_count}, expected~{expected_per_cell}): {counts:?}"
    );
    // Under the old all-dimensions-identical bug, 12 of the 16 cells
    // (every off-diagonal cell) would be completely empty.
    assert!(
        empty_cells <= 8,
        "too many completely empty grid cells ({empty_cells}/16), sequence looks degenerate: {counts:?}"
    );
}

/// All generated points must land in the unit hypercube [0, 1).
#[test]
fn test_sobol_advanced_points_in_unit_cube() {
    let sequence_type = EnhancedSequenceType::SobolAdvanced {
        owen_scrambling: true,
        digital_shift: true,
        nested_scrambling: false,
    };
    let mut generator: EnhancedQMCGenerator<f64> = EnhancedQMCGenerator::new(
        sequence_type,
        5,
        EnhancedQMCConfig {
            seed: Some(42),
            ..Default::default()
        },
    )
    .expect("generator should construct");

    let points = generator.generate(32).expect("generation should succeed");
    for &v in points.iter() {
        assert!((0.0..1.0).contains(&v), "value {v} out of [0, 1) range");
    }
}

/// The public `enhanced_sobol` convenience function should also produce
/// a genuinely multi-dimensional sequence (dimensions differ), not the
/// degenerate all-dimensions-identical sequence from the old bug.
#[test]
fn test_enhanced_sobol_convenience_fn_dimensions_differ() {
    let points: Array2<f64> =
        enhanced_sobol(16, 4, false, Some(7)).expect("enhanced_sobol should succeed");

    let mut saw_differing_point = false;
    for i in 1..points.nrows() {
        let row = points.row(i);
        let first = row[0];
        if row.iter().any(|&v| (v - first).abs() > 1e-12) {
            saw_differing_point = true;
            break;
        }
    }
    assert!(
        saw_differing_point,
        "enhanced_sobol produced identical values across dimensions: {points:?}"
    );
}

/// Direction numbers must actually vary across dimensions (the direct
/// root cause of the bug: the old code never even looked at
/// per-dimension direction numbers).
#[test]
fn test_sobol_direction_numbers_vary_by_dimension() {
    let sequence_type = EnhancedSequenceType::SobolAdvanced {
        owen_scrambling: false,
        digital_shift: false,
        nested_scrambling: false,
    };
    let generator: EnhancedQMCGenerator<f64> =
        EnhancedQMCGenerator::new(sequence_type, 4, EnhancedQMCConfig::default())
            .expect("generator should construct");

    assert_eq!(generator.sobol_direction_numbers.len(), 4);
    // Dimension 0 is the canonical van-der-Corput direction numbers;
    // every other dimension must differ from it (and, pairwise, from
    // each other) -- otherwise every dimension collapses back to the
    // same 1-D sequence.
    for dim in 1..4 {
        assert_ne!(
            generator.sobol_direction_numbers[dim], generator.sobol_direction_numbers[0],
            "dimension {dim}'s direction numbers must differ from dimension 0's"
        );
        for other in (dim + 1)..4 {
            assert_ne!(
                generator.sobol_direction_numbers[dim], generator.sobol_direction_numbers[other],
                "dimension {dim}'s direction numbers must differ from dimension {other}'s"
            );
        }
    }
}

fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    if var_x <= 0.0 || var_y <= 0.0 {
        return 1.0; // degenerate (zero variance) counts as maximally correlated
    }
    cov / (var_x.sqrt() * var_y.sqrt())
}

// ============================================================================
// `Niederreiter` fix tests.
//
// Wave-1 finding: `compute_niederreiter_enhanced` ignored the (real)
// generating matrices entirely and instead computed a plain per-dimension
// `radical_inverse` in the dimension's own prime base -- i.e. a Halton
// sequence mislabeled as "Niederreiter". Fixed by delegating to the real
// base-2 digital-net construction shared with
// `qmc::advanced::AdvancedQMCGenerator`.
// ============================================================================
mod niederreiter_fix_tests {
    use super::*;

    fn niederreiter_type(matrix_optimization: bool) -> EnhancedSequenceType {
        EnhancedSequenceType::Niederreiter {
            base_strategy: BaseSelectionStrategy::Automatic,
            matrix_optimization,
        }
    }

    /// The real per-dimension generating matrices must actually vary by
    /// dimension (the direct root cause of the bug: the old code never
    /// looked at any generating matrix at all).
    #[test]
    fn test_niederreiter_generating_matrices_vary_by_dimension() {
        let generator: EnhancedQMCGenerator<f64> =
            EnhancedQMCGenerator::new(niederreiter_type(true), 4, EnhancedQMCConfig::default())
                .expect("generator should construct");

        assert_eq!(generator.niederreiter_generating_matrices.len(), 4);
        for dim in 1..4 {
            assert_ne!(
                generator.niederreiter_generating_matrices[dim],
                generator.niederreiter_generating_matrices[0],
                "dimension {dim}'s Niederreiter generating matrix must differ from dimension 0's"
            );
        }
    }

    /// `matrix_optimization` (previously totally ignored, `_matrix_optimization`)
    /// must genuinely affect the generating matrices, not be a no-op.
    #[test]
    fn test_matrix_optimization_flag_changes_generating_matrices() {
        let gen_with: EnhancedQMCGenerator<f64> =
            EnhancedQMCGenerator::new(niederreiter_type(true), 3, EnhancedQMCConfig::default())
                .expect("generator should construct");
        let gen_without: EnhancedQMCGenerator<f64> =
            EnhancedQMCGenerator::new(niederreiter_type(false), 3, EnhancedQMCConfig::default())
                .expect("generator should construct");

        assert_ne!(
            gen_with.niederreiter_generating_matrices, gen_without.niederreiter_generating_matrices,
            "matrix_optimization flag should genuinely affect the generating matrices"
        );
    }

    /// Direct regression test for the historical bug: the old
    /// implementation computed `radical_inverse(index, nth_prime(dim))`
    /// (a plain Halton sequence) regardless of any generating matrix. The
    /// real base-2 digital-net construction must not reproduce that
    /// formula.
    #[test]
    fn test_niederreiter_does_not_match_old_halton_style_bug() {
        fn radical_inverse(index: usize, base: u32) -> f64 {
            let mut result = 0.0;
            let mut fraction = 1.0 / base as f64;
            let mut i = index;
            while i > 0 {
                result += (i % base as usize) as f64 * fraction;
                i /= base as usize;
                fraction /= base as f64;
            }
            result
        }
        // Same first-3 primes table the old buggy `get_prime` used.
        let primes = [2u32, 3, 5];

        let mut generator: EnhancedQMCGenerator<f64> =
            EnhancedQMCGenerator::new(niederreiter_type(true), 3, EnhancedQMCConfig::default())
                .expect("generator should construct");
        let points = generator.generate(20).expect("generation should succeed");

        let mut differs_somewhere = false;
        for index in 1..20 {
            for dim in 0..3 {
                let old_buggy_value = radical_inverse(index, primes[dim]);
                if (points[[index, dim]] - old_buggy_value).abs() > 1e-9 {
                    differs_somewhere = true;
                }
            }
        }
        assert!(
            differs_somewhere,
            "Niederreiter output still matches the old per-dimension-radical-inverse (Halton) bug"
        );
    }

    /// Basic uniformity sanity check (mirrors
    /// `qmc::advanced::tests::test_niederreiter_sequence`): per-dimension
    /// column means should be reasonably close to 0.5 for a genuine
    /// digital-net construction.
    #[test]
    fn test_niederreiter_column_means_near_half() {
        let mut generator: EnhancedQMCGenerator<f64> =
            EnhancedQMCGenerator::new(niederreiter_type(true), 3, EnhancedQMCConfig::default())
                .expect("generator should construct");
        let points = generator.generate(200).expect("generation should succeed");

        for j in 0..3 {
            let mean = points.column(j).iter().sum::<f64>() / points.nrows() as f64;
            assert!(
                (mean - 0.5).abs() < 0.2,
                "column {j} mean {mean} not reasonably close to 0.5"
            );
        }
    }

    /// Coarse low-discrepancy grid-coverage sanity check (same shape as
    /// the Sobol-advanced test above).
    #[test]
    fn test_niederreiter_low_discrepancy_grid_coverage() {
        let mut generator: EnhancedQMCGenerator<f64> =
            EnhancedQMCGenerator::new(niederreiter_type(true), 2, EnhancedQMCConfig::default())
                .expect("generator should construct");

        let n = 64;
        let points = generator.generate(n).expect("generation should succeed");

        let grid = 4usize;
        let mut counts = [[0usize; 4]; 4];
        for row in points.rows() {
            let cx = ((row[0] * grid as f64) as usize).min(grid - 1);
            let cy = ((row[1] * grid as f64) as usize).min(grid - 1);
            counts[cx][cy] += 1;
        }
        let expected_per_cell = n / (grid * grid);
        let max_count = counts.iter().flatten().copied().max().expect("non-empty");
        let empty_cells = counts.iter().flatten().filter(|&&c| c == 0).count();

        assert!(
            max_count <= expected_per_cell * 4 + 2,
            "one grid cell is wildly over-represented (max={max_count}): {counts:?}"
        );
        assert!(
            empty_cells <= 8,
            "too many completely empty grid cells ({empty_cells}/16): {counts:?}"
        );
    }

    #[test]
    fn test_niederreiter_points_in_unit_interval() {
        let mut generator: EnhancedQMCGenerator<f64> =
            EnhancedQMCGenerator::new(niederreiter_type(true), 5, EnhancedQMCConfig::default())
                .expect("generator should construct");
        let points = generator.generate(30).expect("generation should succeed");
        for &v in points.iter() {
            assert!((0.0..1.0).contains(&v), "value {v} out of [0, 1) range");
        }
    }
}

// ============================================================================
// `FaureImproved` fix tests.
//
// Wave-1 finding: `compute_faure_improved` computed a SINGLE radical
// inverse value once and assigned that SAME value to every dimension --
// the same all-dimensions-identical degeneracy as the historical
// `compute_sobol_advanced` bug (see `test_sobol_advanced_dimensions_differ`
// above), just for a different sequence type. Fixed with the real
// permuted-van-der-Corput construction: base = smallest prime >=
// dimension; dimension d's digits are the base-b digit expansion of the
// index transformed by the d-th power of the (mod-base) Pascal matrix.
// ============================================================================
mod faure_fix_tests {
    use super::*;

    fn faure_type() -> EnhancedSequenceType {
        EnhancedSequenceType::FaureImproved {
            permutation_optimization: false,
            radical_inverse_improvements: false,
        }
    }

    /// Base must be the smallest prime >= dimension (Faure 1982's
    /// requirement for the construction's low-discrepancy guarantees).
    #[test]
    fn test_faure_base_is_smallest_prime_geq_dimension() {
        let cases = [
            (1usize, 2u64),
            (2, 2),
            (3, 3),
            (4, 5),
            (5, 5),
            (6, 7),
            (7, 7),
            (10, 11),
        ];
        for (dimension, expected_base) in cases {
            let generator: EnhancedQMCGenerator<f64> =
                EnhancedQMCGenerator::new(faure_type(), dimension, EnhancedQMCConfig::default())
                    .expect("generator should construct");
            assert_eq!(generator.faure_base, expected_base, "dimension={dimension}");
        }
    }

    /// Dimension 0 (Pascal matrix power 0 == identity) must be EXACTLY
    /// the plain base-b van der Corput sequence -- the strongest possible
    /// correctness check on the real construction, independent of any
    /// permutation logic.
    #[test]
    fn test_faure_dimension0_matches_plain_van_der_corput() {
        fn radical_inverse(index: usize, base: u64) -> f64 {
            let mut result = 0.0;
            let mut fraction = 1.0 / base as f64;
            let mut i = index as u64;
            while i > 0 {
                result += (i % base) as f64 * fraction;
                i /= base;
                fraction /= base as f64;
            }
            result
        }

        let dimension = 3; // base = smallest prime >= 3 = 3
        let mut generator: EnhancedQMCGenerator<f64> =
            EnhancedQMCGenerator::new(faure_type(), dimension, EnhancedQMCConfig::default())
                .expect("generator should construct");
        assert_eq!(generator.faure_base, 3);

        let points = generator.generate(20).expect("generation should succeed");
        for index in 0..20 {
            let expected = radical_inverse(index, 3);
            assert!(
                (points[[index, 0]] - expected).abs() < 1e-9,
                "dimension 0 at index {index}: expected plain van der Corput {expected}, got {}",
                points[[index, 0]]
            );
        }
    }

    /// Dimensions > 0 must be genuinely distinct permutations, not the old
    /// "every dimension gets the exact same `radical_inverse(index, base)`
    /// value" bug (the same all-dimensions-identical degeneracy as the
    /// historical Sobol bug, see the module doc comment above).
    #[test]
    fn test_faure_dimensions_differ_and_do_not_match_old_identical_dimension_bug() {
        fn radical_inverse(index: usize, base: u64) -> f64 {
            let mut result = 0.0;
            let mut fraction = 1.0 / base as f64;
            let mut i = index as u64;
            while i > 0 {
                result += (i % base) as f64 * fraction;
                i /= base;
                fraction /= base as f64;
            }
            result
        }

        let dimension = 4; // base = smallest prime >= 4 = 5
        let mut generator: EnhancedQMCGenerator<f64> =
            EnhancedQMCGenerator::new(faure_type(), dimension, EnhancedQMCConfig::default())
                .expect("generator should construct");
        assert_eq!(generator.faure_base, 5);

        let points = generator.generate(30).expect("generation should succeed");

        let mut matched_old_bug_everywhere = true;
        let mut saw_dimension_variety = false;
        for index in 1..30 {
            // The old bug: EVERY dimension was assigned the exact same
            // `radical_inverse(index, base)` value (`base` computed once,
            // with no per-dimension logic at all).
            let old_bug_value = radical_inverse(index, 5);
            let first = points[[index, 0]];
            for dim in 0..dimension {
                if (points[[index, dim]] - old_bug_value).abs() > 1e-9 {
                    matched_old_bug_everywhere = false;
                }
                if dim > 0 && (points[[index, dim]] - first).abs() > 1e-9 {
                    saw_dimension_variety = true;
                }
            }
        }
        assert!(
            !matched_old_bug_everywhere,
            "Faure output still matches the old identical-radical-inverse-per-dimension bug"
        );
        assert!(
            saw_dimension_variety,
            "every dimension was identical (degenerate Faure output)"
        );
    }

    /// Coarse low-discrepancy grid-coverage sanity check.
    #[test]
    fn test_faure_low_discrepancy_grid_coverage() {
        let mut generator: EnhancedQMCGenerator<f64> =
            EnhancedQMCGenerator::new(faure_type(), 2, EnhancedQMCConfig::default())
                .expect("generator should construct");

        let n = 64;
        let points = generator.generate(n).expect("generation should succeed");

        let grid = 4usize;
        let mut counts = [[0usize; 4]; 4];
        for row in points.rows() {
            let cx = ((row[0] * grid as f64) as usize).min(grid - 1);
            let cy = ((row[1] * grid as f64) as usize).min(grid - 1);
            counts[cx][cy] += 1;
        }
        let expected_per_cell = n / (grid * grid);
        let max_count = counts.iter().flatten().copied().max().expect("non-empty");
        let empty_cells = counts.iter().flatten().filter(|&&c| c == 0).count();

        assert!(
            max_count <= expected_per_cell * 4 + 2,
            "one grid cell is wildly over-represented (max={max_count}): {counts:?}"
        );
        assert!(
            empty_cells <= 8,
            "too many completely empty grid cells ({empty_cells}/16): {counts:?}"
        );
    }

    #[test]
    fn test_faure_points_in_unit_interval() {
        let sequence_type = EnhancedSequenceType::FaureImproved {
            permutation_optimization: true,
            radical_inverse_improvements: true,
        };
        let mut generator: EnhancedQMCGenerator<f64> =
            EnhancedQMCGenerator::new(sequence_type, 6, EnhancedQMCConfig::default())
                .expect("generator should construct");
        let points = generator.generate(25).expect("generation should succeed");
        for &v in points.iter() {
            assert!((0.0..1.0).contains(&v), "value {v} out of [0, 1) range");
        }
    }
}

// ============================================================================
// `DigitalNet` fix tests.
//
// Wave-1 finding: `compute_digital_net` ignored `net_params` and
// `construction_method` entirely and always called the Sobol path. Fixed
// to dispatch on `construction_method`: `Sobol` and `NiederreiterXing` now
// reuse the crate's real constructions; `PolynomialLattice`/`FiniteField`
// (which would require polynomial-ring/finite-field arithmetic this crate
// does not implement) now return an honest `NotImplementedError` instead
// of silently substituting Sobol.
// ============================================================================
mod digital_net_fix_tests {
    use super::*;

    fn net_params(base: usize) -> DigitalNetParams {
        DigitalNetParams {
            t: 0,
            m: 32,
            s: 3,
            base,
        }
    }

    #[test]
    fn test_digital_net_sobol_matches_compute_sobol_advanced_directly() {
        let digital_seq = EnhancedSequenceType::DigitalNet {
            net_params: net_params(2),
            construction_method: NetConstructionMethod::Sobol,
        };
        let mut digital_gen: EnhancedQMCGenerator<f64> =
            EnhancedQMCGenerator::new(digital_seq, 3, EnhancedQMCConfig::default())
                .expect("generator should construct");
        let digital_points = digital_gen.generate(10).expect("generation should succeed");

        let sobol_seq = EnhancedSequenceType::SobolAdvanced {
            owen_scrambling: false,
            digital_shift: false,
            nested_scrambling: false,
        };
        let mut sobol_gen: EnhancedQMCGenerator<f64> =
            EnhancedQMCGenerator::new(sobol_seq, 3, EnhancedQMCConfig::default())
                .expect("generator should construct");
        let sobol_points = sobol_gen.generate(10).expect("generation should succeed");

        for i in 0..10 {
            for j in 0..3 {
                assert!(
                    (digital_points[[i, j]] - sobol_points[[i, j]]).abs() < 1e-12,
                    "DigitalNet(Sobol) should match compute_sobol_advanced exactly at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_digital_net_niederreiter_xing_matches_niederreiter_construction() {
        let digital_seq = EnhancedSequenceType::DigitalNet {
            net_params: net_params(2),
            construction_method: NetConstructionMethod::NiederreiterXing,
        };
        let mut digital_gen: EnhancedQMCGenerator<f64> =
            EnhancedQMCGenerator::new(digital_seq, 3, EnhancedQMCConfig::default())
                .expect("generator should construct");
        let digital_points = digital_gen.generate(10).expect("generation should succeed");

        let niederreiter_seq = EnhancedSequenceType::Niederreiter {
            base_strategy: BaseSelectionStrategy::Automatic,
            matrix_optimization: true,
        };
        let mut niederreiter_gen: EnhancedQMCGenerator<f64> =
            EnhancedQMCGenerator::new(niederreiter_seq, 3, EnhancedQMCConfig::default())
                .expect("generator should construct");
        let niederreiter_points = niederreiter_gen
            .generate(10)
            .expect("generation should succeed");

        for i in 0..10 {
            for j in 0..3 {
                assert!(
                    (digital_points[[i, j]] - niederreiter_points[[i, j]]).abs() < 1e-12,
                    "DigitalNet(NiederreiterXing) should match the real Niederreiter \
                     construction at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_digital_net_polynomial_lattice_returns_honest_error_not_silent_sobol_fallback() {
        let sequence_type = EnhancedSequenceType::DigitalNet {
            net_params: net_params(2),
            construction_method: NetConstructionMethod::PolynomialLattice,
        };
        let mut generator: EnhancedQMCGenerator<f64> =
            EnhancedQMCGenerator::new(sequence_type, 3, EnhancedQMCConfig::default())
                .expect("generator should construct");
        let result = generator.generate(10);
        assert!(
            result.is_err(),
            "PolynomialLattice is not implemented and must return an error, not silently \
             fall back to a different construction"
        );
    }

    #[test]
    fn test_digital_net_finite_field_returns_honest_error_not_silent_sobol_fallback() {
        let sequence_type = EnhancedSequenceType::DigitalNet {
            net_params: net_params(2),
            construction_method: NetConstructionMethod::FiniteField,
        };
        let mut generator: EnhancedQMCGenerator<f64> =
            EnhancedQMCGenerator::new(sequence_type, 3, EnhancedQMCConfig::default())
                .expect("generator should construct");
        let result = generator.generate(10);
        assert!(
            result.is_err(),
            "FiniteField is not implemented and must return an error, not silently fall \
             back to a different construction"
        );
    }

    #[test]
    fn test_digital_net_non_base2_returns_honest_error() {
        let sequence_type = EnhancedSequenceType::DigitalNet {
            net_params: net_params(3),
            construction_method: NetConstructionMethod::Sobol,
        };
        let mut generator: EnhancedQMCGenerator<f64> =
            EnhancedQMCGenerator::new(sequence_type, 3, EnhancedQMCConfig::default())
                .expect("generator should construct");
        let result = generator.generate(10);
        assert!(
            result.is_err(),
            "base-3 DigitalNet should be rejected honestly"
        );
    }
}

// ============================================================================
// `Hybrid` fix tests.
//
// Wave-1 finding: `compute_hybrid_sequence` ignored `primary`, `secondary`
// and `combination` entirely and always evaluated the Sobol path. Fixed
// to genuinely combine the two sub-sequences per `combination`.
// ============================================================================
mod hybrid_fix_tests {
    use super::*;

    fn primary_type() -> EnhancedSequenceType {
        EnhancedSequenceType::SobolAdvanced {
            owen_scrambling: false,
            digital_shift: false,
            nested_scrambling: false,
        }
    }

    fn secondary_type() -> EnhancedSequenceType {
        EnhancedSequenceType::Niederreiter {
            base_strategy: BaseSelectionStrategy::Automatic,
            matrix_optimization: true,
        }
    }

    fn generate_all(
        combination: HybridCombinationStrategy,
        n: usize,
        dimension: usize,
    ) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let hybrid_type = EnhancedSequenceType::Hybrid {
            primary: Box::new(primary_type()),
            secondary: Box::new(secondary_type()),
            combination,
        };
        let mut hybrid_gen: EnhancedQMCGenerator<f64> =
            EnhancedQMCGenerator::new(hybrid_type, dimension, EnhancedQMCConfig::default())
                .expect("generator should construct");
        let hybrid_points = hybrid_gen.generate(n).expect("generation should succeed");

        let mut primary_gen: EnhancedQMCGenerator<f64> =
            EnhancedQMCGenerator::new(primary_type(), dimension, EnhancedQMCConfig::default())
                .expect("generator should construct");
        let primary_points = primary_gen.generate(n).expect("generation should succeed");

        let mut secondary_gen: EnhancedQMCGenerator<f64> =
            EnhancedQMCGenerator::new(secondary_type(), dimension, EnhancedQMCConfig::default())
                .expect("generator should construct");
        let secondary_points = secondary_gen
            .generate(n)
            .expect("generation should succeed");

        (hybrid_points, primary_points, secondary_points)
    }

    #[test]
    fn test_hybrid_interleave_alternates_whole_points_by_index_parity() {
        let (hybrid, primary, secondary) =
            generate_all(HybridCombinationStrategy::Interleave, 10, 3);
        for i in 0..10 {
            let expected = if i % 2 == 0 {
                primary.row(i)
            } else {
                secondary.row(i)
            };
            for j in 0..3 {
                assert!(
                    (hybrid[[i, j]] - expected[j]).abs() < 1e-12,
                    "Interleave row {i} col {j}: expected {}, got {}",
                    expected[j],
                    hybrid[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_hybrid_weighted_matches_linear_blend_of_primary_and_secondary() {
        let weight = 0.3;
        let (hybrid, primary, secondary) =
            generate_all(HybridCombinationStrategy::Weighted(weight), 10, 3);
        for i in 0..10 {
            for j in 0..3 {
                let expected = weight * primary[[i, j]] + (1.0 - weight) * secondary[[i, j]];
                assert!(
                    (hybrid[[i, j]] - expected).abs() < 1e-9,
                    "Weighted row {i} col {j}: expected {expected}, got {}",
                    hybrid[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_hybrid_dimension_alternation_alternates_by_dimension_parity() {
        let (hybrid, primary, secondary) =
            generate_all(HybridCombinationStrategy::DimensionAlternation, 10, 3);
        for i in 0..10 {
            for j in 0..3 {
                let expected = if j % 2 == 0 {
                    primary[[i, j]]
                } else {
                    secondary[[i, j]]
                };
                assert!(
                    (hybrid[[i, j]] - expected).abs() < 1e-12,
                    "DimensionAlternation row {i} col {j}: expected {expected}, got {}",
                    hybrid[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_hybrid_adaptive_equals_equal_weight_blend() {
        let (hybrid, primary, secondary) = generate_all(HybridCombinationStrategy::Adaptive, 10, 3);
        for i in 0..10 {
            for j in 0..3 {
                let expected = 0.5 * primary[[i, j]] + 0.5 * secondary[[i, j]];
                assert!(
                    (hybrid[[i, j]] - expected).abs() < 1e-9,
                    "Adaptive row {i} col {j}: expected {expected}, got {}",
                    hybrid[[i, j]]
                );
            }
        }
    }

    /// Direct regression test for the historical bug: the old
    /// implementation always used `compute_sobol_advanced(index, true,
    /// true, false)` regardless of `primary`/`secondary`/`combination`.
    /// Using a `secondary` that is NOT Sobol-shaped (Niederreiter) and
    /// `DimensionAlternation` must therefore differ from a bare Sobol
    /// (owen+shift) sequence.
    #[test]
    fn test_hybrid_does_not_match_old_always_sobol_bug() {
        let (hybrid, _primary, _secondary) =
            generate_all(HybridCombinationStrategy::DimensionAlternation, 10, 3);

        let old_bug_type = EnhancedSequenceType::SobolAdvanced {
            owen_scrambling: true,
            digital_shift: true,
            nested_scrambling: false,
        };
        let mut old_bug_gen: EnhancedQMCGenerator<f64> =
            EnhancedQMCGenerator::new(old_bug_type, 3, EnhancedQMCConfig::default())
                .expect("generator should construct");
        let old_bug_points = old_bug_gen.generate(10).expect("generation should succeed");

        let mut differs_somewhere = false;
        for i in 0..10 {
            for j in 0..3 {
                if (hybrid[[i, j]] - old_bug_points[[i, j]]).abs() > 1e-9 {
                    differs_somewhere = true;
                }
            }
        }
        assert!(
            differs_somewhere,
            "Hybrid output still matches the old always-Sobol(owen=true,shift=true) bug"
        );
    }
}

// ============================================================================
// Quality-metrics fix tests.
//
// Wave-1 finding: `assess_quality` only ever populated `star_discrepancy`;
// `wraparound_discrepancy`, `diaphony` and `figure_of_merit` were left at
// `QualityMetrics::default()`'s `0.0` unconditionally, silently reporting
// "perfect" quality on those three axes regardless of the actual sequence.
// Fixed by computing all four fields for real (see `assess_quality`'s doc
// comment for the exact formulas/truncations).
// ============================================================================
mod quality_metrics_fix_tests {
    use super::*;

    #[test]
    fn test_quality_metrics_no_longer_hardcoded_zero_for_real_sequence() {
        let sequence_type = EnhancedSequenceType::SobolAdvanced {
            owen_scrambling: false,
            digital_shift: false,
            nested_scrambling: false,
        };
        let mut generator: EnhancedQMCGenerator<f64> =
            EnhancedQMCGenerator::new(sequence_type, 3, EnhancedQMCConfig::default())
                .expect("generator should construct");

        // n <= max_assessment_length (default 10000), so `generate`
        // triggers `assess_quality` internally.
        let _points = generator.generate(64).expect("generation should succeed");
        let metrics = generator.quality_metrics();

        // The bug under test: these three were previously hardcoded to
        // 0.0 regardless of the sequence.
        assert!(
            metrics.wraparound_discrepancy > 0.0,
            "wraparound_discrepancy is still hardcoded to 0.0"
        );
        assert!(metrics.diaphony > 0.0, "diaphony is still hardcoded to 0.0");
        assert!(
            metrics.figure_of_merit > 0.0,
            "figure_of_merit is still hardcoded to 0.0"
        );
        assert!(metrics.star_discrepancy >= 0.0);

        // Known sanity bounds: a genuine low-discrepancy sequence in a
        // modest dimension should not have a wildly large discrepancy/
        // diaphony (both are "smaller is better", O(1)-scale quantities).
        assert!(
            metrics.wraparound_discrepancy < 2.0,
            "wraparound_discrepancy implausibly large for a real QMC sequence: {}",
            metrics.wraparound_discrepancy
        );
        assert!(
            metrics.diaphony < 2.0,
            "diaphony implausibly large for a real QMC sequence: {}",
            metrics.diaphony
        );
    }

    #[test]
    fn test_figure_of_merit_equals_max_of_the_other_three_metrics() {
        let sequence_type = EnhancedSequenceType::FaureImproved {
            permutation_optimization: false,
            radical_inverse_improvements: false,
        };
        let mut generator: EnhancedQMCGenerator<f64> =
            EnhancedQMCGenerator::new(sequence_type, 2, EnhancedQMCConfig::default())
                .expect("generator should construct");
        let _points = generator.generate(40).expect("generation should succeed");
        let metrics = generator.quality_metrics();

        let expected_max = metrics
            .star_discrepancy
            .max(metrics.wraparound_discrepancy)
            .max(metrics.diaphony);
        assert!((metrics.figure_of_merit - expected_max).abs() < 1e-12);
    }

    /// "Known discrepancy sanity bounds": a deliberately DEGENERATE point
    /// set (every point on the diagonal x_0 = x_1, matching the historical
    /// Sobol bug's exact failure shape) must score markedly WORSE (larger
    /// wraparound_discrepancy / diaphony / figure_of_merit) than a real,
    /// genuinely-generated low-discrepancy sequence of the same size --
    /// this is the assertion that would have FAILED under the old
    /// hardcoded-0.0-regardless-of-input code (which scored both
    /// identically, at 0.0).
    #[test]
    fn test_quality_metrics_distinguish_degenerate_from_real_sequence() {
        let n = 48;
        let d = 2;

        let sequence_type = EnhancedSequenceType::SobolAdvanced {
            owen_scrambling: false,
            digital_shift: false,
            nested_scrambling: false,
        };
        let mut real_generator: EnhancedQMCGenerator<f64> =
            EnhancedQMCGenerator::new(sequence_type, d, EnhancedQMCConfig::default())
                .expect("generator should construct");
        let _real_points = real_generator
            .generate(n)
            .expect("generation should succeed");
        let real_metrics = real_generator.quality_metrics().clone();

        // Degenerate sequence: every point on the diagonal x_0 == x_1 (the
        // exact failure shape of the historical Sobol bug this module
        // fixed).
        let mut degenerate = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            let v = i as f64 / n as f64;
            degenerate[[i, 0]] = v;
            degenerate[[i, 1]] = v;
        }
        // `assess_quality` is a private method, callable here since this
        // test module is a descendant of `enhanced_sequences`; reuse a
        // freshly constructed generator purely as a vessel for it.
        let mut degenerate_generator: EnhancedQMCGenerator<f64> = EnhancedQMCGenerator::new(
            EnhancedSequenceType::SobolAdvanced {
                owen_scrambling: false,
                digital_shift: false,
                nested_scrambling: false,
            },
            d,
            EnhancedQMCConfig::default(),
        )
        .expect("generator should construct");
        degenerate_generator
            .assess_quality(&degenerate)
            .expect("assess_quality should succeed");
        let degenerate_metrics = degenerate_generator.quality_metrics();

        assert!(
            degenerate_metrics.diaphony > real_metrics.diaphony,
            "degenerate (all-diagonal) sequence should have HIGHER diaphony than a real \
             sequence: degenerate={} real={}",
            degenerate_metrics.diaphony,
            real_metrics.diaphony
        );
        assert!(
            degenerate_metrics.figure_of_merit > real_metrics.figure_of_merit,
            "degenerate sequence should have a worse (larger) figure_of_merit: degenerate={} \
             real={}",
            degenerate_metrics.figure_of_merit,
            real_metrics.figure_of_merit
        );
    }
}
