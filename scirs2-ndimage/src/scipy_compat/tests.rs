use super::*;
use scirs2_core::ndarray::array;

#[test]
fn test_scipy_compat_gaussian() {
    let input = array![[1.0, 2.0], [3.0, 4.0]];
    let result =
        gaussian_filter(&input, vec![1.0], None, None, None, None).expect("Operation failed");
    assert_eq!(result.shape(), input.shape());
}

#[test]
fn test_scipy_compat_gaussian_anisotropic_vs_isotropic() {
    // Impulse at the center of an 11x11 image.
    let mut input = scirs2_core::ndarray::Array2::<f64>::zeros((11, 11));
    input[[5, 5]] = 1.0;

    // Isotropic blur: the response must be symmetric under swapping the
    // row/column offset from the impulse.
    let isotropic =
        gaussian_filter(&input, vec![3.0, 3.0], None, None, None, None).expect("Operation failed");
    assert!((isotropic[[8, 5]] - isotropic[[5, 8]]).abs() < 1e-9);

    // Anisotropic blur: heavy spread along axis 0 (rows, sigma=3.0),
    // essentially none along axis 1 (columns, sigma=0.3). The pre-fix
    // implementation silently discarded every sigma value but the
    // first, so this call would have produced the *same* (symmetric)
    // result as the isotropic case above instead of a genuinely
    // anisotropic one.
    let anisotropic =
        gaussian_filter(&input, vec![3.0, 0.3], None, None, None, None).expect("Operation failed");

    // Far along the heavily-blurred axis there is still meaningful
    // intensity; far along the barely-blurred axis there is none.
    assert!(anisotropic[[8, 5]] > 1e-3, "{anisotropic:?}");
    assert!(anisotropic[[5, 8]] < 1e-6, "{anisotropic:?}");
    assert!((anisotropic[[8, 5]] - anisotropic[[5, 8]]).abs() > 1e-3);
    assert!((anisotropic[[5, 8]] - isotropic[[5, 8]]).abs() > 1e-6);
}

#[test]
fn test_scipy_compat_gaussian_order_derivative() {
    let mut input = scirs2_core::ndarray::Array1::<f64>::zeros(11);
    input[5] = 1.0;

    let result =
        gaussian_filter(&input, vec![1.5], Some(1), None, None, None).expect("Operation failed");

    // A first-derivative-of-Gaussian response to an impulse is an odd
    // (antisymmetric) function about the impulse location, and is
    // exactly zero at the impulse itself. The pre-fix implementation
    // always ignored `order` (never even referencing it), returning a
    // plain order-0 Gaussian response instead -- symmetric, and peaked
    // (nonzero) at the center.
    assert!((result[6] + result[4]).abs() < 1e-9, "{result:?}");
    assert!(result[6].abs() > 1e-3, "{result:?}");
    assert_eq!(result[5], 0.0);
}

#[test]
fn test_scipy_compat_gaussian_rejects_bad_sigma_length() {
    let input = array![[1.0, 2.0], [3.0, 4.0]];
    let result = gaussian_filter(&input, vec![1.0, 1.0, 1.0], None, None, None, None);
    assert!(result.is_err());
}

#[test]
fn test_scipy_compat_modes() {
    assert!(matches!(Mode::from_str("reflect"), Ok(Mode::Reflect)));
    assert!(matches!(Mode::from_str("constant"), Ok(Mode::Constant)));
    assert!(matches!(Mode::from_str("nearest"), Ok(Mode::Nearest)));
    assert!(matches!(Mode::from_str("edge"), Ok(Mode::Nearest)));
    assert!(Mode::from_str("invalid").is_err());
}

#[test]
fn test_scipy_compat_binary_erosion() {
    let input = array![[true, false], [false, true]];
    let result = binary_erosion(
        &input,
        None::<&scirs2_core::ndarray::Array2<bool>>,
        None::<usize>,
        None::<&scirs2_core::ndarray::Array2<bool>>,
        None::<bool>,
    )
    .expect("Test: operation failed");
    assert_eq!(result.shape(), input.shape());
}

#[test]
fn test_scipy_compat_zoom() {
    let input = array![[1.0, 2.0], [3.0, 4.0]];
    let result = zoom(&input, vec![2.0, 2.0], None, None, None, None).expect("Operation failed");
    assert_eq!(result.shape(), &[4, 4]);
}

#[test]
fn test_scipy_compat_rotate() {
    let input = array![[1.0, 2.0], [3.0, 4.0]];
    let result =
        rotate(&input.view(), 45.0, None, None, None, None, None).expect("Operation failed");
    assert_eq!(result.ndim(), 2);
}

#[test]
fn test_scipy_compat_shift() {
    let input = array![[1.0, 2.0], [3.0, 4.0]];
    let result = shift(&input, vec![0.5, 0.5], None, None, None, None).expect("Operation failed");
    assert_eq!(result.shape(), input.shape());
}

#[test]
fn test_scipy_compat_laplace() {
    let input = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let result = laplace(&input, None, None).expect("Operation failed");
    assert_eq!(result.shape(), input.shape());
}

#[test]
fn test_scipy_compat_maximum_filter() {
    let input = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let result = maximum_filter(
        &input,
        Some(vec![3, 3]),
        None::<&scirs2_core::ndarray::Array2<bool>>,
        None,
        None,
        None,
    )
    .expect("Test: operation failed");
    assert_eq!(result.shape(), input.shape());
}

#[test]
fn test_scipy_compat_generic_filter() {
    let input = array![[1.0f64, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let mean_func = |values: &[f64]| -> f64 { values.iter().sum::<f64>() / values.len() as f64 };
    let result = generic_filter(
        &input,
        mean_func,
        Some(vec![3, 3]),
        None::<&scirs2_core::ndarray::Array2<bool>>,
        None,
        None,
        None,
    )
    .expect("Test: operation failed");
    assert_eq!(result.shape(), input.shape());
}

// ─── center_of_mass tests ──────────────────────────────────────────────

#[test]
fn test_center_of_mass_no_labels() {
    // 2×2 array with mass concentrated at bottom-right
    let input = array![[0.0_f64, 0.0], [0.0, 1.0]];
    let com = center_of_mass(&input, None::<&scirs2_core::ndarray::Array2<i32>>, None)
        .expect("center_of_mass failed");
    assert_eq!(com.len(), 1);
    // Only the bottom-right element has mass → COM = (1, 1)
    assert!((com[0][0] - 1.0).abs() < 1e-12);
    assert!((com[0][1] - 1.0).abs() < 1e-12);
}

#[test]
fn test_center_of_mass_with_labels() {
    // 2×4 input: two regions separated by label
    // Label 1 → left half, Label 2 → right half
    let input = array![[1.0_f64, 1.0, 2.0, 2.0]];
    let labels = array![[1_i32, 1, 2, 2]];
    let com = center_of_mass(&input, Some(&labels), None).expect("center_of_mass failed");
    // Two labels: 1 and 2
    assert_eq!(com.len(), 2);
    // Label 1: mass at cols 0,1 → COM col = (0*1 + 1*1)/(1+1) = 0.5
    assert!((com[0][1] - 0.5).abs() < 1e-12);
    // Label 2: mass at cols 2,3 → COM col = (2*2 + 3*2)/(2+2) = 2.5
    assert!((com[1][1] - 2.5).abs() < 1e-12);
}

#[test]
fn test_center_of_mass_with_labels_and_index() {
    // Same as above but request only label 2
    let input = array![[1.0_f64, 1.0, 2.0, 2.0]];
    let labels = array![[1_i32, 1, 2, 2]];
    let com = center_of_mass(&input, Some(&labels), Some(vec![2])).expect("center_of_mass failed");
    assert_eq!(com.len(), 1);
    // Label 2: mass at cols 2,3 → COM col = 2.5
    assert!((com[0][1] - 2.5).abs() < 1e-12);
}

// ─── map_coordinates tests ────────────────────────────────────────────

#[test]
fn test_map_coordinates_exact_grid_2d() {
    // Sample at exact integer grid points — should return the exact values
    let input = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],];

    // Coordinates to sample: (0,0), (1,1), (2,2) → values 1, 5, 9
    let coords =
        scirs2_core::ndarray::Array2::from_shape_vec((2, 3), vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
            .expect("coords");

    let result = map_coordinates(&input, &coords, Some(1), Some("constant"), None, None)
        .expect("map_coordinates should succeed");

    assert_eq!(result.len(), 3);
    assert!(
        (result[0] - 1.0).abs() < 1e-10,
        "Expected 1.0, got {}",
        result[0]
    );
    assert!(
        (result[1] - 5.0).abs() < 1e-10,
        "Expected 5.0, got {}",
        result[1]
    );
    assert!(
        (result[2] - 9.0).abs() < 1e-10,
        "Expected 9.0, got {}",
        result[2]
    );
}

#[test]
fn test_map_coordinates_oob_constant_mode() {
    // Out-of-bounds coordinates should return cval (0.0 by default) in constant mode
    let input = array![[1.0_f64, 2.0], [3.0, 4.0]];

    // Coordinate (-1, 0) is out of bounds
    let coords =
        scirs2_core::ndarray::Array2::from_shape_vec((2, 1), vec![-1.0_f64, 0.0]).expect("coords");

    let result = map_coordinates(&input, &coords, Some(1), Some("constant"), Some(0.0), None)
        .expect("map_coordinates OOB should succeed");

    assert_eq!(result.len(), 1);
    assert!(
        (result[0] - 0.0).abs() < 1e-10,
        "OOB should return cval=0.0, got {}",
        result[0]
    );
}

#[test]
fn test_map_coordinates_nearest_mode() {
    // Nearest mode: clamp to border
    let input = array![[10.0_f64, 20.0], [30.0, 40.0]];

    // Sample at (-0.5, -0.5) → nearest is (0,0) = 10.0
    let coords =
        scirs2_core::ndarray::Array2::from_shape_vec((2, 1), vec![-0.5_f64, -0.5]).expect("coords");

    let result = map_coordinates(&input, &coords, Some(1), Some("nearest"), None, None)
        .expect("map_coordinates nearest should succeed");

    assert_eq!(result.len(), 1);
    assert!(
        (result[0] - 10.0).abs() < 1e-9,
        "Nearest clamped should be 10.0, got {}",
        result[0]
    );
}

#[test]
fn test_map_coordinates_interpolation_2d() {
    // Linear interpolation at the centre of a 2×2 constant array should return the constant
    let input = array![[5.0_f64, 5.0], [5.0, 5.0]];

    // Centre of the 2×2 = (0.5, 0.5)
    let coords =
        scirs2_core::ndarray::Array2::from_shape_vec((2, 1), vec![0.5_f64, 0.5]).expect("coords");

    let result = map_coordinates(&input, &coords, Some(1), Some("reflect"), None, None)
        .expect("map_coordinates interpolation should succeed");

    assert_eq!(result.len(), 1);
    assert!(
        (result[0] - 5.0).abs() < 1e-10,
        "Interpolation of constant should be constant, got {}",
        result[0]
    );
}

// ─── grey_erosion tests ───────────────────────────────────────────────

#[test]
fn test_grey_erosion_all_ones_with_footprint() {
    // Erosion of a 5×5 all-ones image with a 3×3 all-true footprint → still all ones.
    let input = Array2::<f64>::ones((5, 5));
    let footprint = Array2::<bool>::from_elem((3, 3), true);
    let result = grey_erosion(&input, None, Some(&footprint), None, None::<f64>)
        .expect("grey_erosion should succeed");
    assert_eq!(result.dim(), (5, 5));
    for &v in result.iter() {
        assert!(
            (v - 1.0).abs() < 1e-12,
            "All-ones erosion should stay 1.0, got {v}"
        );
    }
}

#[test]
fn test_grey_erosion_default_size_no_footprint() {
    // Erosion using size=None (defaults to 3×3 all-true SE) on a 5×5 all-ones image.
    let input = Array2::<f64>::ones((5, 5));
    let result = grey_erosion(
        &input,
        None,                                        // size → defaults to 3×3
        None::<&scirs2_core::ndarray::Array2<bool>>, // no footprint
        None,
        None::<f64>,
    )
    .expect("grey_erosion without footprint should succeed");
    assert_eq!(result.dim(), (5, 5));
    for &v in result.iter() {
        assert!(
            (v - 1.0).abs() < 1e-12,
            "All-ones erosion (default SE) should stay 1.0, got {v}"
        );
    }
}

#[test]
fn test_grey_erosion_shrinks_island() {
    // A 5×5 image: zeros on border, ones inside the 3×3 centre.
    // After 3×3 erosion the border zeros will invade the centre → fewer ones.
    #[rustfmt::skip]
    let data = vec![
        0.0_f64, 0.0, 0.0, 0.0, 0.0,
        0.0,     1.0, 1.0, 1.0, 0.0,
        0.0,     1.0, 1.0, 1.0, 0.0,
        0.0,     1.0, 1.0, 1.0, 0.0,
        0.0,     0.0, 0.0, 0.0, 0.0,
    ];
    let input = Array2::from_shape_vec((5, 5), data).expect("shape");
    let footprint = Array2::<bool>::from_elem((3, 3), true);
    let result = grey_erosion(&input, None, Some(&footprint), None, None::<f64>)
        .expect("grey_erosion should succeed");
    // The centre should still be 1 (it survives), but border/near-border pixels → 0
    assert_eq!(result[[2, 2]], 1.0, "Centre pixel should survive erosion");
    assert_eq!(result[[0, 0]], 0.0, "Corner stays 0 after erosion");
    assert_eq!(
        result[[1, 1]],
        0.0,
        "Near-border pixel becomes 0 after erosion"
    );
}

// ─── distance_transform_edt tests ────────────────────────────────────

#[test]
fn test_edt_single_foreground_pixel() {
    // 5×5 image: only the centre (2,2) is foreground (non-zero).
    // All other pixels have distance = Euclidean distance to (2,2).
    let mut data = vec![0.0_f64; 25];
    data[2 * 5 + 2] = 1.0; // centre pixel
    let input = Array2::from_shape_vec((5, 5), data).expect("shape");
    let (dist_opt, _) =
        distance_transform_edt(&input, None, Some(true), None).expect("EDT should succeed");
    let dist = dist_opt.expect("distances should be Some");

    // Centre pixel itself is foreground → distance = distance to nearest background = 1
    // (nearest background is immediately adjacent)
    // All background pixels → distance 0
    for i in 0..5_usize {
        for j in 0..5_usize {
            if i == 2 && j == 2 {
                // foreground: nearest bg is at distance 1 (up/down/left/right)
                assert!(
                    (dist[[i, j]] - 1.0).abs() < 1e-10,
                    "Centre pixel distance should be 1.0, got {}",
                    dist[[i, j]]
                );
            } else {
                // background: distance = 0
                assert!(
                    dist[[i, j]].abs() < 1e-10,
                    "Background pixel [{i},{j}] should have distance 0, got {}",
                    dist[[i, j]]
                );
            }
        }
    }
}

#[test]
fn test_edt_all_background() {
    // All-zero image → all distances = 0 (no foreground pixels)
    let input = Array2::<f64>::zeros((5, 5));
    let (dist_opt, _) =
        distance_transform_edt(&input, None, Some(true), None).expect("EDT should succeed");
    let dist = dist_opt.expect("distances should be Some");
    for &v in dist.iter() {
        assert!(v.abs() < 1e-12, "All-background → all zeros, got {v}");
    }
}

#[test]
fn test_edt_column_of_background() {
    // 3×3 image with a vertical column of zeros (col 1) and ones elsewhere.
    // The horizontal distance from col 0/2 to col 1 is 1, so EDT[[i,0]] = EDT[[i,2]] = 1.
    // Col 1 is bg → dist = 0.
    #[rustfmt::skip]
    let data = vec![
        1.0_f64, 0.0, 1.0,
        1.0,     0.0, 1.0,
        1.0,     0.0, 1.0,
    ];
    let input = Array2::from_shape_vec((3, 3), data).expect("shape");
    let (dist_opt, _) =
        distance_transform_edt(&input, None, Some(true), None).expect("EDT should succeed");
    let dist = dist_opt.expect("distances");
    for i in 0..3 {
        assert!((dist[[i, 1]]).abs() < 1e-12, "bg col has dist 0");
        assert!(
            (dist[[i, 0]] - 1.0).abs() < 1e-10,
            "col 0 should have dist 1, got {}",
            dist[[i, 0]]
        );
        assert!(
            (dist[[i, 2]] - 1.0).abs() < 1e-10,
            "col 2 should have dist 1, got {}",
            dist[[i, 2]]
        );
    }
}
