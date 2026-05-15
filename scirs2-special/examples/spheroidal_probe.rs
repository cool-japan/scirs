// Debug probe for the Wave 74 spheroidal Flammer / Bouwkamp pipeline.
//
// Verifies:
// 1. `flammer_eigenvalue` matches SciPy `pro_cv` / `obl_cv` to ≥ 1e-7 across
//    `|c| ∈ [1, 30]`.
// 2. `d_coefficients` produces sensible decay patterns at multiple `c`.
// 3. Angular and radial functions match SciPy reference values.
//
// Useful for development iteration; not a regression test.
//
// Reference SciPy values (computed via `scipy.special.{pro_cv, obl_cv,
// pro_ang1, obl_ang1, pro_rad1, pro_rad2, obl_rad1, obl_rad2}` on Python
// 3.14 / SciPy 1.13):
//
//   pro_cv(0, 1, 1.0)  =   2.5930845800
//   pro_cv(0, 2, 2.0)  =   8.2257130011
//   pro_cv(1, 2, 5.0)  =  14.6429562449
//   pro_cv(0, 2, 10.0) =  45.8689526502
//   pro_cv(0, 2, 30.0) = 146.1483563533
//   obl_cv(0, 1, 1.0)  =   1.3932063104
//   obl_cv(0, 2, 2.0)  =   4.0915091022
//   obl_cv(1, 2, 5.0)  =  -7.1278375188
//   obl_cv(0, 2, 10.0) = -45.4896804974
//   obl_cv(0, 2, 30.0) = -725.1345265135
use scirs2_special::{
    angular_function, d_coefficients, flammer_eigenvalue, obl_ang1, obl_cv, obl_rad1, obl_rad2,
    pro_ang1, pro_cv, pro_rad1, pro_rad2, radial_function, spheroidal_eigenvalue_mn,
    SphericalBesselKind, SpheroidalKind, SpheroidalParity,
};

fn main() {
    println!("=== prolate λ comparison vs SciPy ===");
    let pro_refs: &[(i32, i32, f64, f64)] = &[
        (0, 1, 1.0, 2.5930845800),
        (0, 2, 2.0, 8.2257130011),
        (1, 2, 5.0, 14.6429562449),
        (0, 2, 10.0, 45.8689526502),
        (0, 2, 30.0, 146.1483563533),
    ];
    for &(m, n, c, scipy) in pro_refs {
        let l_pro_cv = pro_cv(m, n, c).unwrap_or(f64::NAN);
        let l_swf = spheroidal_eigenvalue_mn(m as usize, n as usize, c, SpheroidalKind::Prolate)
            .map(|e| e.lambda)
            .unwrap_or(f64::NAN);
        let l_flam = flammer_eigenvalue(SpheroidalParity::Prolate, m, n, c, 80).unwrap_or(f64::NAN);
        println!("m={m} n={n} c={c:>5.1}: pro_cv={l_pro_cv:>14.6} swf={l_swf:>14.6} flam={l_flam:>14.6} scipy={scipy:>14.6}");
    }

    println!("\n=== oblate λ comparison vs SciPy ===");
    let obl_refs: &[(i32, i32, f64, f64)] = &[
        (0, 1, 1.0, 1.3932063104),
        (0, 2, 2.0, 4.0915091022),
        (1, 2, 5.0, -7.1278375188),
        (0, 2, 10.0, -45.4896804974),
        (0, 2, 30.0, -725.1345265135),
    ];
    for &(m, n, c, scipy) in obl_refs {
        let l_obl_cv = obl_cv(m, n, c).unwrap_or(f64::NAN);
        let l_swf = spheroidal_eigenvalue_mn(m as usize, n as usize, c, SpheroidalKind::Oblate)
            .map(|e| e.lambda)
            .unwrap_or(f64::NAN);
        let l_flam = flammer_eigenvalue(SpheroidalParity::Oblate, m, n, c, 80).unwrap_or(f64::NAN);
        println!("m={m} n={n} c={c:>5.1}: obl_cv={l_obl_cv:>14.6} swf={l_swf:>14.6} flam={l_flam:>14.6} scipy={scipy:>14.6}");
    }

    println!("\n=== d-coefficients (m=0, n=2, c=5.0, prolate) ===");
    let lam = flammer_eigenvalue(SpheroidalParity::Prolate, 0, 2, 5.0, 80)
        .expect("flammer_eigenvalue prolate");
    let d =
        d_coefficients(SpheroidalParity::Prolate, 0, 2, 5.0, lam).expect("d_coefficients prolate");
    for (k, dk) in d.iter().enumerate().take(8) {
        println!("d[{k}] = {dk:>14.6e}");
    }

    println!("\n=== d-coefficients (m=1, n=2, c=5.0, prolate odd parity) ===");
    let lam = flammer_eigenvalue(SpheroidalParity::Prolate, 1, 2, 5.0, 80)
        .expect("flammer_eigenvalue prolate odd");
    println!("λ = {lam}");
    let d = d_coefficients(SpheroidalParity::Prolate, 1, 2, 5.0, lam)
        .expect("d_coefficients prolate odd");
    for (k, dk) in d.iter().enumerate().take(8) {
        println!("d[{k}] = {dk:>14.6e}");
    }

    println!("\n=== d-coefficients (m=0, n=2, c=5.0, OBLATE) ===");
    let lam = flammer_eigenvalue(SpheroidalParity::Oblate, 0, 2, 5.0, 80)
        .expect("flammer_eigenvalue oblate");
    println!("λ = {lam}");
    let d =
        d_coefficients(SpheroidalParity::Oblate, 0, 2, 5.0, lam).expect("d_coefficients oblate");
    for (k, dk) in d.iter().enumerate().take(8) {
        println!("d[{k}] = {dk:>14.6e}");
    }

    println!("\n=== angular_function (prolate) vs SciPy pro_ang1 ===");
    let cases: &[(i32, i32, f64, f64, f64, f64)] = &[
        (0, 1, 1.0, 0.5, 0.4877531776, 0.9269534809),
        (0, 2, 5.0, 0.5, 0.3843865522, 2.2538796140),
        (1, 2, 5.0, 0.5, 0.8957259676, -0.1823510692),
        (1, 2, 1.0, 0.5, 1.2762087065, 1.6109240931),
        (0, 1, 30.0, 0.7, 0.0001977152, -0.0052227766),
    ];
    for &(m, n, c, x, scipy_v, scipy_p) in cases {
        match angular_function(SpheroidalParity::Prolate, m, n, c, x) {
            Ok((v, p)) => {
                let dv = (v - scipy_v).abs();
                let dp = (p - scipy_p).abs();
                println!("m={m} n={n} c={c:>4.1} x={x}: S=({v:>10.6}, {p:>10.6}), scipy=({scipy_v:>10.6}, {scipy_p:>10.6}), |dv|={dv:.2e} |dp|={dp:.2e}");
            }
            Err(e) => println!("m={m} n={n} c={c:>4.1} x={x}: ERROR {e}"),
        }
    }

    println!("\n=== angular_function (oblate) vs SciPy obl_ang1 ===");
    let obl_cases: &[(i32, i32, f64, f64, f64, f64)] = &[
        (0, 1, 1.0, 0.5, 0.5127556416, 1.0769923985),
        (0, 2, 5.0, 0.5, -0.5976813382, -0.0916328694),
        (1, 2, 5.0, 0.5, 2.1643722013, 7.0314608606),
        (0, 2, 1.0, 0.5, -0.1507602618, 1.4223950801),
    ];
    for &(m, n, c, x, scipy_v, scipy_p) in obl_cases {
        match angular_function(SpheroidalParity::Oblate, m, n, c, x) {
            Ok((v, p)) => {
                let dv = (v - scipy_v).abs();
                let dp = (p - scipy_p).abs();
                println!("m={m} n={n} c={c:>4.1} x={x}: S=({v:>10.6}, {p:>10.6}), scipy=({scipy_v:>10.6}, {scipy_p:>10.6}), |dv|={dv:.2e} |dp|={dp:.2e}");
            }
            Err(e) => println!("m={m} n={n} c={c:>4.1} x={x}: ERROR {e}"),
        }
    }

    println!("\n=== Test cases used in integration tests ===");

    println!("\n--- pro_rad1(0, n, 5.0, 1.5) ---");
    for (n, expected) in [
        (1_i32, -0.1589056154_f64),
        (2, -0.1261614553),
        (3, 0.0438808791),
    ] {
        let (v, _) = pro_rad1(0, n, 5.0, 1.5).expect("pro_rad1");
        let dv = (v - expected).abs();
        println!("  n={n}: ours={v:.10}, scipy={expected:.10}, |dv|={dv:.2e}");
    }

    println!("\n--- pro_rad2(0, n, 5.0, 1.5) ---");
    for (n, expected) in [
        (1_i32, -0.0434742774_f64),
        (2, 0.1171089567),
        (3, 0.1746589232),
    ] {
        let (v, _) = pro_rad2(0, n, 5.0, 1.5).expect("pro_rad2");
        let dv = (v - expected).abs();
        println!("  n={n}: ours={v:.10}, scipy={expected:.10}, |dv|={dv:.2e}");
    }

    println!("\n--- obl_rad1(0, n, 5.0, 2.0) ---");
    for (n, expected) in [
        (1_i32, 0.0496719809_f64),
        (2, 0.0937807668),
        (3, 0.0174573122),
    ] {
        let (v, _) = obl_rad1(0, n, 5.0, 2.0).expect("obl_rad1");
        let dv = (v - expected).abs();
        println!("  n={n}: ours={v:.10}, scipy={expected:.10}, |dv|={dv:.2e}");
    }

    println!("\n--- obl_rad2(0, n, 5.0, 2.0) ---");
    for (n, expected) in [
        (1_i32, 0.0764146049_f64),
        (2, 0.0061816673),
        (3, -0.0929323047),
    ] {
        let (v, _) = obl_rad2(0, n, 5.0, 2.0).expect("obl_rad2");
        let dv = (v - expected).abs();
        println!("  n={n}: ours={v:.10}, scipy={expected:.10}, |dv|={dv:.2e}");
    }

    println!("\n--- pro_ang1(m, n, 5.0, 0.5) ---");
    for (m, n, expected) in [
        (0_i32, 1_i32, 0.3103664330_f64),
        (0, 2, 0.3843865522),
        (1, 2, 0.8957259676),
    ] {
        let (v, _) = pro_ang1(m, n, 5.0, 0.5).expect("pro_ang1");
        let dv = (v - expected).abs();
        println!("  m={m} n={n}: ours={v:.10}, scipy={expected:.10}, |dv|={dv:.2e}");
    }

    println!("\n--- obl_ang1(0, 1, 10.0, 0.5) ---");
    let (v, p) = obl_ang1(0, 1, 10.0, 0.5).expect("obl_ang1 c=10");
    println!("  ours=({v:.6}, {p:.6}), scipy=(5.4249945012, 50.5079881848)");

    println!("\n--- obl_ang1(0, 2, 30.0, 0.3) [degraded precision] ---");
    let (v, p) = obl_ang1(0, 2, 30.0, 0.3).expect("obl_ang1 c=30");
    println!("  ours=({v:.4}, {p:.4}), scipy=(-10832.20, 201562.50)");

    println!("\n--- radial_function direct API smoke test ---");
    let (r1, r1p) = radial_function(
        SpheroidalParity::Prolate,
        SphericalBesselKind::First,
        0,
        2,
        5.0,
        1.5,
    )
    .expect("radial_function first kind");
    println!("  R1(0, 2, 5.0, 1.5) = ({r1:.6}, {r1p:.6}), scipy ≈ (-0.1262, -0.4922)");
}
