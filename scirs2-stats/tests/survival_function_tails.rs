//! Deep-tail regression tests for the survival functions (`sf`) and inverse
//! survival functions (`isf`) of the continuous distributions.
//!
//! `ContinuousCDF::sf` used to default to `1 - cdf(x)`, which returns exactly 0
//! as soon as the CDF rounds to 1 (about 8 standard deviations for a normal,
//! a few hundred for heavy tails), and `isf(q) = ppf(1 - q)` rounded `1 - q` to
//! 1 for tiny `q`. Every reference value below was computed independently with
//! mpmath at 50 digits, NOT derived from this crate.

use scirs2_stats::distributions::beta::Beta;
use scirs2_stats::distributions::cauchy::Cauchy;
use scirs2_stats::distributions::chi_square::ChiSquare;
use scirs2_stats::distributions::exponential::Exponential;
use scirs2_stats::distributions::f::F as FDist;
use scirs2_stats::distributions::gamma::Gamma;
use scirs2_stats::distributions::inverse_gaussian::InverseGaussian;
use scirs2_stats::distributions::laplace::Laplace;
use scirs2_stats::distributions::logistic::Logistic;
use scirs2_stats::distributions::lognormal::Lognormal;
use scirs2_stats::distributions::normal::Normal;
use scirs2_stats::distributions::pareto::Pareto;
use scirs2_stats::distributions::student_t::StudentT;
use scirs2_stats::distributions::uniform::Uniform;
use scirs2_stats::distributions::weibull::Weibull;
use scirs2_stats::traits::ContinuousCDF;

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Relative-error check; an absolute epsilon would make every deep-tail
/// comparison vacuous.
fn assert_close(label: &str, got: f64, want: f64, max_relative: f64) {
    let relative = ((got - want) / want).abs();
    assert!(
        relative <= max_relative,
        "{label}: got {got:e}, want {want:e}, relative error {relative:e} > {max_relative:e}"
    );
}

#[test]
fn test_normal_sf_deep_tail() -> TestResult {
    let normal = Normal::new(0.0_f64, 1.0)?;
    let cases: &[(f64, f64)] = &[
        (-5.0, 0.9999997133484281),
        (3.0, 0.0013498980316300946),
        (10.0, 7.619853024160525e-24),
        (20.0, 2.7536241186062337e-89),
        (35.0, 1.1249107064724062e-268),
    ];
    for &(x, want) in cases {
        assert_close("Normal::sf", normal.sf(x), want, 1e-12);
        // Through the trait as well (the path `hazard`/`cumhazard` take).
        assert_close(
            "ContinuousCDF::sf",
            ContinuousCDF::sf(&normal, x),
            want,
            1e-12,
        );
    }
    // The lower tail of the CDF is the same computation mirrored.
    assert_close(
        "Normal::cdf",
        normal.cdf(-10.0),
        7.619853024160525e-24,
        1e-12,
    );
    Ok(())
}

#[test]
fn test_normal_and_student_t_isf_small_q() -> TestResult {
    let normal = Normal::new(0.0_f64, 1.0)?;
    // Acklam's quantile approximation is good to ~1.2e-09 relative.
    assert_close("Normal::isf", normal.isf(1e-20)?, 9.262340089798407, 1e-8);
    let shifted = Normal::new(3.0_f64, 2.0)?;
    assert_close(
        "Normal::isf",
        shifted.isf(1e-20)?,
        3.0 + 2.0 * 9.262340089798407,
        1e-8,
    );

    let t = StudentT::new(5.0_f64, 0.0, 1.0)?;
    let x = ContinuousCDF::isf(&t, 9.480007112311813e-10)?;
    assert_close("StudentT::isf", x, 100.0, 1e-6);
    Ok(())
}

#[test]
fn test_heavy_and_exponential_tails() -> TestResult {
    let cauchy = Cauchy::new(0.0_f64, 1.0)?;
    assert_close("Cauchy::sf", cauchy.sf(1e20), 3.1830988618379067e-21, 1e-12);
    let cauchy_shifted = Cauchy::new(1.0_f64, 2.0)?;
    assert_close(
        "Cauchy::sf",
        cauchy_shifted.sf(1e10),
        6.366197724312433e-11,
        1e-12,
    );
    assert_close("Cauchy::isf", cauchy.isf(1e-12)?, 318309886183.79065, 1e-9);

    let laplace = Laplace::new(0.0_f64, 1.0)?;
    assert_close(
        "Laplace::sf",
        laplace.sf(50.0),
        9.643749239819589e-23,
        1e-12,
    );
    assert_close(
        "Laplace::isf",
        laplace.isf(1e-40)?,
        91.41025653920188,
        1e-12,
    );

    let exponential = Exponential::new(2.0_f64, 0.0)?;
    assert_close(
        "Exponential::sf",
        exponential.sf(30.0),
        8.75651076269652e-27,
        1e-12,
    );
    assert_close(
        "ContinuousCDF::isf",
        ContinuousCDF::isf(&exponential, 1e-30)?,
        34.538776394910684,
        1e-12,
    );

    let weibull = Weibull::new(2.0_f64, 1.0, 0.0)?;
    assert_close("Weibull::sf", weibull.sf(8.0), 1.603810890548638e-28, 1e-12);

    let logistic = Logistic::new(0.0_f64, 1.0)?;
    assert_close(
        "Logistic::sf",
        logistic.sf(60.0),
        8.75651076269652e-27,
        1e-12,
    );

    let lognormal = Lognormal::new(0.0_f64, 1.0, 0.0)?;
    assert_close(
        "Lognormal::sf",
        lognormal.sf(12.0_f64.exp()),
        1.776482112077679e-33,
        1e-10,
    );

    let pareto = Pareto::new(3.0_f64, 1.0, 0.0)?;
    assert_close("Pareto::sf", pareto.sf(1e6), 1e-18, 1e-12);

    let uniform = Uniform::new(0.0_f64, 1.0)?;
    assert_close("Uniform::sf", uniform.sf(1.0 - 1e-12), 1e-12, 1e-3);
    assert_close("Uniform::isf", uniform.isf(0.25)?, 0.75, 1e-15);
    Ok(())
}

#[test]
fn test_regularized_function_tails() -> TestResult {
    let f = FDist::new(5.0_f64, 10.0, 0.0, 1.0)?;
    assert_close("F::sf", f.sf(50.0), 9.402259788055298e-07, 1e-9);
    assert_close("F::sf", f.sf(1e4), 3.749061229200163e-18, 1e-9);

    let t5 = StudentT::new(5.0_f64, 0.0, 1.0)?;
    assert_close("StudentT::sf", t5.sf(100.0), 9.480007112311813e-10, 1e-9);
    let t30 = StudentT::new(30.0_f64, 0.0, 1.0)?;
    assert_close(
        "StudentT::sf",
        ContinuousCDF::sf(&t30, 40.0),
        6.863022597203201e-28,
        1e-9,
    );

    let chi4 = ChiSquare::new(4.0_f64, 0.0, 1.0)?;
    assert_close("ChiSquare::sf", chi4.sf(200.0), 3.757276735781044e-42, 1e-9);
    let chi300 = ChiSquare::new(300.0_f64, 0.0, 1.0)?;
    assert_close(
        "ChiSquare::sf",
        ContinuousCDF::sf(&chi300, 700.0),
        5.2670543937455585e-34,
        1e-9,
    );

    let gamma = Gamma::new(3.0_f64, 2.0, 0.0)?;
    assert_close(
        "Gamma::sf",
        ContinuousCDF::sf(&gamma, 150.0),
        7.737242864182633e-30,
        1e-9,
    );

    let beta = Beta::new(2.0_f64, 3.0, 0.0, 1.0)?;
    assert_close(
        "Beta::sf",
        ContinuousCDF::sf(&beta, 0.999999),
        3.999997e-18,
        1e-9,
    );
    let beta50 = Beta::new(50.0_f64, 50.0, 0.0, 1.0)?;
    assert_close("Beta::sf", beta50.sf(0.9), 3.232182234973745e-24, 1e-9);

    let ig = InverseGaussian::new(1.0_f64, 2.0)?;
    assert_close(
        "InverseGaussian::sf",
        ContinuousCDF::sf(&ig, 3.0),
        0.021456426126114547,
        1e-10,
    );
    assert_close(
        "InverseGaussian::sf",
        ig.sf(10.0),
        4.792648627575855e-06,
        1e-9,
    );
    Ok(())
}

/// `hazard` and `cumhazard` are built on `sf`; with `1 - cdf` they turned into
/// `inf` far in the tail. For the standard normal at z = 20 the cumulative
/// hazard is -ln(sf) = 203.917... (mpmath: -ln(2.7536241186062337e-89)).
#[test]
fn test_cumhazard_uses_direct_tail() -> TestResult {
    let normal = Normal::new(0.0_f64, 1.0)?;
    assert_close(
        "cumhazard",
        normal.cumhazard(20.0),
        203.917_155_371_097_26,
        1e-12,
    );
    assert!(normal.hazard(20.0).is_finite());
    Ok(())
}
