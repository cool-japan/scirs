# SciRS2 Special

[![crates.io](https://img.shields.io/crates/v/scirs2-special.svg)](https://crates.io/crates/scirs2-special)
[![License](https://img.shields.io/crates/l/scirs2-special.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-special)](https://docs.rs/scirs2-special)

Special functions module for the SciRS2 scientific computing library. This module provides implementations of various special mathematical functions used in scientific computing, engineering, and statistics.

## Features

- **Elementary Special Functions**: Gamma, error, and related functions
- **Bessel Functions**: Various kinds of Bessel functions
- **Elliptic Functions**: Complete and incomplete elliptic integrals
- **Orthogonal Polynomials**: Legendre, Chebyshev, Hermite, Laguerre polynomials
- **Spherical Harmonics**: Implementation of spherical harmonic functions
- **Airy Functions**: Airy functions and their derivatives
- **Hypergeometric Functions**: Various hypergeometric functions
- **Mathieu Functions**: Solutions to Mathieu's differential equation
- **Zeta Functions**: Riemann and Hurwitz zeta functions

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
scirs2-special = "0.1.0-alpha.1"
```

To enable optimizations through the core module, add feature flags:

```toml
[dependencies]
scirs2-special = { version = "0.1.0-alpha.1", features = ["parallel"] }
```

## Usage

Basic usage examples:

```rust
use scirs2_special::{gamma, erf, bessel, elliptic, orthogonal, airy, hypergeometric};
use scirs2_core::error::CoreResult;

// Gamma and related functions
fn gamma_example() -> CoreResult<()> {
    // Gamma function
    let x = 4.5;
    let gamma_x = gamma::gamma(x)?;
    println!("Gamma({}) = {}", x, gamma_x);
    
    // Log gamma function (more numerically stable for large inputs)
    let log_gamma_x = gamma::log_gamma(x)?;
    println!("Log-gamma({}) = {}", x, log_gamma_x);
    
    // Beta function
    let a = 2.0;
    let b = 3.0;
    let beta_ab = gamma::beta(a, b)?;
    println!("Beta({}, {}) = {}", a, b, beta_ab);
    
    // Incomplete gamma function
    let x = 2.0;
    let a = 1.5;
    let inc_gamma = gamma::inc_gamma(a, x)?;
    println!("Incomplete gamma({}, {}) = {}", a, x, inc_gamma);
    
    Ok(())
}

// Error function and related functions
fn erf_example() -> CoreResult<()> {
    // Error function
    let x = 1.0;
    let erf_x = erf::erf(x)?;
    println!("erf({}) = {}", x, erf_x);
    
    // Complementary error function
    let erfc_x = erf::erfc(x)?;
    println!("erfc({}) = {}", x, erfc_x);
    
    // Scaled complementary error function
    let erfcx_x = erf::erfcx(x)?;
    println!("erfcx({}) = {}", x, erfcx_x);
    
    // Error function integral
    let erfi_x = erf::erfi(x)?;
    println!("erfi({}) = {}", x, erfi_x);
    
    // Inverse error function
    let inv_erf_x = erf::inverse_erf(0.8)?;
    println!("inverse_erf(0.8) = {}", inv_erf_x);
    
    Ok(())
}

// Bessel functions
fn bessel_example() -> CoreResult<()> {
    // Bessel function of the first kind
    let n = 0;
    let x = 1.0;
    let j0 = bessel::j(n, x)?;
    println!("J_{}({}) = {}", n, x, j0);
    
    // Bessel function of the second kind
    let y0 = bessel::y(n, x)?;
    println!("Y_{}({}) = {}", n, x, y0);
    
    // Modified Bessel function of the first kind
    let i0 = bessel::i(n, x)?;
    println!("I_{}({}) = {}", n, x, i0);
    
    // Modified Bessel function of the second kind
    let k0 = bessel::k(n, x)?;
    println!("K_{}({}) = {}", n, x, k0);
    
    // Spherical Bessel function
    let sj0 = bessel::spherical_jn(n, x)?;
    println!("spherical_jn({}, {}) = {}", n, x, sj0);
    
    Ok(())
}

// Elliptic functions
fn elliptic_example() -> CoreResult<()> {
    // Complete elliptic integral of the first kind
    let k = 0.5;
    let k1 = elliptic::ellipk(k)?;
    println!("K({}) = {}", k, k1);
    
    // Complete elliptic integral of the second kind
    let e1 = elliptic::ellipe(k)?;
    println!("E({}) = {}", k, e1);
    
    // Incomplete elliptic integral of the first kind
    let phi = 0.5;
    let f1 = elliptic::ellipkinc(phi, k)?;
    println!("F({}, {}) = {}", phi, k, f1);
    
    // Incomplete elliptic integral of the second kind
    let e1 = elliptic::ellipeinc(phi, k)?;
    println!("E({}, {}) = {}", phi, k, e1);
    
    Ok(())
}

// Orthogonal polynomials
fn orthogonal_example() -> CoreResult<()> {
    // Legendre polynomial
    let n = 3;
    let x = 0.5;
    let p3 = orthogonal::legendre(n, x)?;
    println!("P_{}({}) = {}", n, x, p3);
    
    // Chebyshev polynomial of the first kind
    let t3 = orthogonal::chebyshev_t(n, x)?;
    println!("T_{}({}) = {}", n, x, t3);
    
    // Chebyshev polynomial of the second kind
    let u3 = orthogonal::chebyshev_u(n, x)?;
    println!("U_{}({}) = {}", n, x, u3);
    
    // Hermite polynomial (physicist's version)
    let h3 = orthogonal::hermite(n, x)?;
    println!("H_{}({}) = {}", n, x, h3);
    
    // Laguerre polynomial
    let l3 = orthogonal::laguerre(n, x)?;
    println!("L_{}({}) = {}", n, x, l3);
    
    Ok(())
}

// Airy functions
fn airy_example() -> CoreResult<()> {
    let x = 1.0;
    
    // Airy function of the first kind
    let ai = airy::airy_ai(x)?;
    println!("Ai({}) = {}", x, ai);
    
    // Derivative of Airy function of the first kind
    let ai_prime = airy::airy_ai_prime(x)?;
    println!("Ai'({}) = {}", x, ai_prime);
    
    // Airy function of the second kind
    let bi = airy::airy_bi(x)?;
    println!("Bi({}) = {}", x, bi);
    
    // Derivative of Airy function of the second kind
    let bi_prime = airy::airy_bi_prime(x)?;
    println!("Bi'({}) = {}", x, bi_prime);
    
    Ok(())
}

// Hypergeometric functions
fn hypergeometric_example() -> CoreResult<()> {
    // Hypergeometric function 1F1
    let a = 1.0;
    let b = 2.0;
    let x = 0.5;
    let hyp1f1 = hypergeometric::hyp1f1(a, b, x)?;
    println!("1F1({}, {}, {}) = {}", a, b, x, hyp1f1);
    
    // Hypergeometric function 2F1
    let a = 1.0;
    let b = 2.0;
    let c = 3.0;
    let x = 0.3;
    let hyp2f1 = hypergeometric::hyp2f1(a, b, c, x)?;
    println!("2F1({}, {}, {}, {}) = {}", a, b, c, x, hyp2f1);
    
    Ok(())
}
```

## Components

### Gamma Functions

Gamma and related functions:

```rust
use scirs2_special::gamma::{
    gamma,                  // Gamma function
    log_gamma,              // Natural logarithm of gamma function
    digamma,                // Digamma function (derivative of log_gamma)
    trigamma,               // Trigamma function (second derivative of log_gamma)
    beta,                   // Beta function
    inc_gamma,              // Incomplete gamma function
    inc_gamma_upper,        // Upper incomplete gamma function
    inc_beta,               // Incomplete beta function
    factorial,              // Factorial function
    binom,                  // Binomial coefficient
};
```

### Error Functions

Error function and variants:

```rust
use scirs2_special::erf::{
    erf,                    // Error function
    erfc,                   // Complementary error function
    erfcx,                  // Scaled complementary error function
    erfi,                   // Imaginary error function
    dawsn,                  // Dawson's integral
    inverse_erf,            // Inverse error function
    inverse_erfc,           // Inverse complementary error function
};
```

### Bessel Functions

Bessel and related functions:

```rust
use scirs2_special::bessel::{
    // Bessel functions of the first kind
    j,                      // Bessel function of the first kind
    j0,                     // Bessel function of the first kind, order 0
    j1,                     // Bessel function of the first kind, order 1
    
    // Bessel functions of the second kind
    y,                      // Bessel function of the second kind
    y0,                     // Bessel function of the second kind, order 0
    y1,                     // Bessel function of the second kind, order 1
    
    // Modified Bessel functions
    i,                      // Modified Bessel function of the first kind
    i0,                     // Modified Bessel function of the first kind, order 0
    i1,                     // Modified Bessel function of the first kind, order 1
    k,                      // Modified Bessel function of the second kind
    k0,                     // Modified Bessel function of the second kind, order 0
    k1,                     // Modified Bessel function of the second kind, order 1
    
    // Spherical Bessel functions
    spherical_jn,           // Spherical Bessel function of the first kind
    spherical_yn,           // Spherical Bessel function of the second kind
    
    // Hankel functions
    hankel1,                // Hankel function of the first kind
    hankel2,                // Hankel function of the second kind
};
```

### Elliptic Functions

Elliptic integrals and related functions:

```rust
use scirs2_special::elliptic::{
    // Complete elliptic integrals
    ellipk,                 // Complete elliptic integral of the first kind
    ellipe,                 // Complete elliptic integral of the second kind
    ellippi,                // Complete elliptic integral of the third kind
    
    // Incomplete elliptic integrals
    ellipkinc,              // Incomplete elliptic integral of the first kind
    ellipeinc,              // Incomplete elliptic integral of the second kind
    ellippinc,              // Incomplete elliptic integral of the third kind
    
    // Jacobi elliptic functions
    jacobi_sn,              // Jacobi elliptic function sn
    jacobi_cn,              // Jacobi elliptic function cn
    jacobi_dn,              // Jacobi elliptic function dn
    
    // Carlson's elliptic integrals
    elliprf,                // Carlson's elliptic integral of the first kind
    elliprd,                // Carlson's elliptic integral of the second kind
    elliprj,                // Carlson's elliptic integral of the third kind
};
```

### Orthogonal Polynomials

Various orthogonal polynomials:

```rust
use scirs2_special::orthogonal::{
    // Legendre polynomials
    legendre,               // Legendre polynomial
    legendre_p,             // Associated Legendre polynomial
    
    // Chebyshev polynomials
    chebyshev_t,            // Chebyshev polynomial of the first kind
    chebyshev_u,            // Chebyshev polynomial of the second kind
    
    // Hermite polynomials
    hermite,                // Hermite polynomial (physicist's version)
    hermite_h,              // Hermite polynomial (probabilist's version)
    
    // Laguerre polynomials
    laguerre,               // Laguerre polynomial
    laguerre_l,             // Associated Laguerre polynomial
    
    // Gegenbauer polynomials
    gegenbauer,             // Gegenbauer polynomial
    
    // Jacobi polynomials
    jacobi,                 // Jacobi polynomial
};
```

### Spherical Harmonics

Spherical harmonic functions:

```rust
use scirs2_special::spherical_harmonics::{
    sph_harm,               // Spherical harmonic function
    real_sph_harm,          // Real spherical harmonic function
    sph_harm_theta_phi,     // Spherical harmonic using theta and phi
    gaunt,                  // Gaunt coefficient (integral of 3 spherical harmonics)
};
```

### Airy Functions

Airy functions and their derivatives:

```rust
use scirs2_special::airy::{
    airy_ai,                // Airy function of the first kind
    airy_ai_prime,          // Derivative of Airy function of the first kind
    airy_bi,                // Airy function of the second kind
    airy_bi_prime,          // Derivative of Airy function of the second kind
};
```

### Hypergeometric Functions

Various hypergeometric functions:

```rust
use scirs2_special::hypergeometric::{
    hyp0f1,                 // Confluent hypergeometric limit function
    hyp1f1,                 // Kummer confluent hypergeometric function
    hyp2f1,                 // Gauss hypergeometric function
    hypu,                   // Tricomi confluent hypergeometric function
};
```

### Mathieu Functions

Solutions to Mathieu's differential equation:

```rust
use scirs2_special::mathieu::{
    mathieu_a,              // Characteristic value of even Mathieu function
    mathieu_b,              // Characteristic value of odd Mathieu function
    mathieu_ce,             // Even Mathieu function
    mathieu_se,             // Odd Mathieu function
    mathieu_mc,             // Radial Mathieu function of the first kind
    mathieu_ms,             // Radial Mathieu function of the second kind
};
```

### Zeta Functions

Zeta and related functions:

```rust
use scirs2_special::zeta::{
    zeta,                   // Riemann zeta function
    hurwitz_zeta,           // Hurwitz zeta function
    eta,                    // Dirichlet eta function
    lambert_w,              // Lambert W function
};
```

## Examples

The module includes several example applications in the `examples/` directory:

- `get_values.rs`: Basic usage of various special functions
- `airy_functions.rs`: Demonstration of Airy functions
- `elliptic_functions.rs`: Working with elliptic integrals
- `hypergeometric_functions.rs`: Hypergeometric function examples
- `mathieu_functions.rs`: Mathieu function demonstration
- `spherical_harmonics.rs`: Visualizing spherical harmonics
- `zeta_functions.rs`: Working with zeta functions

## Contributing

See the [CONTRIBUTING.md](../CONTRIBUTING.md) file for contribution guidelines.

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](../LICENSE) file for details.