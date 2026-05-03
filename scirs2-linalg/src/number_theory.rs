//! Number theory algorithms used in numerical methods.
//!
//! Provides primality testing, factorization, modular arithmetic, the Chinese
//! Remainder Theorem, Number Theoretic Transform (NTT), and lattice reduction
//! (LLL algorithm).
//!
//! # Quick start
//!
//! ```rust
//! use scirs2_linalg::number_theory::{is_prime, gcd, primes_up_to, ntt_multiply};
//!
//! assert!(is_prime(17));
//! assert_eq!(gcd(12, 8), 4);
//! let primes = primes_up_to(20);
//! assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19]);
//!
//! // Polynomial multiplication mod a prime via NTT
//! let a = vec![1i64, 2, 3];
//! let b = vec![4i64, 5, 6];
//! let MOD: i64 = 998_244_353;
//! let g: i64 = 3;
//! let product = ntt_multiply(&a, &b, MOD, g);
//! assert_eq!(product[0], 4);   // 1*4
//! assert_eq!(product[1], 13);  // 1*5 + 2*4
//! assert_eq!(product[2], 28);  // 1*6 + 2*5 + 3*4
//! ```

use crate::error::{LinalgError, LinalgResult};

// ============================================================================
// Primality tests
// ============================================================================

/// Miller–Rabin primality test (deterministic for n < 3,317,044,064,679,887,385,961,981).
///
/// Uses the 12 deterministic witnesses that cover all 64-bit integers.
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::number_theory::is_prime;
/// assert!(!is_prime(0));
/// assert!(!is_prime(1));
/// assert!(is_prime(2));
/// assert!(is_prime(7_919));
/// assert!(!is_prime(7_921)); // 89^2
/// ```
pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n < 4 {
        return true;
    }
    if n % 2 == 0 || n % 3 == 0 {
        return false;
    }

    // Deterministic witnesses for all n < 3,317,044,064,679,887,385,961,981
    let witnesses: &[u64] = &[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

    // Write n-1 as 2^r * d
    let mut d = n - 1;
    let mut r = 0u32;
    while d % 2 == 0 {
        d /= 2;
        r += 1;
    }

    'outer: for &a in witnesses {
        if a >= n {
            continue;
        }
        let mut x = mod_pow(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 0..r - 1 {
            x = mul_mod(x, x, n);
            if x == n - 1 {
                continue 'outer;
            }
        }
        return false;
    }
    true
}

/// Sieve of Eratosthenes: returns all primes up to and including `limit`.
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::number_theory::primes_up_to;
/// assert_eq!(primes_up_to(10), vec![2, 3, 5, 7]);
/// ```
pub fn primes_up_to(limit: usize) -> Vec<usize> {
    if limit < 2 {
        return vec![];
    }
    let mut is_composite = vec![false; limit + 1];
    is_composite[0] = true;
    is_composite[1] = true;
    let mut p = 2;
    while p * p <= limit {
        if !is_composite[p] {
            let mut multiple = p * p;
            while multiple <= limit {
                is_composite[multiple] = true;
                multiple += p;
            }
        }
        p += 1;
    }
    (2..=limit).filter(|&i| !is_composite[i]).collect()
}

// ============================================================================
// GCD / LCM
// ============================================================================

/// Greatest common divisor via the Euclidean algorithm.
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::number_theory::gcd;
/// assert_eq!(gcd(48, 18), 6);
/// assert_eq!(gcd(0, 5), 5);
/// ```
pub fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Least common multiple.
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::number_theory::lcm;
/// assert_eq!(lcm(4, 6), 12);
/// ```
pub fn lcm(a: u64, b: u64) -> u64 {
    if a == 0 || b == 0 {
        return 0;
    }
    a / gcd(a, b) * b
}

/// Extended Euclidean algorithm.
///
/// Returns `(g, x, y)` such that `a*x + b*y = g = gcd(|a|, |b|)`.
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::number_theory::extended_gcd;
/// let (g, x, y) = extended_gcd(30, 12);
/// assert_eq!(g, 6);
/// assert_eq!(30 * x + 12 * y, g);
/// ```
pub fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
    if b == 0 {
        if a < 0 {
            return (-a, -1, 0);
        }
        return (a, 1, 0);
    }
    let (g, x1, y1) = extended_gcd(b, a % b);
    (g, y1, x1 - (a / b) * y1)
}

// ============================================================================
// Modular arithmetic
// ============================================================================

/// Compute `base^exp mod modulus` using fast exponentiation by squaring.
///
/// Uses 128-bit intermediate to avoid overflow for 64-bit inputs.
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::number_theory::mod_pow;
/// assert_eq!(mod_pow(2, 10, 1000), 24);
/// ```
pub fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
    let mut result: u64 = 1;
    base %= modulus;
    while exp > 0 {
        if exp % 2 == 1 {
            result = mul_mod(result, base, modulus);
        }
        exp /= 2;
        base = mul_mod(base, base, modulus);
    }
    result
}

/// Modular inverse of `a` modulo `m` (requires gcd(a, m) = 1).
///
/// Returns `None` if the inverse does not exist.
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::number_theory::mod_inverse;
/// assert_eq!(mod_inverse(3, 7), Some(5)); // 3*5 = 15 ≡ 1 (mod 7)
/// assert_eq!(mod_inverse(2, 4), None);    // gcd(2,4) ≠ 1
/// ```
pub fn mod_inverse(a: i64, m: i64) -> Option<i64> {
    if m <= 1 {
        return None;
    }
    let (g, x, _) = extended_gcd(a.rem_euclid(m), m);
    if g != 1 {
        return None;
    }
    Some(x.rem_euclid(m))
}

/// Chinese Remainder Theorem: find `x` such that `x ≡ remainders[i] (mod moduli[i])`.
///
/// The moduli must be pairwise coprime. Returns `None` if the system has no solution
/// or if a modular inverse cannot be computed.
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::number_theory::crt;
/// // x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7) => x ≡ 23 (mod 105)
/// let x = crt(&[2, 3, 2], &[3, 5, 7]).expect("valid input");
/// assert_eq!(x % 3, 2);
/// assert_eq!(x % 5, 3);
/// assert_eq!(x % 7, 2);
/// ```
pub fn crt(remainders: &[i64], moduli: &[i64]) -> Option<i64> {
    if remainders.len() != moduli.len() || remainders.is_empty() {
        return None;
    }
    let m: i64 = moduli.iter().product();
    let mut x: i64 = 0;
    for (&r, &mi) in remainders.iter().zip(moduli.iter()) {
        let big_m = m / mi;
        let inv = mod_inverse(big_m % mi, mi)?;
        // Accumulate with overflow-safe i128 arithmetic
        x = (x as i128 + r as i128 * big_m as i128 * inv as i128).rem_euclid(m as i128) as i64;
    }
    Some(x)
}

/// Euler's totient function φ(n): count of integers in [1, n] coprime to n.
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::number_theory::euler_totient;
/// assert_eq!(euler_totient(1), 1);
/// assert_eq!(euler_totient(6), 2); // 1, 5
/// assert_eq!(euler_totient(7), 6); // 1..6 all coprime to prime 7
/// ```
pub fn euler_totient(mut n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let original = n;
    let mut result = n;
    let mut p = 2u64;
    while p * p <= n {
        if n % p == 0 {
            while n % p == 0 {
                n /= p;
            }
            result -= result / p;
        }
        p += 1;
    }
    if n > 1 {
        result -= result / n;
    }
    let _ = original;
    result
}

/// Prime factorization: returns `(prime, exponent)` pairs in ascending order.
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::number_theory::prime_factorization;
/// assert_eq!(prime_factorization(12), vec![(2, 2), (3, 1)]);
/// assert_eq!(prime_factorization(1), vec![]);
/// ```
pub fn prime_factorization(mut n: u64) -> Vec<(u64, u32)> {
    let mut factors = Vec::new();
    let mut p = 2u64;
    while p * p <= n {
        if n % p == 0 {
            let mut exp = 0u32;
            while n % p == 0 {
                n /= p;
                exp += 1;
            }
            factors.push((p, exp));
        }
        p += 1;
    }
    if n > 1 {
        factors.push((n, 1));
    }
    factors
}

// ============================================================================
// Legendre symbol and quadratic residues
// ============================================================================

/// Legendre symbol `(a/p)` for odd prime `p`.
///
/// Returns 0 if `p | a`, 1 if `a` is a quadratic residue mod `p`,
/// and -1 otherwise.
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::number_theory::legendre_symbol;
/// assert_eq!(legendre_symbol(2, 7), 1);  // 2 is a QR mod 7 (3^2=9≡2)
/// assert_eq!(legendre_symbol(3, 7), -1);
/// assert_eq!(legendre_symbol(7, 7), 0);
/// ```
pub fn legendre_symbol(a: i64, p: i64) -> i32 {
    if p <= 1 {
        return 0;
    }
    let a_mod = a.rem_euclid(p) as u64;
    let p_u64 = p as u64;
    if a_mod == 0 {
        return 0;
    }
    let val = mod_pow(a_mod, (p_u64 - 1) / 2, p_u64);
    if val == 1 {
        1
    } else {
        -1
    }
}

/// Check whether `a` is a quadratic residue modulo prime `p`.
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::number_theory::is_quadratic_residue;
/// assert!(is_quadratic_residue(4, 7));  // 2^2 = 4
/// assert!(!is_quadratic_residue(3, 7));
/// ```
pub fn is_quadratic_residue(a: u64, p: u64) -> bool {
    if p == 2 {
        return true;
    }
    let a_mod = a % p;
    if a_mod == 0 {
        return true;
    }
    mod_pow(a_mod, (p - 1) / 2, p) == 1
}

/// Tonelli–Shanks algorithm: compute `sqrt(n) mod p` for odd prime `p`.
///
/// Returns `None` if `n` is not a quadratic residue mod `p`.
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::number_theory::sqrt_mod_prime;
/// let r = sqrt_mod_prime(2, 7).expect("valid input");
/// assert_eq!((r * r) % 7, 2 % 7);
/// ```
pub fn sqrt_mod_prime(n: u64, p: u64) -> Option<u64> {
    let n = n % p;
    if n == 0 {
        return Some(0);
    }
    if !is_quadratic_residue(n, p) {
        return None;
    }
    if p % 4 == 3 {
        return Some(mod_pow(n, (p + 1) / 4, p));
    }

    // Factor p-1 = Q * 2^S
    let mut q = p - 1;
    let mut s = 0u32;
    while q % 2 == 0 {
        q /= 2;
        s += 1;
    }

    // Find a non-residue z
    let z = (2..p).find(|&z| !is_quadratic_residue(z, p)).unwrap_or(2);

    let mut m = s;
    let mut c = mod_pow(z, q, p);
    let mut t = mod_pow(n, q, p);
    let mut r = mod_pow(n, q.div_ceil(2), p);

    loop {
        if t == 1 {
            return Some(r);
        }
        // Find least i > 0 such that t^(2^i) ≡ 1
        let mut i = 1u32;
        let mut tmp = mul_mod(t, t, p);
        while tmp != 1 && i < m {
            tmp = mul_mod(tmp, tmp, p);
            i += 1;
        }
        if i == m {
            return None; // should not happen
        }
        let b = mod_pow(c, mod_pow(2, (m - i - 1) as u64, p - 1), p);
        m = i;
        c = mul_mod(b, b, p);
        t = mul_mod(t, c, p);
        r = mul_mod(r, b, p);
    }
}

// ============================================================================
// Number Theoretic Transform (NTT)
// ============================================================================

/// In-place Number Theoretic Transform (or its inverse).
///
/// Operates on `a` of length equal to a power of 2, modulo `modulus`,
/// using `primitive_root` as the generator of the multiplicative group.
///
/// For `invert = true`, the inverse NTT is computed (applies `n^{-1} mod p`).
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::number_theory::ntt;
/// let MOD: i64 = 998_244_353;
/// let g: i64 = 3;
/// let mut a = vec![1i64, 2, 3, 4];
/// let b = a.clone();
/// ntt(&mut a, false, MOD, g);
/// ntt(&mut a, true, MOD, g);
/// assert_eq!(a, b);
/// ```
pub fn ntt(a: &mut [i64], invert: bool, modulus: i64, primitive_root: i64) {
    let n = a.len();
    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            a.swap(i, j);
        }
    }

    let mut len = 2usize;
    while len <= n {
        let w = if invert {
            // w = primitive_root^((modulus-1) - (modulus-1)/len) mod modulus
            mod_pow_i64(
                primitive_root,
                modulus - 1 - (modulus - 1) / len as i64,
                modulus,
            )
        } else {
            mod_pow_i64(primitive_root, (modulus - 1) / len as i64, modulus)
        };
        let mut i = 0;
        while i < n {
            let mut wn = 1i64;
            for jj in 0..len / 2 {
                let u = a[i + jj];
                let v =
                    (a[i + jj + len / 2] as i128 * wn as i128).rem_euclid(modulus as i128) as i64;
                a[i + jj] = (u + v).rem_euclid(modulus);
                a[i + jj + len / 2] = (u - v).rem_euclid(modulus);
                wn = (wn as i128 * w as i128).rem_euclid(modulus as i128) as i64;
            }
            i += len;
        }
        len <<= 1;
    }

    if invert {
        let n_inv = mod_pow_i64(n as i64, modulus - 2, modulus);
        for x in a.iter_mut() {
            *x = (*x as i128 * n_inv as i128).rem_euclid(modulus as i128) as i64;
        }
    }
}

/// Polynomial multiplication via NTT modulo `modulus`.
///
/// Returns the coefficient vector of `a * b` (length `a.len() + b.len() - 1`).
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::number_theory::ntt_multiply;
/// let MOD: i64 = 998_244_353;
/// let g: i64 = 3;
/// let prod = ntt_multiply(&[1, 2, 3], &[4, 5, 6], MOD, g);
/// // (1 + 2x + 3x^2)(4 + 5x + 6x^2) = 4 + 13x + 28x^2 + 27x^3 + 18x^4
/// assert_eq!(prod[0], 4);
/// assert_eq!(prod[1], 13);
/// assert_eq!(prod[2], 28);
/// assert_eq!(prod[3], 27);
/// assert_eq!(prod[4], 18);
/// ```
pub fn ntt_multiply(a: &[i64], b: &[i64], modulus: i64, primitive_root: i64) -> Vec<i64> {
    let result_len = a.len() + b.len() - 1;
    let n = result_len.next_power_of_two();
    let mut fa: Vec<i64> = a.iter().map(|&x| x.rem_euclid(modulus)).collect();
    fa.resize(n, 0);
    let mut fb: Vec<i64> = b.iter().map(|&x| x.rem_euclid(modulus)).collect();
    fb.resize(n, 0);

    ntt(&mut fa, false, modulus, primitive_root);
    ntt(&mut fb, false, modulus, primitive_root);

    for i in 0..n {
        fa[i] = (fa[i] as i128 * fb[i] as i128).rem_euclid(modulus as i128) as i64;
    }
    ntt(&mut fa, true, modulus, primitive_root);
    fa.truncate(result_len);
    fa
}

// ============================================================================
// LLL Basis Reduction
// ============================================================================

/// Lenstra–Lenstra–Lovász (LLL) lattice basis reduction.
///
/// Given a basis as rows of `basis`, returns a reduced basis (also as rows)
/// with `delta` in (0.25, 1.0] (typically 0.75).
///
/// The algorithm runs in polynomial time and produces a basis where the first
/// vector has length at most `2^{(n-1)/2}` times the length of the shortest
/// vector in the lattice.
///
/// # Arguments
///
/// * `basis` - Rows are basis vectors (n × d matrix, n vectors in R^d)
/// * `delta` - Lovász condition parameter in (0.25, 1.0], typically 0.75
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::number_theory::lll_reduce;
/// // Standard LLL example
/// let basis = vec![
///     vec![1.0_f64, 1.0, 1.0],
///     vec![-1.0, 0.0, 2.0],
///     vec![3.0, 5.0, 6.0],
/// ];
/// let reduced = lll_reduce(&basis, 0.75);
/// assert_eq!(reduced.len(), 3);
/// // First basis vector should be short
/// let len0: f64 = reduced[0].iter().map(|x| x * x).sum::<f64>().sqrt();
/// assert!(len0 < 10.0);
/// ```
pub fn lll_reduce(basis: &[Vec<f64>], delta: f64) -> Vec<Vec<f64>> {
    if basis.is_empty() {
        return vec![];
    }
    let n = basis.len();
    let d = basis[0].len();

    // Work on a mutable copy
    let mut b: Vec<Vec<f64>> = basis.to_vec();
    // Gram-Schmidt orthogonalization (stored as b_star[i] and mu[i][j])
    let mut b_star: Vec<Vec<f64>> = vec![vec![0.0; d]; n];
    let mut mu: Vec<Vec<f64>> = vec![vec![0.0; n]; n];

    // Helper closures
    let dot = |u: &[f64], v: &[f64]| -> f64 { u.iter().zip(v).map(|(a, b)| a * b).sum() };
    let norm_sq = |u: &[f64]| -> f64 { u.iter().map(|x| x * x).sum() };

    // Gram-Schmidt
    let gram_schmidt = |b: &[Vec<f64>], b_star: &mut Vec<Vec<f64>>, mu: &mut Vec<Vec<f64>>| {
        let n = b.len();
        let d = b[0].len();
        for i in 0..n {
            b_star[i] = b[i].clone();
            for j in 0..i {
                let mu_ij = dot(&b[i], &b_star[j]) / dot(&b_star[j], &b_star[j]).max(1e-300);
                mu[i][j] = mu_ij;
                for k in 0..d {
                    b_star[i][k] -= mu_ij * b_star[j][k];
                }
            }
        }
    };

    gram_schmidt(&b, &mut b_star, &mut mu);

    let mut k = 1usize;
    while k < n {
        // Size reduce b[k] against b[j] for j = k-1 downto 0
        for j in (0..k).rev() {
            let mu_kj = mu[k][j];
            if mu_kj.abs() > 0.5 {
                let rounded = mu_kj.round();
                for l in 0..d {
                    let bj_l = b[j][l];
                    b[k][l] -= rounded * bj_l;
                }
                gram_schmidt(&b, &mut b_star, &mut mu);
            }
        }

        // Lovász condition: ||b*_k||^2 >= (delta - mu[k][k-1]^2) * ||b*_{k-1}||^2
        let lhs = norm_sq(&b_star[k]);
        let rhs = (delta - mu[k][k - 1].powi(2)) * norm_sq(&b_star[k - 1]);
        if lhs >= rhs {
            k += 1;
        } else {
            b.swap(k, k - 1);
            gram_schmidt(&b, &mut b_star, &mut mu);
            if k > 1 {
                k -= 1;
            }
        }
    }
    b
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Multiply a * b mod m using 128-bit intermediates (avoids 64-bit overflow).
#[inline(always)]
fn mul_mod(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

/// Modular exponentiation for signed i64 values (reduces to u64 internally).
fn mod_pow_i64(base: i64, exp: i64, modulus: i64) -> i64 {
    let b = base.rem_euclid(modulus) as u64;
    let e = exp.rem_euclid(modulus - 1) as u64; // By Fermat's little theorem when p is prime
    mod_pow(b, e, modulus as u64) as i64
}

// ============================================================================
// Convenience wrapper with Result
// ============================================================================

/// Wrapper for [`crt`] that returns a [`LinalgResult`].
pub fn crt_result(remainders: &[i64], moduli: &[i64]) -> LinalgResult<i64> {
    crt(remainders, moduli).ok_or_else(|| {
        LinalgError::ComputationError(
            "CRT: no solution exists (moduli not pairwise coprime?)".into(),
        )
    })
}

/// Wrapper for [`sqrt_mod_prime`] that returns a [`LinalgResult`].
pub fn sqrt_mod_prime_result(n: u64, p: u64) -> LinalgResult<u64> {
    sqrt_mod_prime(n, p).ok_or_else(|| {
        LinalgError::ComputationError(format!("{} is not a quadratic residue mod {}", n, p))
    })
}

/// Wrapper for [`mod_inverse`] that returns a [`LinalgResult`].
pub fn mod_inverse_result(a: i64, m: i64) -> LinalgResult<i64> {
    mod_inverse(a, m).ok_or_else(|| {
        LinalgError::ComputationError(format!("No modular inverse for {} mod {}", a, m))
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_prime() {
        assert!(!is_prime(0));
        assert!(!is_prime(1));
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(!is_prime(4));
        assert!(is_prime(7_919));
        assert!(!is_prime(7_921)); // 89^2
        assert!(is_prime(998_244_353)); // NTT prime
        assert!(is_prime(1_000_000_007));
    }

    #[test]
    fn test_primes_up_to() {
        assert_eq!(primes_up_to(0), Vec::<usize>::new());
        assert_eq!(primes_up_to(1), Vec::<usize>::new());
        assert_eq!(primes_up_to(2), vec![2]);
        assert_eq!(primes_up_to(10), vec![2, 3, 5, 7]);
        assert_eq!(primes_up_to(20), vec![2, 3, 5, 7, 11, 13, 17, 19]);
    }

    #[test]
    fn test_gcd_lcm() {
        assert_eq!(gcd(0, 5), 5);
        assert_eq!(gcd(48, 18), 6);
        assert_eq!(lcm(4, 6), 12);
        assert_eq!(lcm(0, 5), 0);
    }

    #[test]
    fn test_extended_gcd() {
        let (g, x, y) = extended_gcd(30, 12);
        assert_eq!(g, 6);
        assert_eq!(30 * x + 12 * y, g);
        let (g2, x2, y2) = extended_gcd(7, 3);
        assert_eq!(g2, 1);
        assert_eq!(7 * x2 + 3 * y2, 1);
    }

    #[test]
    fn test_mod_pow() {
        assert_eq!(mod_pow(2, 10, 1000), 24);
        assert_eq!(mod_pow(3, 0, 5), 1);
        // 2^62 mod 1_000_000_007 = 145_586_002
        assert_eq!(mod_pow(2, 62, 1_000_000_007), 145_586_002);
    }

    #[test]
    fn test_mod_inverse() {
        assert_eq!(mod_inverse(3, 7), Some(5));
        assert_eq!(mod_inverse(2, 4), None);
        // Verify: 3 * 5 mod 7 = 1
        assert_eq!((3 * 5) % 7, 1);
    }

    #[test]
    fn test_crt() {
        let x = crt(&[2, 3, 2], &[3, 5, 7]).expect("failed to create x");
        assert_eq!(x % 3, 2);
        assert_eq!(x % 5, 3);
        assert_eq!(x % 7, 2);
    }

    #[test]
    fn test_euler_totient() {
        assert_eq!(euler_totient(1), 1);
        assert_eq!(euler_totient(6), 2);
        assert_eq!(euler_totient(7), 6);
        assert_eq!(euler_totient(12), 4);
    }

    #[test]
    fn test_prime_factorization() {
        assert_eq!(prime_factorization(1), Vec::<(u64, u32)>::new());
        assert_eq!(prime_factorization(12), vec![(2, 2), (3, 1)]);
        assert_eq!(prime_factorization(2), vec![(2, 1)]);
        assert_eq!(prime_factorization(360), vec![(2, 3), (3, 2), (5, 1)]);
    }

    #[test]
    fn test_legendre_symbol() {
        // 2 is a QR mod 7 (3^2=9≡2), so legendre(2,7)=1
        assert_eq!(legendre_symbol(2, 7), 1);
        assert_eq!(legendre_symbol(3, 7), -1);
        assert_eq!(legendre_symbol(7, 7), 0);
    }

    #[test]
    fn test_sqrt_mod_prime() {
        let r = sqrt_mod_prime(2, 7).expect("failed to create r");
        assert_eq!((r * r) % 7, 2);
        assert!(sqrt_mod_prime(3, 7).is_none());
        let r2 = sqrt_mod_prime(4, 7).expect("failed to create r2");
        assert_eq!((r2 * r2) % 7, 4);
    }

    #[test]
    fn test_ntt_roundtrip() {
        let modulus: i64 = 998_244_353;
        let g: i64 = 3;
        let original = vec![1i64, 2, 3, 4];
        let mut a = original.clone();
        ntt(&mut a, false, modulus, g);
        ntt(&mut a, true, modulus, g);
        assert_eq!(a, original);
    }

    #[test]
    fn test_ntt_multiply() {
        let modulus: i64 = 998_244_353;
        let g: i64 = 3;
        // (1 + 2x + 3x^2)(4 + 5x + 6x^2) = 4 + 13x + 28x^2 + 27x^3 + 18x^4
        let prod = ntt_multiply(&[1, 2, 3], &[4, 5, 6], modulus, g);
        assert_eq!(prod, vec![4, 13, 28, 27, 18]);
    }

    #[test]
    fn test_lll_reduce() {
        let basis = vec![
            vec![1.0_f64, 1.0, 1.0],
            vec![-1.0, 0.0, 2.0],
            vec![3.0, 5.0, 6.0],
        ];
        let reduced = lll_reduce(&basis, 0.75);
        assert_eq!(reduced.len(), 3);
        // All reduced vectors should be shorter than original last vector
        let orig_len: f64 = basis[2].iter().map(|x| x * x).sum::<f64>().sqrt();
        let red0_len: f64 = reduced[0].iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(red0_len <= orig_len + 1e-9);
    }
}
