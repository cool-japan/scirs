//! Crate-local `Either` sum type.
//!
//! This module replaces the external `either` crate (pure dependency
//! reduction). The public path `scirs2_stats::Either` is preserved via a
//! re-export in `lib.rs`, but the type is no longer interoperable with
//! `either::Either` from crates.io — it is an independent, crate-owned enum.

/// A value that is one of two possible variants: [`Either::Left`] or
/// [`Either::Right`].
///
/// This is a minimal, crate-local replacement for the `either` crate's
/// `Either` type (the external dependency was removed for dependency
/// reduction). Only the surface actually needed by `scirs2-stats` is
/// provided: construction, pattern matching, and a small set of
/// ergonomic helpers.
///
/// # Examples
///
/// ```
/// use scirs2_stats::Either;
///
/// let l: Either<i32, &str> = Either::Left(7);
/// assert!(l.is_left());
/// assert_eq!(l.left(), Some(7));
///
/// let r: Either<i32, &str> = Either::Right("stats");
/// assert!(r.is_right());
/// assert_eq!(r.right(), Some("stats"));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Either<L, R> {
    /// The left variant.
    Left(L),
    /// The right variant.
    Right(R),
}

impl<L, R> Either<L, R> {
    /// Returns `true` if this is the [`Either::Left`] variant.
    pub fn is_left(&self) -> bool {
        matches!(self, Either::Left(_))
    }

    /// Returns `true` if this is the [`Either::Right`] variant.
    pub fn is_right(&self) -> bool {
        matches!(self, Either::Right(_))
    }

    /// Consumes the value, returning the left value if present.
    pub fn left(self) -> Option<L> {
        match self {
            Either::Left(l) => Some(l),
            Either::Right(_) => None,
        }
    }

    /// Consumes the value, returning the right value if present.
    pub fn right(self) -> Option<R> {
        match self {
            Either::Left(_) => None,
            Either::Right(r) => Some(r),
        }
    }

    /// Maps the left value with `f`, leaving a right value untouched.
    pub fn map_left<T, F: FnOnce(L) -> T>(self, f: F) -> Either<T, R> {
        match self {
            Either::Left(l) => Either::Left(f(l)),
            Either::Right(r) => Either::Right(r),
        }
    }

    /// Maps the right value with `f`, leaving a left value untouched.
    pub fn map_right<T, F: FnOnce(R) -> T>(self, f: F) -> Either<L, T> {
        match self {
            Either::Left(l) => Either::Left(l),
            Either::Right(r) => Either::Right(f(r)),
        }
    }

    /// Converts `&Either<L, R>` to `Either<&L, &R>`.
    pub fn as_ref(&self) -> Either<&L, &R> {
        match self {
            Either::Left(l) => Either::Left(l),
            Either::Right(r) => Either::Right(r),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Either;

    #[test]
    fn construction_and_match() {
        let l: Either<i32, f64> = Either::Left(3);
        let r: Either<i32, f64> = Either::Right(2.5);

        match l {
            Either::Left(v) => assert_eq!(v, 3),
            Either::Right(_) => panic!("expected Left"),
        }
        match r {
            Either::Left(_) => panic!("expected Right"),
            Either::Right(v) => assert!((v - 2.5).abs() < f64::EPSILON),
        }
    }

    #[test]
    fn predicates_and_extractors() {
        let l: Either<i32, &str> = Either::Left(1);
        let r: Either<i32, &str> = Either::Right("x");

        assert!(l.is_left());
        assert!(!l.is_right());
        assert!(r.is_right());
        assert!(!r.is_left());

        assert_eq!(l.left(), Some(1));
        assert_eq!(l.right(), None);
        assert_eq!(r.left(), None);
        assert_eq!(r.right(), Some("x"));
    }

    #[test]
    fn map_left_and_map_right() {
        let l: Either<i32, &str> = Either::Left(10);
        let r: Either<i32, &str> = Either::Right("y");

        assert_eq!(l.map_left(|v| v * 2), Either::Left(20));
        assert_eq!(l.map_right(str::len), Either::Left(10));
        assert_eq!(r.map_left(|v| v * 2), Either::Right("y"));
        assert_eq!(r.map_right(str::len), Either::Right(1));
    }

    #[test]
    fn as_ref_and_derives() {
        let l: Either<i32, &str> = Either::Left(5);
        let copied = l; // Copy
        assert_eq!(l, copied); // PartialEq
        assert_eq!(l.as_ref(), Either::Left(&5));
        let cloned = l.clone(); // Clone (explicit for coverage)
        assert_eq!(format!("{cloned:?}"), "Left(5)"); // Debug
    }
}
