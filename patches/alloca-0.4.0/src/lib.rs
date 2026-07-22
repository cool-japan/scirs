//! Pure-Rust drop-in replacement for the upstream [`alloca`](https://crates.io/crates/alloca)
//! crate (patched in via `[patch.crates-io]` in the workspace root `Cargo.toml`).
//!
//! ## What this replaces, and why
//!
//! Upstream `alloca` 0.4.0 ships a `build.rs` that unconditionally runs `cc::Build::new()
//! .file("alloca.c")...compile("calloca")` -- a small C shim implementing a portable
//! `alloca()`-style stack allocation primitive -- with **no Cargo feature** to opt out of the C
//! compilation. It reaches this workspace transitively and unconditionally through `criterion`
//! (the `benches` / `scirs2-benchmarks` workspace member's plain, non-optional `criterion`
//! dependency): `criterion`'s measurement loop calls `alloca::with_alloca` on every
//! `Bencher::iter` invocation to vary the stack allocation size as an anti-measurement-bias
//! heuristic, so upstream's `alloca.c` was compiled on *every* workspace build, independent of
//! `--all-features`.
//!
//! Per the COOLJAPAN Pure Rust Policy ("no cc-compiled C in the build"; all compiled C/C++/Fortran
//! must be feature-gated behind a non-default feature, and upstream `alloca` offers none), this
//! crate reimplements the entire public API of upstream `alloca` 0.4.0 -- [`with_alloca`],
//! [`with_alloca_zeroed`], and [`alloca`] -- in Pure Rust, with no `build.rs` and no `cc`/`cmake`
//! build-dependency, so any consumer written against the upstream API (such as `criterion`)
//! compiles and runs unmodified against this patch.
//!
//! ## Semantic difference from upstream: heap instead of stack placement
//!
//! Upstream places the scratch buffer on the caller's *stack* via a real `alloca()` call.
//! This reimplementation instead backs the same buffer with a *heap* allocation
//! (`Vec<MaybeUninit<u8>>` / `Vec<u8>`) of the exact same requested length. Every observable part
//! of the contract that callers rely on is fully preserved:
//!
//! - the slice handed to the closure has exactly the requested length,
//! - the slice (and the memory behind it) is valid for the duration of the closure call only,
//! - the closure's return value is propagated back out of the function unchanged.
//!
//! Only the stack-vs-heap *placement* of the backing memory differs. For this workspace's sole
//! consumer, `criterion`, that placement is used purely as a measurement-jitter heuristic (varying
//! where scratch memory lives between benchmark iterations to reduce cache-alignment bias) and is
//! not a correctness requirement -- a heap allocation of the same varying size shifts the working
//! set through the address space just as effectively as a stack one does.

#![no_std]

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::mem::MaybeUninit;

/// Allocates `size` bytes of scratch memory and invokes `f` with a `&mut [MaybeUninit<u8>]` of
/// that length, returning whatever `f` returns.
///
/// Unlike upstream `alloca::with_alloca`, the memory backing the slice lives on the heap rather
/// than the stack -- see the crate-level documentation for why this is a safe, contract-preserving
/// substitution for this workspace's sole consumer, `criterion`.
pub fn with_alloca<R>(size: usize, f: impl FnOnce(&mut [MaybeUninit<u8>]) -> R) -> R {
    let mut buffer: Vec<MaybeUninit<u8>> = vec![MaybeUninit::uninit(); size];
    f(&mut buffer)
}

/// Same as [`with_alloca`] except the memory slice is zeroed before `f` is invoked.
pub fn with_alloca_zeroed<R>(size: usize, f: impl FnOnce(&mut [u8]) -> R) -> R {
    let mut buffer: Vec<u8> = vec![0u8; size];
    f(&mut buffer)
}

/// Allocates scratch memory sized and aligned for `T` and invokes `f` with a
/// `&mut MaybeUninit<T>` pointing into it, returning whatever `f` returns.
///
/// This reimplementation simply reserves a correctly-sized-and-aligned local `MaybeUninit<T>`
/// slot (Rust guarantees the alignment for us), rather than replicating upstream's manual pointer
/// alignment arithmetic over a raw byte buffer -- the observable contract (a `&mut MaybeUninit<T>`
/// valid only for the duration of the closure) is identical.
pub fn alloca<T, R>(f: impl FnOnce(&mut MaybeUninit<T>) -> R) -> R {
    let mut slot: MaybeUninit<T> = MaybeUninit::uninit();
    f(&mut slot)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_alloca_reports_requested_length() {
        let len = with_alloca(4096, |memory| memory.len());
        assert_eq!(len, 4096);
    }

    #[test]
    fn with_alloca_round_trips_writes() {
        let x = with_alloca(4096, |memory| {
            memory[0] = MaybeUninit::new(42);
            memory[1] = MaybeUninit::new(3);
            memory[3072] = MaybeUninit::new(4);
            unsafe {
                memory[0].assume_init() + memory[1].assume_init() + memory[3072].assume_init()
            }
        });
        assert_eq!(x, 42 + 3 + 4);
    }

    #[test]
    fn with_alloca_zeroed_is_zero_filled() {
        with_alloca_zeroed(256, |memory| {
            assert_eq!(memory.len(), 256);
            assert!(memory.iter().all(|&b| b == 0));
        });
    }

    #[test]
    fn alloca_is_aligned_for_t() {
        alloca::<u64, ()>(|slot| {
            let ptr = slot.as_mut_ptr();
            assert_eq!(ptr as usize % core::mem::align_of::<u64>(), 0);
            unsafe {
                ptr.write(0xdead_beef_u64);
                assert_eq!(ptr.read(), 0xdead_beef_u64);
            }
        });
    }
}
