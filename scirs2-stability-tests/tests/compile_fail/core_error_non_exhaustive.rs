// Compile-fail test for the `#[non_exhaustive]` contract on
// `scirs2_core::error::CoreError`.
//
// `CoreError` is declared `#[non_exhaustive]`, so a downstream crate (this one)
// must include a wildcard `_` arm even after matching every *named* variant:
// the attribute reserves a hidden "future variant" that downstream code can
// never name.
//
// This test deliberately matches ALL named variants and OMITS the `_` arm.
// Because every named variant is covered, the only pattern still uncovered is
// the hidden one that `#[non_exhaustive]` forces, so compilation must fail with
// `error[E0004]: non-exhaustive patterns: \`_\` not covered`.
//
// Covering every named variant is what isolates the contract under test: if
// someone removes `#[non_exhaustive]` from `CoreError`, this match becomes
// exhaustive and COMPILES, which makes trybuild's compile-fail expectation fail
// and surfaces the silent API regression.  (A previous version matched only two
// variants, so it kept "failing to compile" for the unrelated reason that the
// other named variants were unmatched — and silently stopped testing the
// attribute at all once the enum grew.  Matching every variant prevents that
// masking.)
use scirs2_core::error::CoreError;

fn check_non_exhaustive(e: CoreError) {
    // Every named variant is covered; there is intentionally NO `_` arm, so the
    // match is non-exhaustive solely because of `#[non_exhaustive]`.
    match e {
        CoreError::ComputationError(_) => {}
        CoreError::DomainError(_) => {}
        CoreError::DispatchError(_) => {}
        CoreError::ConvergenceError(_) => {}
        CoreError::DimensionError(_) => {}
        CoreError::ShapeError(_) => {}
        CoreError::IndexError(_) => {}
        CoreError::ValueError(_) => {}
        CoreError::TypeError(_) => {}
        CoreError::NotImplementedError(_) => {}
        CoreError::ImplementationError(_) => {}
        CoreError::MemoryError(_) => {}
        CoreError::AllocationError(_) => {}
        CoreError::ConfigError(_) => {}
        CoreError::InvalidArgument(_) => {}
        CoreError::InvalidInput(_) => {}
        CoreError::PermissionError(_) => {}
        CoreError::ValidationError(_) => {}
        CoreError::InvalidState(_) => {}
        CoreError::JITError(_) => {}
        CoreError::JSONError(_) => {}
        CoreError::IoError(_) => {}
        CoreError::SchedulerError(_) => {}
        CoreError::TimeoutError(_) => {}
        CoreError::CompressionError(_) => {}
        CoreError::InvalidShape(_) => {}
        CoreError::DeviceError(_) => {}
        CoreError::MutexError(_) => {}
        CoreError::ThreadError(_) => {}
        CoreError::StreamError(_) => {}
        CoreError::EndOfStream(_) => {}
        CoreError::ResourceError(_) => {}
        CoreError::CommunicationError(_) => {}
        CoreError::SecurityError(_) => {}
    }
}

fn main() {}
