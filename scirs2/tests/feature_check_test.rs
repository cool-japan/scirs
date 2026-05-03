//! Integration test: verify that the feature-check script exists and is executable.
//!
//! This test acts as a sentinel ensuring the `scripts/check-features.sh`
//! script created for scirs2 TODO line 69 is always present in the repository.
//! The script itself is validated by CI; this test guards against accidental
//! deletion.
//!
//! Run:
//!   cargo nextest run -p scirs2 --test feature_check_test

use std::path::PathBuf;

/// Return the absolute path to `scripts/check-features.sh` relative to the
/// workspace root (two levels above this crate's manifest directory).
fn script_path() -> PathBuf {
    let manifest_dir =
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set by cargo");
    let crate_dir = PathBuf::from(manifest_dir);
    // crate_dir = <workspace>/scirs2/
    // workspace root = parent of crate_dir
    let workspace_root = crate_dir
        .parent()
        .expect("scirs2 crate must have a parent workspace directory");
    workspace_root.join("scripts").join("check-features.sh")
}

#[test]
fn feature_check_script_exists() {
    let path = script_path();
    assert!(path.exists(), "check-features.sh must exist at {:?}", path);
}

#[test]
fn feature_check_script_is_not_empty() {
    let path = script_path();
    if !path.exists() {
        return; // already caught by the existence test
    }
    let metadata = std::fs::metadata(&path).expect("could not stat check-features.sh");
    assert!(
        metadata.len() > 0,
        "check-features.sh must not be empty (found {} bytes)",
        metadata.len()
    );
}

#[test]
#[cfg(unix)]
fn feature_check_script_is_executable() {
    use std::os::unix::fs::PermissionsExt;

    let path = script_path();
    if !path.exists() {
        return; // already caught by the existence test
    }
    let perms = std::fs::metadata(&path)
        .expect("could not stat check-features.sh")
        .permissions();
    let mode = perms.mode();
    // Check owner execute bit (0o100)
    assert!(
        mode & 0o100 != 0,
        "check-features.sh must be executable (mode: {:o})",
        mode
    );
}
