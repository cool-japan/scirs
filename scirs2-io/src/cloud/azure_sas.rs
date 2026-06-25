//! Azure Blob Storage SAS (Shared Access Signature) token generation and
//! validation (simulation mode).
//!
//! In production Azure Blob Storage, a SAS token is produced by building a
//! **string-to-sign** from the storage account name, container, blob, expiry,
//! permissions, and other parameters, then signing it with the storage account
//! key using HMAC-SHA256.  The resulting URL looks like:
//!
//! ```text
//! https://{account}.blob.core.windows.net/{container}/{blob}?{sas_token}
//! ```
//!
//! # Simulation mode
//!
//! This module provides a **deterministic mock** instead of performing real
//! HMAC-SHA256 signing:
//!
//! - The "signature" is computed as an XOR-folded hash of the string-to-sign
//!   and the storage account key.  This is **not cryptographically secure** and
//!   must **never** be used in production.
//! - For production use, replace `mock_sign` with an HMAC-SHA256 call using
//!   the `sha2` and `digest` crates (both already present in the workspace).
//!
//! # Example
//!
//! ```rust
//! use scirs2_io::cloud::azure_sas::{AzureSasParams, SasPermissions, SasResource,
//!     generate_sas_token, build_sas_url, is_sas_valid, parse_sas_token};
//!
//! let params = AzureSasParams {
//!     account_name: "mystorageaccount".into(),
//!     container: "data".into(),
//!     blob: Some("file.bin".into()),
//!     permissions: SasPermissions::read_only(),
//!     expiry: 9_999_999_999,
//!     start: None,
//!     resource: SasResource::Blob,
//! };
//! let key = b"not-a-real-key";
//! let token = generate_sas_token(&params, key);
//! assert!(token.contains("sv=") && token.contains("sp=") && token.contains("se="));
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// SasPermissions
// ---------------------------------------------------------------------------

/// Bit-flag set of Azure SAS permissions.
///
/// Maps to the Azure `sp` query parameter.  Each flag corresponds to one
/// character in the Azure permission string:
///
/// | Character | Flag    |
/// |-----------|---------|
/// | `r`       | read    |
/// | `w`       | write   |
/// | `d`       | delete  |
/// | `l`       | list    |
/// | `a`       | add     |
/// | `c`       | create  |
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SasPermissions {
    /// Permission to read blobs.
    pub read: bool,
    /// Permission to write / overwrite blobs.
    pub write: bool,
    /// Permission to delete blobs.
    pub delete: bool,
    /// Permission to list blobs within a container.
    pub list: bool,
    /// Permission to append data to a blob.
    pub add: bool,
    /// Permission to create a new blob.
    pub create: bool,
}

impl SasPermissions {
    /// Create a read-only permission set (`r`).
    pub fn read_only() -> Self {
        Self {
            read: true,
            write: false,
            delete: false,
            list: false,
            add: false,
            create: false,
        }
    }

    /// Create a read-write permission set (`rw`).
    pub fn read_write() -> Self {
        Self {
            read: true,
            write: true,
            delete: false,
            list: false,
            add: false,
            create: false,
        }
    }

    /// Create a full-access permission set (`rwdlac`).
    pub fn full() -> Self {
        Self {
            read: true,
            write: true,
            delete: true,
            list: true,
            add: true,
            create: true,
        }
    }

    /// Encode the permission set as the Azure `sp` query-parameter value.
    ///
    /// The characters are emitted in the canonical order `r w d l a c`.
    pub fn as_permission_string(&self) -> String {
        let mut s = String::with_capacity(6);
        if self.read {
            s.push('r');
        }
        if self.write {
            s.push('w');
        }
        if self.delete {
            s.push('d');
        }
        if self.list {
            s.push('l');
        }
        if self.add {
            s.push('a');
        }
        if self.create {
            s.push('c');
        }
        s
    }

    /// Decode an Azure permission string back to a `SasPermissions` value.
    pub fn from_permission_string(s: &str) -> Self {
        Self {
            read: s.contains('r'),
            write: s.contains('w'),
            delete: s.contains('d'),
            list: s.contains('l'),
            add: s.contains('a'),
            create: s.contains('c'),
        }
    }
}

// ---------------------------------------------------------------------------
// SasResource
// ---------------------------------------------------------------------------

/// The type of Azure storage resource targeted by a SAS token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SasResource {
    /// A single blob (`ss=b`).
    Blob,
    /// A container (`ss=c`).
    Container,
    /// A queue (`ss=q`).
    Queue,
    /// A table (`ss=t`).
    Table,
}

impl SasResource {
    /// Return the single-character code used in the `ss` query parameter.
    pub fn as_code(&self) -> &'static str {
        match self {
            SasResource::Blob => "b",
            SasResource::Container => "c",
            SasResource::Queue => "q",
            SasResource::Table => "t",
        }
    }
}

// ---------------------------------------------------------------------------
// AzureSasParams
// ---------------------------------------------------------------------------

/// Parameters used to generate an Azure SAS token.
#[derive(Debug, Clone)]
pub struct AzureSasParams {
    /// Storage account name (e.g. `"mystorageaccount"`).
    pub account_name: String,
    /// Container name.
    pub container: String,
    /// Blob name (None for container-level SAS).
    pub blob: Option<String>,
    /// Permission set.
    pub permissions: SasPermissions,
    /// Token expiry as a Unix timestamp (seconds since epoch).
    pub expiry: u64,
    /// Optional token start time as a Unix timestamp.
    pub start: Option<u64>,
    /// The resource type targeted by this SAS.
    pub resource: SasResource,
}

// ---------------------------------------------------------------------------
// AzureError
// ---------------------------------------------------------------------------

/// Errors returned by Azure SAS operations.
#[derive(Debug, thiserror::Error)]
pub enum AzureError {
    /// Failed to parse a SAS token query string.
    #[error("parse error: {0}")]
    Parse(String),
    /// A required field is missing from the SAS token.
    #[error("missing field: {0}")]
    MissingField(String),
    /// The SAS token has passed its expiry time.
    #[error("expired token")]
    Expired,
}

// ---------------------------------------------------------------------------
// Token generation
// ---------------------------------------------------------------------------

/// SAS service version string embedded in every token.
const SAS_VERSION: &str = "2021-06-08";

/// Generate a SAS token query string.
///
/// # Parameters
///
/// - `params` — token parameters.
/// - `account_key` — raw storage account key bytes.
///
/// # Implementation note (simulation)
///
/// The `sig` field is computed by `mock_sign` — an XOR-folded deterministic
/// hash.  **Do not use this in production.**  For real signing replace the
/// `mock_sign` call with HMAC-SHA256 over `string_to_sign` using the decoded
/// base-64 account key (see `hmac` + `sha2` crates).
///
/// # Returns
///
/// A percent-encoded query string, e.g.:
/// `sv=2021-06-08&ss=b&srt=o&sp=r&se=2026-12-31T00%3A00%3A00Z&sig=…`
pub fn generate_sas_token(params: &AzureSasParams, account_key: &[u8]) -> String {
    let expiry_str = unix_to_iso8601(params.expiry);
    let start_str = params.start.map(unix_to_iso8601);
    let perm_str = params.permissions.as_permission_string();
    let resource_code = params.resource.as_code();

    // Build the canonical string-to-sign (simplified vs. real Azure spec,
    // which includes many more newline-separated fields).
    let signed_resource = match &params.blob {
        Some(blob) => format!("{}/{}/{}", params.account_name, params.container, blob),
        None => format!("{}/{}", params.account_name, params.container),
    };

    let string_to_sign = format!(
        "{account}\n{permissions}\n{expiry}\n{resource}\n{version}\n{resource_code}",
        account = params.account_name,
        permissions = perm_str,
        expiry = expiry_str,
        resource = signed_resource,
        version = SAS_VERSION,
        resource_code = resource_code,
    );

    let sig_bytes = mock_sign(string_to_sign.as_bytes(), account_key);
    let sig_hex = crate::encoding_utils::hex_encode(sig_bytes);

    // Assemble the query string.
    let mut parts: Vec<String> = Vec::new();
    parts.push(format!(
        "sv={}",
        crate::encoding_utils::percent_encode(SAS_VERSION)
    ));
    parts.push(format!("ss={}", resource_code));
    parts.push("srt=o".to_owned()); // resource type: object
    parts.push(format!(
        "sp={}",
        crate::encoding_utils::percent_encode(&perm_str)
    ));
    if let Some(ref s) = start_str {
        parts.push(format!("st={}", crate::encoding_utils::percent_encode(s)));
    }
    parts.push(format!(
        "se={}",
        crate::encoding_utils::percent_encode(&expiry_str)
    ));
    parts.push(format!(
        "sig={}",
        crate::encoding_utils::percent_encode(&sig_hex)
    ));

    parts.join("&")
}

/// Build a complete Azure Blob Storage URL with an embedded SAS token.
///
/// Format: `https://{account}.blob.core.windows.net/{container}/{blob}?{token}`
///
/// If `params.blob` is `None` the URL omits the blob path component.
pub fn build_sas_url(params: &AzureSasParams, account_key: &[u8]) -> String {
    let token = generate_sas_token(params, account_key);
    let blob_path = match &params.blob {
        Some(b) => format!("/{}", crate::encoding_utils::percent_encode(b)),
        None => String::new(),
    };
    format!(
        "https://{}.blob.core.windows.net/{}{blob_path}?{token}",
        params.account_name,
        crate::encoding_utils::percent_encode(&params.container),
    )
}

/// Parse a SAS token query string into a key-value map.
///
/// Handles both `+`-encoded and `%xx`-encoded values.
///
/// # Errors
///
/// Returns [`AzureError::Parse`] if the string is not valid `key=value` pairs.
pub fn parse_sas_token(token: &str) -> Result<HashMap<String, String>, AzureError> {
    let mut map = HashMap::new();
    for part in token.split('&') {
        if part.is_empty() {
            continue;
        }
        let eq_pos = part
            .find('=')
            .ok_or_else(|| AzureError::Parse(format!("missing '=' in token segment: {part}")))?;
        let key = &part[..eq_pos];
        let raw_value = &part[eq_pos + 1..];
        let value = crate::encoding_utils::percent_decode(raw_value)
            .unwrap_or_else(|_| raw_value.to_owned());
        map.insert(key.to_owned(), value);
    }
    Ok(map)
}

/// Validate that a parsed SAS token has not expired.
///
/// `current_time` is compared against the `se` (signed expiry) field which is
/// stored as an ISO-8601 UTC timestamp.
///
/// Returns `true` if the token is valid (not yet expired), `false` otherwise.
pub fn is_sas_valid(token_params: &HashMap<String, String>, current_time: u64) -> bool {
    match token_params.get("se") {
        Some(se) => {
            let expiry = iso8601_to_unix(se).unwrap_or(0);
            current_time < expiry
        }
        None => false,
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Convert a Unix timestamp to a truncated ISO-8601 UTC string.
///
/// Produces the format `YYYY-MM-DDTHH:MM:SSZ` that Azure expects in `se`/`st`.
fn unix_to_iso8601(ts: u64) -> String {
    // Hand-roll conversion to avoid pulling in chrono here (it is available
    // workspace-wide but we keep this module self-contained for clarity).
    let secs = ts;
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hh = time_of_day / 3600;
    let mm = (time_of_day % 3600) / 60;
    let ss = time_of_day % 60;

    // Gregorian calendar computation.
    let (year, month, day) = days_to_ymd(days);
    format!("{year:04}-{month:02}-{day:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

/// Parse an ISO-8601 UTC string back to a Unix timestamp.
///
/// Accepts `YYYY-MM-DDTHH:MM:SSZ` format.  Returns `None` on parse failure.
fn iso8601_to_unix(s: &str) -> Option<u64> {
    // Expected: "YYYY-MM-DDTHH:MM:SSZ"  (20 chars)
    if s.len() < 19 {
        return None;
    }
    let year: u64 = s[0..4].parse().ok()?;
    let month: u64 = s[5..7].parse().ok()?;
    let day: u64 = s[8..10].parse().ok()?;
    let hh: u64 = s[11..13].parse().ok()?;
    let mm: u64 = s[14..16].parse().ok()?;
    let ss: u64 = s[17..19].parse().ok()?;

    let days = ymd_to_days(year, month, day);
    Some(days * 86400 + hh * 3600 + mm * 60 + ss)
}

/// Convert an epoch day count to `(year, month, day)`.
fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    // Proleptic Gregorian calendar via the civil date algorithm.
    // Reference: http://howardhinnant.github.io/date_algorithms.html
    let z = days + 719468;
    let era = z / 146097;
    let doe = z % 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

/// Convert `(year, month, day)` to an epoch day count.
fn ymd_to_days(y: u64, m: u64, d: u64) -> u64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y / 400;
    let yoe = y % 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// Deterministic mock signature (XOR-based; NOT cryptographically secure).
///
/// Folds the `data` bytes XOR'd with cycling `key` bytes into a 32-byte array.
/// This provides a stable, key-dependent output that is reproducible across
/// runs — useful for testing token round-trips — but offers no security
/// guarantees.
///
/// **Production replacement**: use HMAC-SHA256 from the `hmac` + `sha2` crates,
/// signing `string_to_sign` with the base-64-decoded storage account key.
fn mock_sign(data: &[u8], key: &[u8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    if key.is_empty() {
        // No key: just XOR data into the output buffer.
        for (i, &b) in data.iter().enumerate() {
            out[i % 32] ^= b;
        }
        return out;
    }
    for (i, &b) in data.iter().enumerate() {
        let k = key[i % key.len()];
        out[i % 32] ^= b ^ k;
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// SasPermissions::read_only() encodes to 'r' but not 'w'.
    #[test]
    fn test_read_only_permission_string() {
        let perm = SasPermissions::read_only();
        let s = perm.as_permission_string();
        assert!(s.contains('r'), "read_only must contain 'r'");
        assert!(!s.contains('w'), "read_only must not contain 'w'");
        assert!(!s.contains('d'), "read_only must not contain 'd'");
        assert!(!s.contains('l'), "read_only must not contain 'l'");
    }

    /// SasPermissions::full() encodes all six characters.
    #[test]
    fn test_full_permission_string() {
        let s = SasPermissions::full().as_permission_string();
        for ch in ['r', 'w', 'd', 'l', 'a', 'c'] {
            assert!(s.contains(ch), "full must contain '{ch}'");
        }
    }

    /// generate_sas_token output contains "sv=", "sp=", and "se=".
    #[test]
    fn test_generate_sas_token_contains_required_fields() {
        let params = AzureSasParams {
            account_name: "account".into(),
            container: "container".into(),
            blob: Some("file.bin".into()),
            permissions: SasPermissions::read_write(),
            expiry: 9_999_999_999,
            start: None,
            resource: SasResource::Blob,
        };
        let token = generate_sas_token(&params, b"fake-key");
        assert!(token.contains("sv="), "token must contain sv=");
        assert!(token.contains("sp="), "token must contain sp=");
        assert!(token.contains("se="), "token must contain se=");
        assert!(token.contains("sig="), "token must contain sig=");
    }

    /// parse_sas_token round-trips the key-value pairs.
    #[test]
    fn test_parse_sas_token_round_trip() {
        let params = AzureSasParams {
            account_name: "roundtrip".into(),
            container: "c".into(),
            blob: None,
            permissions: SasPermissions::full(),
            expiry: 9_000_000_000,
            start: None,
            resource: SasResource::Container,
        };
        let token = generate_sas_token(&params, b"key");
        let map = parse_sas_token(&token).expect("parse");

        // sv field must be the service version constant.
        assert_eq!(map.get("sv").map(|s| s.as_str()), Some(SAS_VERSION));

        // sp field must decode to the full permission string.
        let perm_encoded = SasPermissions::full().as_permission_string();
        assert_eq!(
            map.get("sp").map(|s| s.as_str()),
            Some(perm_encoded.as_str())
        );
    }

    /// is_sas_valid returns true for future expiry, false for past.
    #[test]
    fn test_is_sas_valid_expiry() {
        let future_params = AzureSasParams {
            account_name: "a".into(),
            container: "c".into(),
            blob: None,
            permissions: SasPermissions::read_only(),
            expiry: 9_999_999_999,
            start: None,
            resource: SasResource::Blob,
        };
        let future_token = generate_sas_token(&future_params, b"k");
        let future_map = parse_sas_token(&future_token).expect("parse future");

        let past_params = AzureSasParams {
            account_name: "a".into(),
            container: "c".into(),
            blob: None,
            permissions: SasPermissions::read_only(),
            expiry: 1_000_000, // year ~1970
            start: None,
            resource: SasResource::Blob,
        };
        let past_token = generate_sas_token(&past_params, b"k");
        let past_map = parse_sas_token(&past_token).expect("parse past");

        // Current simulation time: somewhere in 2026 (about 1_744_000_000 s).
        let now: u64 = 1_744_000_000;

        assert!(
            is_sas_valid(&future_map, now),
            "future token should be valid"
        );
        assert!(
            !is_sas_valid(&past_map, now),
            "past token should be expired"
        );
    }

    /// Missing 'se' field means the token is invalid.
    #[test]
    fn test_is_sas_valid_missing_se() {
        let map: HashMap<String, String> = [("sv".into(), SAS_VERSION.into())].into();
        assert!(!is_sas_valid(&map, 1_000_000));
    }

    /// from_permission_string round-trips through as_permission_string.
    #[test]
    fn test_permission_round_trip() {
        for perm in [
            SasPermissions::read_only(),
            SasPermissions::read_write(),
            SasPermissions::full(),
        ] {
            let s = perm.as_permission_string();
            let decoded = SasPermissions::from_permission_string(&s);
            assert_eq!(decoded, perm, "round-trip failed for '{s}'");
        }
    }

    /// build_sas_url produces a URL that starts with the account endpoint.
    #[test]
    fn test_build_sas_url_format() {
        let params = AzureSasParams {
            account_name: "myaccount".into(),
            container: "mycontainer".into(),
            blob: Some("blob.bin".into()),
            permissions: SasPermissions::read_only(),
            expiry: 9_999_999_999,
            start: None,
            resource: SasResource::Blob,
        };
        let url = build_sas_url(&params, b"key");
        assert!(url.starts_with("https://myaccount.blob.core.windows.net/"));
        assert!(url.contains('?'), "URL must contain a query string");
        assert!(url.contains("sig="), "URL query must contain sig=");
    }

    /// Different account keys produce different signatures (mock_sign is key-sensitive).
    #[test]
    fn test_different_keys_produce_different_signatures() {
        let params = AzureSasParams {
            account_name: "a".into(),
            container: "c".into(),
            blob: None,
            permissions: SasPermissions::read_only(),
            expiry: 9_000_000_000,
            start: None,
            resource: SasResource::Blob,
        };
        let t1 = generate_sas_token(&params, b"key-one");
        let t2 = generate_sas_token(&params, b"key-two");
        // The sig= fields should differ.
        let m1 = parse_sas_token(&t1).expect("parse 1");
        let m2 = parse_sas_token(&t2).expect("parse 2");
        assert_ne!(
            m1.get("sig"),
            m2.get("sig"),
            "different keys must produce different signatures"
        );
    }
}
