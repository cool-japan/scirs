//! Crate-internal, std-only encoding helpers.
//!
//! These small vendored implementations replace the external `hex`,
//! `urlencoding`, and `data-encoding` crates for the limited functionality
//! this crate actually uses:
//!
//! - [`hex_encode`]: lowercase hexadecimal encoding of a byte slice
//! - [`percent_encode`] / [`percent_decode`]: RFC 3986 percent-encoding with
//!   the same semantics as the `urlencoding` crate (unreserved characters
//!   `[A-Za-z0-9-_.~]` are left as-is, everything else is percent-encoded
//!   byte-wise with uppercase hex digits)
//! - [`base64_encode`]: standard RFC 4648 base64 with padding

use crate::error::IoError;

const HEX_LOWER: &[u8; 16] = b"0123456789abcdef";
const HEX_UPPER: &[u8; 16] = b"0123456789ABCDEF";
const BASE64_STD: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Encode bytes as a lowercase hexadecimal string (drop-in for `hex::encode`).
pub(crate) fn hex_encode(bytes: impl AsRef<[u8]>) -> String {
    let bytes = bytes.as_ref();
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX_LOWER[(b >> 4) as usize] as char);
        out.push(HEX_LOWER[(b & 0x0f) as usize] as char);
    }
    out
}

/// `true` for RFC 3986 unreserved characters: `A-Z a-z 0-9 - _ . ~`.
fn is_unreserved(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~')
}

/// Percent-encode a string per RFC 3986 (drop-in for `urlencoding::encode`).
///
/// Unreserved characters `[A-Za-z0-9-_.~]` are passed through; every other
/// byte (including each byte of a multi-byte UTF-8 sequence) is encoded as
/// `%XX` with uppercase hex digits.
pub(crate) fn percent_encode(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for &b in input.as_bytes() {
        if is_unreserved(b) {
            out.push(b as char);
        } else {
            out.push('%');
            out.push(HEX_UPPER[(b >> 4) as usize] as char);
            out.push(HEX_UPPER[(b & 0x0f) as usize] as char);
        }
    }
    out
}

/// Decode a value of a single hex digit, or `None` if it is not one.
fn hex_digit_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

/// Decode a percent-encoded string (replacement for `urlencoding::decode`).
///
/// `%XX` sequences are decoded byte-wise; `+` is *not* treated as a space
/// (matching the `urlencoding` crate). Invalid or truncated `%` sequences and
/// decoded bytes that are not valid UTF-8 produce an [`IoError::ParseError`].
pub(crate) fn percent_decode(input: &str) -> Result<String, IoError> {
    let bytes = input.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let (hi, lo) = (bytes.get(i + 1).copied(), bytes.get(i + 2).copied());
            match (hi.and_then(hex_digit_value), lo.and_then(hex_digit_value)) {
                (Some(h), Some(l)) => {
                    out.push((h << 4) | l);
                    i += 3;
                }
                _ => {
                    return Err(IoError::ParseError(format!(
                        "invalid percent-encoding at byte {i} in {input:?}"
                    )));
                }
            }
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out)
        .map_err(|e| IoError::ParseError(format!("percent-decoded bytes are not UTF-8: {e}")))
}

/// Encode bytes as standard RFC 4648 base64 with `=` padding (drop-in for
/// `data_encoding::BASE64.encode`).
pub(crate) fn base64_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    let mut chunks = bytes.chunks_exact(3);
    for chunk in chunks.by_ref() {
        let n = ((chunk[0] as u32) << 16) | ((chunk[1] as u32) << 8) | (chunk[2] as u32);
        out.push(BASE64_STD[((n >> 18) & 0x3f) as usize] as char);
        out.push(BASE64_STD[((n >> 12) & 0x3f) as usize] as char);
        out.push(BASE64_STD[((n >> 6) & 0x3f) as usize] as char);
        out.push(BASE64_STD[(n & 0x3f) as usize] as char);
    }
    match chunks.remainder() {
        [a] => {
            let n = (*a as u32) << 16;
            out.push(BASE64_STD[((n >> 18) & 0x3f) as usize] as char);
            out.push(BASE64_STD[((n >> 12) & 0x3f) as usize] as char);
            out.push('=');
            out.push('=');
        }
        [a, b] => {
            let n = ((*a as u32) << 16) | ((*b as u32) << 8);
            out.push(BASE64_STD[((n >> 18) & 0x3f) as usize] as char);
            out.push(BASE64_STD[((n >> 12) & 0x3f) as usize] as char);
            out.push(BASE64_STD[((n >> 6) & 0x3f) as usize] as char);
            out.push('=');
        }
        _ => {}
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── hex ──────────────────────────────────────────────────────────────

    #[test]
    fn hex_encode_empty() {
        assert_eq!(hex_encode([]), "");
    }

    #[test]
    fn hex_encode_zero_and_ff() {
        assert_eq!(hex_encode([0x00]), "00");
        assert_eq!(hex_encode([0xff]), "ff");
        assert_eq!(hex_encode([0x00, 0xff, 0x10, 0xab]), "00ff10ab");
    }

    #[test]
    fn hex_encode_known_vector() {
        assert_eq!(hex_encode(b"foobar"), "666f6f626172");
    }

    // ── percent-encoding ─────────────────────────────────────────────────

    #[test]
    fn percent_encode_unreserved_passthrough() {
        let s = "AZaz09-_.~";
        assert_eq!(percent_encode(s), s);
    }

    #[test]
    fn percent_encode_space_and_plus() {
        // Matches urlencoding: space -> %20 (never '+'), '+' -> %2B.
        assert_eq!(percent_encode("a b+c"), "a%20b%2Bc");
    }

    #[test]
    fn percent_encode_unicode_multibyte() {
        assert_eq!(percent_encode("é"), "%C3%A9");
        assert_eq!(percent_encode("日本"), "%E6%97%A5%E6%9C%AC");
    }

    #[test]
    fn percent_decode_round_trip() -> Result<(), IoError> {
        for s in ["", "plain", "a b+c", "é日本", "100% sure?&=", "~-._"] {
            assert_eq!(percent_decode(&percent_encode(s))?, s);
        }
        Ok(())
    }

    #[test]
    fn percent_decode_plus_is_literal() -> Result<(), IoError> {
        assert_eq!(percent_decode("a+b")?, "a+b");
        Ok(())
    }

    #[test]
    fn percent_decode_invalid_sequences_error() {
        assert!(percent_decode("%").is_err());
        assert!(percent_decode("%1").is_err());
        assert!(percent_decode("%G1").is_err());
        assert!(percent_decode("abc%zz").is_err());
        // Lone 0xFF byte is not valid UTF-8.
        assert!(percent_decode("%FF").is_err());
    }

    // ── base64 ───────────────────────────────────────────────────────────

    #[test]
    fn base64_rfc4648_test_vectors() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn base64_binary_bytes() {
        assert_eq!(base64_encode(&[0x00, 0xff, 0x10]), "AP8Q");
        assert_eq!(base64_encode(&[0xfb, 0xff]), "+/8=");
    }
}
