//! Process-default rustls `CryptoProvider` bootstrap (pure-Rust OxiTLS).
//!
//! The workspace `reqwest` dependency is built with `rustls-no-provider` (no aws-lc-rs/ring
//! bundled -- see the "PURE RUST BLOCKER (reqwest)" note in the workspace root `Cargo.toml`),
//! so constructing a `reqwest::Client` panics with "No rustls crypto provider is configured"
//! unless a process-default provider exists. The workspace `ureq` dependency likewise uses
//! `rustls-no-provider` and panics ("No CryptoProvider for Rustls") when an HTTPS request is
//! made without one. [`ensure_default_tls_provider`] is called immediately before every eager
//! reqwest client construction in this crate ([`crate::external::ExternalClient::with_config`]
//! and [`crate::cache::download_data`]) and before every ureq HTTPS request
//! ([`crate::cache::fetch_data`] and the `download-sync` fallback in `crate::external`) so
//! that networking works out of the box, while an application that prefers a different
//! provider can still install its own before first use of SciRS2 networking.

/// Install the pure-Rust OxiTLS CryptoProvider (oxitls-rustcrypto-provider)
/// as the process-default rustls provider if the application has not
/// already installed one. Applications that want a different provider
/// simply install it before first use of SciRS2 networking.
pub(crate) fn ensure_default_tls_provider() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if rustls::crypto::CryptoProvider::get_default().is_none() {
            // A racing install elsewhere is fine -- ignore the Err.
            let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
        }
    });
}
