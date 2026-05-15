//! # scirs2-symbolic-macros
//!
//! Proc-macro DSL helpers for the `scirs2-symbolic` crate.
//!
//! ## Macros
//!
//! * [`eml_pattern!`] — construct a `Pattern` from a concise DSL expression.
//! * [`eml_template!`] — same syntax, different label (marks the *right-hand side*
//!   of a rewrite rule to aid readability).
//!
//! Both macros emit fully-qualified `scirs2_symbolic::cas::pattern` construction
//! code so no `use` imports are required at the call site.
//!
//! ## Mini-DSL reference
//!
//! ```text
//! ?0, ?1, ?2          → PatVar(0), PatVar(1), PatVar(2)
//! var(0), var(1)      → PatGroundVar(0), PatGroundVar(1)
//! const(f)            → PatConst(f)          (f may be a float or integer literal)
//! int(n)              → PatConstInt(n)        (n is a u32 integer literal)
//! add(A, B)           → PatOp2(BinaryKind::Add, A, B)
//! sub(A, B)           → PatOp2(BinaryKind::Sub, A, B)
//! mul(A, B)           → PatOp2(BinaryKind::Mul, A, B)
//! div(A, B)           → PatOp2(BinaryKind::Div, A, B)
//! pow(A, B)           → PatOp2(BinaryKind::Pow, A, B)
//! neg(A)              → PatOp1(UnaryKind::Neg, A)
//! sin(A) … tanh(A)    → PatOp1(UnaryKind::Sin …)
//! exp(A), ln(A)       → PatOp1(UnaryKind::Exp / Ln)
//! sqrt(A), abs(A)     → PatOp1(UnaryKind::Sqrt / Abs)
//! arcsin … arctanh    → PatOp1(UnaryKind::Arcsin …)
//! ```

use proc_macro::TokenStream;
use proc_macro2::{Literal, Span, TokenStream as TokenStream2};
use quote::quote;
use syn::{
    ext::IdentExt,
    parse::{Parse, ParseStream, Result as SynResult},
    parse_macro_input, LitFloat, LitInt, Token,
};

// ---------------------------------------------------------------------------
// Module paths — single source of truth for the emitted fully-qualified names.
// ---------------------------------------------------------------------------
macro_rules! pat_path {
    () => {
        quote!(::scirs2_symbolic::cas::pattern)
    };
}

// ---------------------------------------------------------------------------
// PatternExpr — the parsed DSL node
// ---------------------------------------------------------------------------

/// A single DSL expression that emits `Pattern` construction code.
struct PatternExpr(TokenStream2);

impl Parse for PatternExpr {
    fn parse(input: ParseStream<'_>) -> SynResult<Self> {
        parse_pattern_expr(input).map(PatternExpr)
    }
}

/// Core recursive parser.  Uses `IdentExt::parse_any` so that Rust keywords
/// such as `const` are parsed as identifier-like tokens rather than causing
/// a hard parse error.
fn parse_pattern_expr(input: ParseStream<'_>) -> SynResult<TokenStream2> {
    // Peek: `?` → wildcard capture variable
    if input.peek(Token![?]) {
        return parse_patvar(input);
    }

    // Everything else starts with an identifier (possibly a keyword such as `const`).
    let name_ident = syn::Ident::parse_any(input)?;
    let name = name_ident.to_string();
    let span = name_ident.span();

    match name.as_str() {
        // ----------------------------------------------------------------
        // const(f) → PatConst(f as f64)
        // ----------------------------------------------------------------
        "const" => {
            let content;
            syn::parenthesized!(content in input);
            let ts = parse_const_arg(&content, span)?;
            Ok(ts)
        }

        // ----------------------------------------------------------------
        // int(n) → PatConstInt(n as u32)
        // ----------------------------------------------------------------
        "int" => {
            let content;
            syn::parenthesized!(content in input);
            let lit: LitInt = content.parse()?;
            let n: u32 = lit
                .base10_parse()
                .map_err(|e| syn::Error::new(lit.span(), format!("expected u32 for int(): {e}")))?;
            let lit_u32 = Literal::u32_suffixed(n);
            let p = pat_path!();
            Ok(quote! { #p::Pattern::PatConstInt(#lit_u32) })
        }

        // ----------------------------------------------------------------
        // var(n) → PatGroundVar(n as usize)
        // ----------------------------------------------------------------
        "var" => {
            let content;
            syn::parenthesized!(content in input);
            let lit: LitInt = content.parse()?;
            let n: usize = lit.base10_parse().map_err(|e| {
                syn::Error::new(lit.span(), format!("expected usize for var(): {e}"))
            })?;
            let lit_usize = Literal::usize_suffixed(n);
            let p = pat_path!();
            Ok(quote! { #p::Pattern::PatGroundVar(#lit_usize) })
        }

        // ----------------------------------------------------------------
        // Binary operators: add, sub, mul, div, pow
        // ----------------------------------------------------------------
        "add" | "sub" | "mul" | "div" | "pow" => {
            let kind_ts = binary_kind_tokens(&name, span)?;
            let content;
            syn::parenthesized!(content in input);
            let left = parse_pattern_expr(&content)?;
            content.parse::<Token![,]>()?;
            let right = parse_pattern_expr(&content)?;
            let p = pat_path!();
            Ok(quote! {
                #p::Pattern::PatOp2(
                    #kind_ts,
                    ::std::boxed::Box::new(#left),
                    ::std::boxed::Box::new(#right),
                )
            })
        }

        // ----------------------------------------------------------------
        // Unary operators: neg, sin, cos, tan, exp, ln, sqrt, abs,
        //                  sinh, cosh, tanh, arcsin, arccos, arctan,
        //                  arcsinh, arccosh, arctanh
        // ----------------------------------------------------------------
        "neg" | "sin" | "cos" | "tan" | "exp" | "ln" | "sqrt" | "abs" | "sinh" | "cosh"
        | "tanh" | "arcsin" | "arccos" | "arctan" | "arcsinh" | "arccosh" | "arctanh" => {
            let kind_ts = unary_kind_tokens(&name, span)?;
            let content;
            syn::parenthesized!(content in input);
            let child = parse_pattern_expr(&content)?;
            let p = pat_path!();
            Ok(quote! {
                #p::Pattern::PatOp1(
                    #kind_ts,
                    ::std::boxed::Box::new(#child),
                )
            })
        }

        other => Err(syn::Error::new(
            span,
            format!(
                "unknown pattern operator `{other}`; expected one of: \
                 const, int, var, add, sub, mul, div, pow, neg, sin, cos, tan, exp, ln, \
                 sqrt, abs, sinh, cosh, tanh, arcsin, arccos, arctan, arcsinh, arccosh, arctanh"
            ),
        )),
    }
}

// ---------------------------------------------------------------------------
// `?N` → PatVar(N)
// ---------------------------------------------------------------------------

fn parse_patvar(input: ParseStream<'_>) -> SynResult<TokenStream2> {
    let q_span = input.span();
    input.parse::<Token![?]>()?;
    let lit: LitInt = input
        .parse()
        .map_err(|_| syn::Error::new(q_span, "expected an integer after `?`, e.g. `?0`, `?1`"))?;
    let n: u32 = lit
        .base10_parse()
        .map_err(|e| syn::Error::new(lit.span(), format!("wildcard index must be u32: {e}")))?;
    let lit_u32 = Literal::u32_suffixed(n);
    let p = pat_path!();
    Ok(quote! { #p::Pattern::PatVar(#lit_u32) })
}

// ---------------------------------------------------------------------------
// `const(f)` — handles both float literals (2.0) and integer literals (0)
// ---------------------------------------------------------------------------

fn parse_const_arg(content: &syn::parse::ParseBuffer<'_>, span: Span) -> SynResult<TokenStream2> {
    let p = pat_path!();

    // Try float literal first.
    if content.peek(LitFloat) {
        let lit: LitFloat = content.parse()?;
        let v: f64 = lit
            .base10_parse()
            .map_err(|e| syn::Error::new(lit.span(), format!("expected f64 float literal: {e}")))?;
        let lit_f64 = Literal::f64_suffixed(v);
        return Ok(quote! { #p::Pattern::PatConst(#lit_f64) });
    }

    // Handle negative sign before a literal.
    if content.peek(Token![-]) {
        content.parse::<Token![-]>()?;
        if content.peek(LitFloat) {
            let lit: LitFloat = content.parse()?;
            let v: f64 = lit.base10_parse().map_err(|e| {
                syn::Error::new(lit.span(), format!("expected f64 float literal: {e}"))
            })?;
            let neg_lit = Literal::f64_suffixed(-v);
            return Ok(quote! { #p::Pattern::PatConst(#neg_lit) });
        }
        let lit: LitInt = content.parse()?;
        let v: u64 = lit
            .base10_parse()
            .map_err(|e| syn::Error::new(lit.span(), format!("expected integer literal: {e}")))?;
        let neg_lit = Literal::f64_suffixed(-(v as f64));
        return Ok(quote! { #p::Pattern::PatConst(#neg_lit) });
    }

    // Integer literal — convert to f64.
    if content.peek(LitInt) {
        let lit: LitInt = content.parse()?;
        let v: u64 = lit.base10_parse().map_err(|e| {
            syn::Error::new(
                lit.span(),
                format!("expected integer literal for const(): {e}"),
            )
        })?;
        let lit_f64 = Literal::f64_suffixed(v as f64);
        return Ok(quote! { #p::Pattern::PatConst(#lit_f64) });
    }

    Err(syn::Error::new(
        span,
        "expected a numeric literal for const() (e.g. `const(0)`, `const(2.0)`)",
    ))
}

// ---------------------------------------------------------------------------
// Helpers that emit BinaryKind / UnaryKind tokens
// ---------------------------------------------------------------------------

fn binary_kind_tokens(name: &str, span: Span) -> SynResult<TokenStream2> {
    let p = pat_path!();
    match name {
        "add" => Ok(quote! { #p::BinaryKind::Add }),
        "sub" => Ok(quote! { #p::BinaryKind::Sub }),
        "mul" => Ok(quote! { #p::BinaryKind::Mul }),
        "div" => Ok(quote! { #p::BinaryKind::Div }),
        "pow" => Ok(quote! { #p::BinaryKind::Pow }),
        _ => Err(syn::Error::new(
            span,
            format!("unknown binary operator: {name}"),
        )),
    }
}

fn unary_kind_tokens(name: &str, span: Span) -> SynResult<TokenStream2> {
    let p = pat_path!();
    match name {
        "neg" => Ok(quote! { #p::UnaryKind::Neg }),
        "exp" => Ok(quote! { #p::UnaryKind::Exp }),
        "ln" => Ok(quote! { #p::UnaryKind::Ln }),
        "sin" => Ok(quote! { #p::UnaryKind::Sin }),
        "cos" => Ok(quote! { #p::UnaryKind::Cos }),
        "tan" => Ok(quote! { #p::UnaryKind::Tan }),
        "sinh" => Ok(quote! { #p::UnaryKind::Sinh }),
        "cosh" => Ok(quote! { #p::UnaryKind::Cosh }),
        "tanh" => Ok(quote! { #p::UnaryKind::Tanh }),
        "arcsin" => Ok(quote! { #p::UnaryKind::Arcsin }),
        "arccos" => Ok(quote! { #p::UnaryKind::Arccos }),
        "arctan" => Ok(quote! { #p::UnaryKind::Arctan }),
        "arcsinh" => Ok(quote! { #p::UnaryKind::Arcsinh }),
        "arccosh" => Ok(quote! { #p::UnaryKind::Arccosh }),
        "arctanh" => Ok(quote! { #p::UnaryKind::Arctanh }),
        "sqrt" => Ok(quote! { #p::UnaryKind::Sqrt }),
        "abs" => Ok(quote! { #p::UnaryKind::Abs }),
        _ => Err(syn::Error::new(
            span,
            format!("unknown unary operator: {name}"),
        )),
    }
}

// ---------------------------------------------------------------------------
// Public proc-macros
// ---------------------------------------------------------------------------

/// Construct a `scirs2_symbolic::cas::pattern::Pattern` from the EML mini-DSL.
///
/// # Syntax
///
/// ```text
/// eml_pattern!( <expr> )
/// ```
///
/// where `<expr>` is one of:
///
/// | Token | Expansion |
/// |-------|-----------|
/// | `?N` | `Pattern::PatVar(N)` |
/// | `var(N)` | `Pattern::PatGroundVar(N)` |
/// | `const(f)` | `Pattern::PatConst(f)` |
/// | `int(n)` | `Pattern::PatConstInt(n)` |
/// | `add(A,B)` `sub(A,B)` `mul(A,B)` `div(A,B)` `pow(A,B)` | `Pattern::PatOp2(…, A, B)` |
/// | `neg(A)` `sin(A)` `cos(A)` … | `Pattern::PatOp1(…, A)` |
///
/// # Example
///
/// ```rust,ignore
/// use scirs2_symbolic::eml_pattern;
///
/// let pat = eml_pattern!(add(?0, const(0)));
/// ```
#[proc_macro]
pub fn eml_pattern(input: TokenStream) -> TokenStream {
    let expr = parse_macro_input!(input as PatternExpr);
    expr.0.into()
}

/// Construct a `scirs2_symbolic::cas::pattern::Pattern` from the EML mini-DSL.
///
/// Identical to [`eml_pattern!`] — the different name labels the *right-hand side*
/// (template / replacement) of a rewrite rule for readability.
///
/// # Example
///
/// ```rust,ignore
/// use scirs2_symbolic::{eml_pattern, eml_template};
///
/// let lhs = eml_pattern!(add(?0, ?1));
/// let rhs = eml_template!(add(?1, ?0));  // commutativity rewrite
/// ```
#[proc_macro]
pub fn eml_template(input: TokenStream) -> TokenStream {
    let expr = parse_macro_input!(input as PatternExpr);
    expr.0.into()
}
