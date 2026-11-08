//! The injectable agent dylib. The loader forces this library into the target
//! with `DYLD_INSERT_LIBRARIES`; the OS then runs the constructor below as the
//! library loads, which hands off to [`maycord_loader::agent::run`].
//!
//! This crate is tiny on purpose — only the entry point is here. All the agent's
//! real work lives in `maycord_loader::agent`, where it can be tested without
//! injecting into anything.

/// Runs as the agent dylib loads into the target. It runs the agent and prints
/// any failure to stderr without bringing the process down — a mod that can't
/// apply must never crash the client.
#[cfg(target_os = "macos")]
#[ctor::ctor]
fn maycord_init() {
    if let Err(error) = maycord_loader::agent::run() {
        eprintln!("maycord agent: {error}");
    }
}
