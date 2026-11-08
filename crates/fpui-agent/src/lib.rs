//! The injectable agent dylib. The loader forces this library into the target
//! with `DYLD_INSERT_LIBRARIES`; the OS then runs the constructor below as the
//! library loads, which hands off to [`fpui_loader::agent::run`].
//!
//! This crate is tiny on purpose — only the entry point is here. All the agent's
//! real work lives in `fpui_loader::agent`, where it can be tested without
//! injecting into anything.

/// Runs as the agent dylib loads into the target. It runs the agent and prints
/// any failure to stderr without bringing the process down — a mod that can't
/// apply must never crash the client.
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[ctor::ctor]
fn fpui_init() {
    if let Err(error) = fpui_loader::agent::run() {
        eprintln!("fpui agent: {error}");
    }
}
