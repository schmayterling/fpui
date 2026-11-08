//! The injectable agent dylib. The loader forces this library into the target
//! with `DYLD_INSERT_LIBRARIES`; the platform's loader then runs the constructor
//! below as the image loads, and it hands off to [`maycord_loader::agent::run`].
//!
//! The crate is intentionally tiny — only the entry point lives here. All of the
//! agent's logic is in `maycord_loader::agent`, where it is testable without
//! injecting into anything.

/// Runs as the agent dylib is loaded into the target. It runs the agent flow and
/// reports failure to stderr without taking the host process down — a mod that
/// cannot apply must never crash the client.
#[cfg(target_os = "macos")]
#[ctor::ctor]
fn maycord_init() {
    if let Err(error) = maycord_loader::agent::run() {
        eprintln!("maycord agent: {error}");
    }
}
