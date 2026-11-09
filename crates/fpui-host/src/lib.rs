//! The host side of fpui: it loads plugins and runs them through capability-
//! gated hooks.
//!
//! A plugin never touches the client directly. It implements [`Plugin`], and the
//! host only invokes a hook the plugin was *granted* the matching [`Capability`]
//! for. fpui is general: it knows nothing about what a hook *means*. A hook is
//! just a name — `"message"`, `"draft"`, … defined by the application layer —
//! carrying an opaque byte payload. There are two kinds:
//!
//! * **transform** — [`Plugin::transform`] gets a payload and may return a
//!   rewritten one; the host chains transforms and uses the final bytes.
//! * **observe** — [`Plugin::observe`] gets a payload it can react to but not
//!   change.
//!
//! The host is the only place a capability is checked: if a plugin wasn't
//! granted one, the host simply never invokes a hook under it. The application
//! (e.g. maycord) owns the vocabulary — the capability names, the hook names,
//! and the payload shapes; fpui transports bytes and enforces the grant. This
//! crate is a plain, safe *reference* host. A sandboxed WebAssembly host
//! enforces the same boundary a different way but implements the same [`Plugin`]
//! trait, so hooks work the same whether a plugin is native or wasm.
//!
//! What a plugin is *granted* is a subset of what its manifest *asks for*: the
//! manifest is the request, and [`Host::register`] takes the capabilities the
//! user actually approved. Granting a capability the manifest never asked for is
//! rejected, so approval can only shrink the request, never grow it.
//!
//! ```
//! use fpui_host::{Host, Plugin};
//! use fpui_manifest::{Capability, ExtensionKind, Manifest, API_VERSION};
//!
//! // A plugin that redacts a word from any text payload it is shown.
//! struct Redact;
//! impl Plugin for Redact {
//!     fn transform(&mut self, _hook: &str, payload: &[u8]) -> Option<Vec<u8>> {
//!         let text = String::from_utf8_lossy(payload);
//!         Some(text.replace("secret", "[redacted]").into_bytes())
//!     }
//! }
//!
//! let manifest = Manifest {
//!     api_version: API_VERSION,
//!     id: "fpui-redact".into(), name: "Redact".into(), version: "1.0.0".into(),
//!     author: "fpui".into(), license: "MIT".into(),
//!     source: "https://example.invalid/redact".into(),
//!     kind: ExtensionKind::Plugin,
//!     capabilities: vec![Capability::new("message-content")],
//! };
//!
//! let cap = Capability::new("message-content");
//! let mut host = Host::new();
//! host.register(&manifest, &[cap.clone()], Box::new(Redact)).unwrap();
//! let out = host.transform(&cap, "message", b"the secret is out");
//! assert_eq!(String::from_utf8(out).unwrap(), "the [redacted] is out");
//! ```

use std::collections::BTreeSet;

use fpui_manifest::{Capability, ExtensionKind, Manifest, ManifestError};

/// What a plugin does. Both methods do nothing by default, so a plugin only
/// implements the hooks it cares about. The host never invokes a hook the plugin
/// wasn't granted, so implementing one it lacks the capability for is harmless.
///
/// A `hook` is an application-defined name; `payload` is opaque bytes whose shape
/// the application defines. fpui neither parses nor interprets either — that is
/// the application's job (maycord serializes a Discord message to the payload,
/// another app serializes its own thing).
pub trait Plugin {
    /// Transform a payload for a named hook, returning the rewritten bytes, or
    /// `None` to leave it unchanged.
    fn transform(&mut self, hook: &str, payload: &[u8]) -> Option<Vec<u8>> {
        let _ = (hook, payload);
        None
    }

    /// Observe a payload for a named hook. The plugin can react but not change
    /// anything the host uses.
    fn observe(&mut self, hook: &str, payload: &[u8]) {
        let _ = (hook, payload);
    }
}

/// Why [`Host::register`] rejected a plugin.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HostError {
    /// The manifest failed [`Manifest::validate`].
    Manifest(ManifestError),
    /// The manifest was a theme; only plugins are registered with a host.
    NotAPlugin,
    /// A capability was granted that the manifest never asked for. Approval can
    /// only shrink the request, so this is always a caller bug.
    UngrantedCapability(Capability),
    /// Another registered plugin already uses this id.
    DuplicateId(String),
}

impl core::fmt::Display for HostError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Manifest(e) => write!(f, "{e}"),
            Self::NotAPlugin => write!(f, "only a plugin may be registered with a host"),
            Self::UngrantedCapability(cap) => {
                write!(
                    f,
                    "granted capability `{cap}` was not requested by the manifest"
                )
            }
            Self::DuplicateId(id) => write!(f, "a plugin with id `{id}` is already registered"),
        }
    }
}

impl std::error::Error for HostError {}

/// One loaded plugin and the capabilities its user granted it.
struct Registered {
    id: String,
    granted: BTreeSet<Capability>,
    plugin: Box<dyn Plugin>,
}

/// Loads plugins and runs them through the hooks, making sure each plugin is
/// only reached through a hook it was granted.
///
/// Hooks run in the order plugins were registered, so a chain of transforms is
/// predictable: the first plugin sees the original payload, the next sees the
/// first's output, and so on.
#[derive(Default)]
pub struct Host {
    plugins: Vec<Registered>,
}

impl Host {
    /// An empty host with no plugins.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a plugin. `granted` is the set of capabilities the user approved;
    /// it must be a subset of `manifest.capabilities`. The manifest must pass
    /// [`Manifest::validate`] and must be a [`plugin`](ExtensionKind::Plugin).
    pub fn register(
        &mut self,
        manifest: &Manifest,
        granted: &[Capability],
        plugin: Box<dyn Plugin>,
    ) -> Result<(), HostError> {
        manifest.validate().map_err(HostError::Manifest)?;
        if manifest.kind != ExtensionKind::Plugin {
            return Err(HostError::NotAPlugin);
        }
        for cap in granted {
            if !manifest.capabilities.contains(cap) {
                return Err(HostError::UngrantedCapability(cap.clone()));
            }
        }
        if self.plugins.iter().any(|p| p.id == manifest.id) {
            return Err(HostError::DuplicateId(manifest.id.clone()));
        }
        self.plugins.push(Registered {
            id: manifest.id.clone(),
            granted: granted.iter().cloned().collect(),
            plugin,
        });
        Ok(())
    }

    /// How many plugins are registered.
    pub fn len(&self) -> usize {
        self.plugins.len()
    }

    /// Whether no plugins are registered.
    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }

    /// Run every plugin granted `cap` through the `hook` transform, in
    /// registration order, chaining each one's output into the next, and return
    /// the final payload. A plugin that returns `None` leaves the running payload
    /// unchanged — a failed or uninterested mod changes nothing.
    pub fn transform(&mut self, cap: &Capability, hook: &str, input: &[u8]) -> Vec<u8> {
        let mut payload = input.to_vec();
        self.each_granted(cap, |p| {
            if let Some(out) = p.transform(hook, &payload) {
                payload = out;
            }
        });
        payload
    }

    /// Hand `payload` to every plugin granted `cap` as an observed `hook`, in
    /// registration order. Observers can't change what the host uses.
    pub fn observe(&mut self, cap: &Capability, hook: &str, payload: &[u8]) {
        self.each_granted(cap, |p| p.observe(hook, payload));
    }

    /// The single place a capability is checked: run `f` on every plugin that
    /// holds `cap`, in registration order, and no others.
    fn each_granted(&mut self, cap: &Capability, mut f: impl FnMut(&mut dyn Plugin)) {
        for reg in &mut self.plugins {
            if reg.granted.contains(cap) {
                f(reg.plugin.as_mut());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest(id: &str, caps: &[Capability]) -> Manifest {
        Manifest {
            api_version: fpui_manifest::API_VERSION,
            id: id.into(),
            name: id.into(),
            version: "1.0.0".into(),
            author: "fpui".into(),
            license: "MIT".into(),
            source: "https://example.invalid/p".into(),
            kind: ExtensionKind::Plugin,
            capabilities: caps.to_vec(),
        }
    }

    fn content() -> Capability {
        Capability::new("content")
    }
    fn events() -> Capability {
        Capability::new("events")
    }

    /// Uppercases a transform payload and records every payload it observes.
    #[derive(Default)]
    struct Spy {
        seen: std::rc::Rc<std::cell::RefCell<Vec<String>>>,
    }
    impl Plugin for Spy {
        fn transform(&mut self, _hook: &str, payload: &[u8]) -> Option<Vec<u8>> {
            Some(String::from_utf8_lossy(payload).to_uppercase().into_bytes())
        }
        fn observe(&mut self, _hook: &str, payload: &[u8]) {
            self.seen
                .borrow_mut()
                .push(String::from_utf8_lossy(payload).into_owned());
        }
    }

    fn run(host: &mut Host, cap: &Capability, hook: &str, input: &str) -> String {
        String::from_utf8(host.transform(cap, hook, input.as_bytes())).unwrap()
    }

    #[test]
    fn a_granted_transform_runs() {
        let mut host = Host::new();
        host.register(
            &manifest("up", &[content()]),
            &[content()],
            Box::new(Spy::default()),
        )
        .unwrap();
        assert_eq!(
            run(&mut host, &content(), "message", "hi there"),
            "HI THERE"
        );
    }

    #[test]
    fn a_hook_without_its_capability_never_runs() {
        // The plugin implements transform but is granted only `events`, so a
        // transform under `content` must leave the payload alone.
        let mut host = Host::new();
        host.register(
            &manifest("up", &[content(), events()]),
            &[events()],
            Box::new(Spy::default()),
        )
        .unwrap();
        assert_eq!(
            run(&mut host, &content(), "message", "hi there"),
            "hi there"
        );
    }

    #[test]
    fn transforms_compose_in_registration_order() {
        struct Wrap(&'static str, &'static str);
        impl Plugin for Wrap {
            fn transform(&mut self, _hook: &str, payload: &[u8]) -> Option<Vec<u8>> {
                Some(
                    format!("{}{}{}", self.0, String::from_utf8_lossy(payload), self.1)
                        .into_bytes(),
                )
            }
        }
        let mut host = Host::new();
        host.register(
            &manifest("a", &[content()]),
            &[content()],
            Box::new(Wrap("(", ")")),
        )
        .unwrap();
        host.register(
            &manifest("b", &[content()]),
            &[content()],
            Box::new(Wrap("[", "]")),
        )
        .unwrap();
        // a wraps first, then b wraps a's output.
        assert_eq!(run(&mut host, &content(), "message", "mid"), "[(mid)]");
    }

    #[test]
    fn observe_reaches_only_granted_plugins() {
        let seen = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let mut host = Host::new();
        host.register(
            &manifest("watcher", &[events()]),
            &[events()],
            Box::new(Spy { seen: seen.clone() }),
        )
        .unwrap();
        // A second plugin granted only `content` must not observe.
        host.register(
            &manifest("blind", &[content()]),
            &[content()],
            Box::new(Spy::default()),
        )
        .unwrap();
        host.observe(&events(), "ready", b"payload");
        assert_eq!(&*seen.borrow(), &vec!["payload".to_string()]);
    }

    #[test]
    fn rejects_granting_a_capability_the_manifest_didnt_ask_for() {
        let mut host = Host::new();
        let err = host
            .register(
                &manifest("p", &[content()]),
                &[events()],
                Box::new(Spy::default()),
            )
            .unwrap_err();
        assert_eq!(err, HostError::UngrantedCapability(events()));
    }

    #[test]
    fn rejects_a_duplicate_id() {
        let mut host = Host::new();
        host.register(
            &manifest("dup", &[content()]),
            &[content()],
            Box::new(Spy::default()),
        )
        .unwrap();
        let err = host
            .register(
                &manifest("dup", &[content()]),
                &[content()],
                Box::new(Spy::default()),
            )
            .unwrap_err();
        assert_eq!(err, HostError::DuplicateId("dup".into()));
    }
}
