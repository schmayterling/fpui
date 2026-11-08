//! The host side of maycord: it loads plugins and runs them against a small,
//! capability-gated hook surface.
//!
//! A plugin never touches the client directly. It implements [`Plugin`] and the
//! host reaches it only through hooks the plugin was *granted* a [`Capability`]
//! for. There are two kinds of hook:
//!
//! * **observe** — [`Plugin::on_event`] is handed an [`Event`] the plugin may
//!   react to but not change. Gated by [`Capability::Events`].
//! * **transform** — [`Plugin::transform_message`] and
//!   [`Plugin::transform_draft`] are handed content the plugin may rewrite in
//!   place, and the host uses the result. Gated by
//!   [`Capability::MessageContent`] and [`Capability::Composer`].
//!
//! The host is the single place a capability is enforced: a plugin that was not
//! granted a capability is never invoked through that hook, whatever it
//! implements. This crate is a pure, safe *reference* host. A sandboxed
//! WebAssembly host — where the boundary is enforced by the runtime refusing a
//! host call, not by the host declining to make one — implements the same
//! [`Plugin`] contract across the wasm edge, so the hook model is identical
//! whether a plugin is native or wasm.
//!
//! What a plugin is *granted* is a subset of what its manifest *requests*: the
//! manifest is the plugin's ask, and [`Host::register`] takes the capabilities
//! the user actually consented to. Granting a capability the manifest never
//! requested is rejected, so consent can only narrow the ask, never widen it.
//!
//! ```
//! use maycord_host::{Host, Message, Plugin};
//! use maycord_manifest::{Capability, ExtensionKind, Manifest, API_VERSION};
//!
//! // A plugin that redacts a word from every message it is shown.
//! struct Redact;
//! impl Plugin for Redact {
//!     fn transform_message(&mut self, m: &mut Message) {
//!         m.content = m.content.replace("secret", "[redacted]");
//!     }
//! }
//!
//! let manifest = Manifest {
//!     api_version: API_VERSION,
//!     id: "maycord-redact".into(), name: "Redact".into(), version: "1.0.0".into(),
//!     author: "maycord".into(), license: "MIT".into(),
//!     source: "https://example.invalid/redact".into(),
//!     kind: ExtensionKind::Plugin,
//!     capabilities: vec![Capability::MessageContent],
//! };
//!
//! let mut host = Host::new();
//! host.register(&manifest, &[Capability::MessageContent], Box::new(Redact)).unwrap();
//! assert_eq!(host.render_message("alice", "the secret is out"), "the [redacted] is out");
//! ```

use std::collections::BTreeSet;

use maycord_manifest::{Capability, ExtensionKind, Manifest, ManifestError};

/// Something that happened in the client that a plugin may observe. Observing
/// requires [`Capability::Events`]. The set is deliberately small and grows
/// additively as the client grows.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Event {
    /// The host finished loading every plugin and the client is ready.
    Ready,
    /// A message became visible in the current channel. Observe-only; to change
    /// what renders, implement [`Plugin::transform_message`] instead.
    MessageDisplayed { author: String, content: String },
    /// The client is shutting down; the plugin's last chance to flush state.
    Shutdown,
}

/// A message the client is about to render. A [`Capability::MessageContent`]
/// plugin may rewrite [`content`](Self::content) in place; [`author`](Self::author)
/// is context for that decision and is never read back by the host.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    pub author: String,
    pub content: String,
}

/// The user's unsent draft. A [`Capability::Composer`] plugin may rewrite
/// [`text`](Self::text) in place before the client sends it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Draft {
    pub text: String,
}

/// The behavior a plugin contributes. Every method defaults to a no-op, so a
/// plugin implements only the hooks it wants. Implementing a hook the plugin
/// was not granted has no effect, because the host never calls it.
pub trait Plugin {
    /// React to an observed [`Event`]. Requires [`Capability::Events`].
    fn on_event(&mut self, event: &Event) {
        let _ = event;
    }

    /// Rewrite an incoming message before it renders. Requires
    /// [`Capability::MessageContent`].
    fn transform_message(&mut self, message: &mut Message) {
        let _ = message;
    }

    /// Rewrite the user's draft before it is sent. Requires
    /// [`Capability::Composer`].
    fn transform_draft(&mut self, draft: &mut Draft) {
        let _ = draft;
    }
}

/// Why [`Host::register`] rejected a plugin.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HostError {
    /// The manifest failed [`Manifest::validate`].
    Manifest(ManifestError),
    /// The manifest declared a theme; only plugins are registered with a host.
    NotAPlugin,
    /// A capability was granted that the manifest never requested. Consent may
    /// only narrow the manifest's request, so this is always a caller bug.
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
                write!(f, "granted capability {cap:?} was not requested by the manifest")
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

/// Loads plugins and runs them against the hook surface, enforcing that each
/// plugin is only ever reached through a hook it was granted.
///
/// Hooks run in registration order, so a transform pipeline is deterministic:
/// the first plugin registered sees the original content, the next sees the
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

    /// Register a plugin. `granted` is the capability set the user consented to;
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
                return Err(HostError::UngrantedCapability(*cap));
            }
        }
        if self.plugins.iter().any(|p| p.id == manifest.id) {
            return Err(HostError::DuplicateId(manifest.id.clone()));
        }
        self.plugins.push(Registered {
            id: manifest.id.clone(),
            granted: granted.iter().copied().collect(),
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

    /// Deliver an observed event to every plugin granted [`Capability::Events`],
    /// in registration order.
    pub fn broadcast(&mut self, event: &Event) {
        self.each_granted(Capability::Events, |p| p.on_event(event));
    }

    /// Run every [`Capability::MessageContent`] plugin's message transform, in
    /// registration order, and return the final rendered content.
    pub fn render_message(&mut self, author: &str, content: &str) -> String {
        let mut message = Message {
            author: author.to_string(),
            content: content.to_string(),
        };
        self.each_granted(Capability::MessageContent, |p| p.transform_message(&mut message));
        message.content
    }

    /// Run every [`Capability::Composer`] plugin's draft transform over `draft`
    /// in place, in registration order.
    pub fn edit_draft(&mut self, draft: &mut Draft) {
        self.each_granted(Capability::Composer, |p| p.transform_draft(draft));
    }

    /// The one chokepoint where a capability is enforced: run `f` against every
    /// plugin that holds `cap`, in registration order, and no others.
    fn each_granted(&mut self, cap: Capability, mut f: impl FnMut(&mut dyn Plugin)) {
        for reg in &mut self.plugins {
            if reg.granted.contains(&cap) {
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
            api_version: maycord_manifest::API_VERSION,
            id: id.into(),
            name: id.into(),
            version: "1.0.0".into(),
            author: "maycord".into(),
            license: "MIT".into(),
            source: "https://example.invalid/p".into(),
            kind: ExtensionKind::Plugin,
            capabilities: caps.to_vec(),
        }
    }

    /// Uppercases message content and appends a marker to a draft; records every
    /// event it is given. One plugin exercising all three hooks.
    #[derive(Default)]
    struct Spy {
        events: std::rc::Rc<std::cell::RefCell<Vec<Event>>>,
    }
    impl Plugin for Spy {
        fn on_event(&mut self, event: &Event) {
            self.events.borrow_mut().push(event.clone());
        }
        fn transform_message(&mut self, m: &mut Message) {
            m.content = m.content.to_uppercase();
        }
        fn transform_draft(&mut self, d: &mut Draft) {
            d.text.push_str(" [sent via maycord]");
        }
    }

    #[test]
    fn a_granted_message_transform_runs() {
        let mut host = Host::new();
        host.register(
            &manifest("up", &[Capability::MessageContent]),
            &[Capability::MessageContent],
            Box::new(Spy::default()),
        )
        .unwrap();
        assert_eq!(host.render_message("alice", "hi there"), "HI THERE");
    }

    #[test]
    fn a_hook_without_its_capability_never_runs() {
        // The plugin implements transform_message, but is granted only Events,
        // so render_message must leave the content untouched.
        let mut host = Host::new();
        host.register(
            &manifest("up", &[Capability::MessageContent, Capability::Events]),
            &[Capability::Events],
            Box::new(Spy::default()),
        )
        .unwrap();
        assert_eq!(host.render_message("alice", "hi there"), "hi there");
    }

    #[test]
    fn transforms_compose_in_registration_order() {
        struct Wrap(&'static str, &'static str);
        impl Plugin for Wrap {
            fn transform_message(&mut self, m: &mut Message) {
                m.content = format!("{}{}{}", self.0, m.content, self.1);
            }
        }
        let mut host = Host::new();
        host.register(&manifest("a", &[Capability::MessageContent]), &[Capability::MessageContent], Box::new(Wrap("(", ")"))).unwrap();
        host.register(&manifest("b", &[Capability::MessageContent]), &[Capability::MessageContent], Box::new(Wrap("[", "]"))).unwrap();
        // a wraps first, then b wraps a's output.
        assert_eq!(host.render_message("x", "mid"), "[(mid)]");
    }

    #[test]
    fn broadcast_reaches_only_events_plugins() {
        let seen = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let spy = Spy { events: seen.clone() };
        let mut host = Host::new();
        host.register(&manifest("s", &[Capability::Events]), &[Capability::Events], Box::new(spy)).unwrap();
        // A second plugin without Events must not see the broadcast.
        host.register(&manifest("mute", &[Capability::MessageContent]), &[Capability::MessageContent], Box::new(Spy::default())).unwrap();
        host.broadcast(&Event::Ready);
        host.broadcast(&Event::Shutdown);
        assert_eq!(*seen.borrow(), vec![Event::Ready, Event::Shutdown]);
    }

    #[test]
    fn draft_transform_edits_in_place() {
        let mut host = Host::new();
        host.register(&manifest("d", &[Capability::Composer]), &[Capability::Composer], Box::new(Spy::default())).unwrap();
        let mut draft = Draft { text: "hello".into() };
        host.edit_draft(&mut draft);
        assert_eq!(draft.text, "hello [sent via maycord]");
    }

    #[test]
    fn register_rejects_a_theme() {
        let mut m = manifest("t", &[]);
        m.kind = ExtensionKind::Theme;
        let mut host = Host::new();
        assert_eq!(host.register(&m, &[], Box::new(Spy::default())), Err(HostError::NotAPlugin));
    }

    #[test]
    fn register_rejects_granting_more_than_requested() {
        let mut host = Host::new();
        let err = host.register(
            &manifest("p", &[Capability::Events]),
            &[Capability::Storage],
            Box::new(Spy::default()),
        );
        assert_eq!(err, Err(HostError::UngrantedCapability(Capability::Storage)));
    }

    #[test]
    fn register_rejects_a_duplicate_id() {
        let mut host = Host::new();
        host.register(&manifest("dup", &[]), &[], Box::new(Spy::default())).unwrap();
        assert_eq!(
            host.register(&manifest("dup", &[]), &[], Box::new(Spy::default())),
            Err(HostError::DuplicateId("dup".into()))
        );
    }

    #[test]
    fn register_rejects_an_invalid_manifest() {
        let mut host = Host::new();
        let bad = manifest("BAD ID", &[]);
        assert!(matches!(host.register(&bad, &[], Box::new(Spy::default())), Err(HostError::Manifest(_))));
    }

    #[test]
    fn granting_a_subset_of_the_request_is_allowed() {
        let mut host = Host::new();
        // Requests two, user consents to one; the ungranted one simply stays off.
        host.register(
            &manifest("sub", &[Capability::MessageContent, Capability::Composer]),
            &[Capability::MessageContent],
            Box::new(Spy::default()),
        )
        .unwrap();
        assert_eq!(host.render_message("a", "x"), "X");
        let mut draft = Draft { text: "keep".into() };
        host.edit_draft(&mut draft);
        assert_eq!(draft.text, "keep"); // Composer was not granted.
    }
}
