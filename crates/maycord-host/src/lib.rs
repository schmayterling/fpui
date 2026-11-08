//! The host side of maycord: it loads plugins and runs them against a small set
//! of hooks, each guarded by a capability.
//!
//! A plugin never touches the client directly. It implements [`Plugin`], and the
//! host only calls a hook the plugin was *granted* the matching [`Capability`]
//! for. There are two kinds of hook:
//!
//! * **observe** — [`Plugin::on_event`] gets an [`Event`] the plugin can react
//!   to but not change. Needs [`Capability::Events`].
//! * **transform** — [`Plugin::transform_message`] and
//!   [`Plugin::transform_draft`] get content the plugin can rewrite in place,
//!   and the host uses the result. Need [`Capability::MessageContent`] and
//!   [`Capability::Composer`].
//!
//! The host is the only place a capability is checked: if a plugin wasn't
//! granted one, the host simply never calls that hook, no matter what the plugin
//! implements. This crate is a plain, safe *reference* host. A sandboxed
//! WebAssembly host enforces the same boundary a different way — the runtime
//! refuses a host call instead of the host declining to make one — but it
//! implements the same [`Plugin`] trait, so the hooks work the same whether a
//! plugin is native or wasm.
//!
//! What a plugin is *granted* is a subset of what its manifest *asks for*: the
//! manifest is the request, and [`Host::register`] takes the capabilities the
//! user actually approved. Granting a capability the manifest never asked for is
//! rejected, so approval can only shrink the request, never grow it.
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

/// Something that happened in the client that a plugin can watch. Watching needs
/// [`Capability::Events`]. The list is small on purpose and grows over time.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Event {
    /// The host finished loading every plugin and the client is ready.
    Ready,
    /// A message showed up in the current channel. This carries the `author`
    /// only, not the text: it's an [`Events`](Capability::Events) signal, and
    /// reading or changing message text needs [`Capability::MessageContent`] via
    /// [`Plugin::transform_message`] instead. Watching for messages must not be a
    /// way to read text the plugin wasn't granted.
    MessageDisplayed { author: String },
    /// The client is shutting down; the plugin's last chance to save state.
    Shutdown,
}

/// A message the client is about to show. A [`Capability::MessageContent`]
/// plugin can rewrite [`content`](Self::content) in place; [`author`](Self::author)
/// is there for context and the host never reads it back.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    pub author: String,
    pub content: String,
}

/// The user's unsent draft. A [`Capability::Composer`] plugin can rewrite
/// [`text`](Self::text) in place before the client sends it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Draft {
    pub text: String,
}

/// What a plugin does. Every method does nothing by default, so a plugin only
/// implements the hooks it cares about. Implementing a hook the plugin wasn't
/// granted does nothing, because the host never calls it.
pub trait Plugin {
    /// React to an observed [`Event`]. Requires [`Capability::Events`].
    fn on_event(&mut self, event: &Event) {
        let _ = event;
    }

    /// Rewrite an incoming message before it's shown. Requires
    /// [`Capability::MessageContent`].
    fn transform_message(&mut self, message: &mut Message) {
        let _ = message;
    }

    /// Rewrite the user's draft before it's sent. Requires
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

/// Loads plugins and runs them through the hooks, making sure each plugin is
/// only reached through a hook it was granted.
///
/// Hooks run in the order plugins were registered, so a chain of transforms is
/// predictable: the first plugin sees the original content, the next sees the
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

    /// Send an event to every plugin granted [`Capability::Events`], in
    /// registration order.
    pub fn broadcast(&mut self, event: &Event) {
        self.each_granted(Capability::Events, |p| p.on_event(event));
    }

    /// Run each [`Capability::MessageContent`] plugin's message transform in
    /// order and return the final text.
    pub fn render_message(&mut self, author: &str, content: &str) -> String {
        let mut message = Message {
            author: author.to_string(),
            content: content.to_string(),
        };
        self.each_granted(Capability::MessageContent, |p| p.transform_message(&mut message));
        message.content
    }

    /// Run each [`Capability::Composer`] plugin's draft transform over `draft`
    /// in place, in order.
    pub fn edit_draft(&mut self, draft: &mut Draft) {
        self.each_granted(Capability::Composer, |p| p.transform_draft(draft));
    }

    /// The single place a capability is checked: run `f` on every plugin that
    /// holds `cap`, in registration order, and no others.
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

    /// Uppercases message content, appends a marker to a draft, and records every
    /// event it's given. One plugin that uses all three hooks.
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
        // The plugin implements transform_message but is granted only Events, so
        // render_message must leave the content alone.
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
        host.broadcast(&Event::MessageDisplayed { author: "alice".into() });
        host.broadcast(&Event::Shutdown);
        assert_eq!(
            *seen.borrow(),
            vec![
                Event::Ready,
                Event::MessageDisplayed { author: "alice".into() },
                Event::Shutdown
            ]
        );
    }

    #[test]
    fn events_payload_carries_no_message_content() {
        // Guards against a capability leak: an Events-only plugin must never get
        // message text. MessageDisplayed carries the author only; text comes only
        // through the MessageContent transform, which this plugin wasn't granted.
        let seen = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let mut host = Host::new();
        host.register(
            &manifest("obs", &[Capability::Events]),
            &[Capability::Events],
            Box::new(Spy { events: seen.clone() }),
        )
        .unwrap();
        host.broadcast(&Event::MessageDisplayed { author: "bob".into() });
        let events = seen.borrow();
        match &events[0] {
            Event::MessageDisplayed { author } => assert_eq!(author, "bob"),
            other => panic!("unexpected event: {other:?}"),
        }
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
        // Asks for two, user approves one; the one not granted just stays off.
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
