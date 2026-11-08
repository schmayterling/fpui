// Example fpui JS plugin: uppercase the "message" hook, pass everything else
// through unchanged. Bundle after ../fpui.js, then `javy build` the pair.
//
//   cat crates/fpui-wasm/js/fpui.js crates/fpui-wasm/js/examples/uppercase-message.js > bundle.js
//   javy build bundle.js -o uppercase-message.wasm
//
// Proves hook dispatch end to end: with hook "message" the payload is
// uppercased; with any other hook the plugin writes nothing and the host keeps
// the original payload.
fpui.transform({
  message: (text) => text.toUpperCase(),
});
