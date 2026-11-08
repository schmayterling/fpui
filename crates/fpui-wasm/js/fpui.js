// fpui JS plugin helper (Path A: Javy WASI-command model).
//
// The host frames the plugin's stdin as `[u32 LE hook_len][hook][payload]` (see
// crates/fpui-wasm/src/javy.rs) so the plugin receives BOTH the hook name and
// the payload. This helper parses that frame and dispatches to a per-hook
// handler, so one plugin can transform one hook and pass the others through
// unchanged. The result is written to stdout unframed (just the new payload).
//
// Authoring: bundle this file ahead of your plugin into one script, then
// `javy build bundle.js -o plugin.wasm`. Exposes `globalThis.fpui`.
globalThis.fpui = (() => {
  // Drain all of stdin into one Uint8Array.
  const readInput = () => {
    const chunks = [];
    const buf = new Uint8Array(65536);
    for (let n; (n = Javy.IO.readSync(0, buf)) > 0; ) chunks.push(buf.slice(0, n));
    const all = new Uint8Array(chunks.reduce((sum, c) => sum + c.length, 0));
    let off = 0;
    for (const c of chunks) { all.set(c, off); off += c.length; }
    return all;
  };

  // Parse the frame into { hook: string, payload: Uint8Array }.
  const input = () => {
    const bytes = readInput();
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const hookLen = view.getUint32(0, true); // little-endian, matches the host
    const hook = new TextDecoder().decode(bytes.subarray(4, 4 + hookLen));
    return { hook, payload: bytes.subarray(4 + hookLen) };
  };

  // Write the rewritten payload. A string is UTF-8 encoded; a Uint8Array is sent
  // as-is. Writing nothing (never calling this) tells the host "no change".
  const output = (data) =>
    Javy.IO.writeSync(1, data instanceof Uint8Array ? data : new TextEncoder().encode(String(data)));

  // Dispatch a TEXT transform by hook. `handlers` maps a hook name to
  // `(text, hook) => string | null`. A hook with no handler, or a handler that
  // returns null/undefined, writes nothing → the host keeps the original
  // payload. For binary payloads, use `fpui.input()` / `fpui.output()` directly.
  const transform = (handlers) => {
    const { hook, payload } = input();
    const handler = handlers[hook];
    if (!handler) return; // unhandled hook → no change
    const out = handler(new TextDecoder().decode(payload), hook);
    if (out != null) output(out);
  };

  return { input, output, transform };
})();
