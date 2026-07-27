// Mermaid's vendored browser build is a classic script, not an ES module.
// Load it in classic-script mode so its global export is initialized, then
// expose that export in the module shape expected by sphinxcontrib-mermaid.
if (!globalThis.mermaid) {
  await new Promise((resolve, reject) => {
    const script = document.createElement("script");
    // Keep the runtime URL versioned.  Without a query token, a long-lived
    // local preview tab can reuse an older Mermaid bundle after the vendored
    // file is updated; older releases report native $$...$$ labels as a
    // generic "Syntax error in text".
    script.src = new URL("./mermaid.min.js?v=11.12.1", import.meta.url).href;
    script.onload = resolve;
    script.onerror = () => reject(new Error("Unable to load the local Mermaid bundle"));
    document.head.appendChild(script);
  });
}

if (!globalThis.mermaid) {
  throw new Error("The local Mermaid bundle did not expose its browser API");
}

export default globalThis.mermaid;
