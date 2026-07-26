// Mermaid's vendored browser build is a classic script, not an ES module.
// Load it in classic-script mode so its global export is initialized, then
// expose that export in the module shape expected by sphinxcontrib-mermaid.
if (!globalThis.mermaid) {
  await new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = new URL("./mermaid.min.js", import.meta.url).href;
    script.onload = resolve;
    script.onerror = () => reject(new Error("Unable to load the local Mermaid bundle"));
    document.head.appendChild(script);
  });
}

if (!globalThis.mermaid) {
  throw new Error("The local Mermaid bundle did not expose its browser API");
}

export default globalThis.mermaid;
