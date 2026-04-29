// Cheap WebGL availability check. Used to decide whether to mount the 3D
// canvas at all; if WebGL is unavailable (remote desktop, hardware
// acceleration off, strict browser, exhausted contexts) we render a
// static fallback rather than letting Three.js throw at mount time.

export function hasWebGL() {
  if (typeof window === "undefined") return false;
  try {
    const canvas = document.createElement("canvas");
    const gl =
      canvas.getContext("webgl2") ||
      canvas.getContext("webgl") ||
      canvas.getContext("experimental-webgl");
    if (!gl || typeof gl.getParameter !== "function") return false;
    // Some hardened browsers report a context but disable all queries.
    try {
      gl.getParameter(gl.VERSION);
    } catch (_) {
      return false;
    }
    return true;
  } catch (_) {
    return false;
  }
}
