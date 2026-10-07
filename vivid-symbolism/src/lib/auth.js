// Password gate. The single password lives in the SITE_PASSWORD environment
// variable. The session cookie is "<expiry>:<HMAC-SHA256(password, expiry)>",
// so changing the password signs everyone out. With no password set, nothing
// but the login page is served.

export const SESSION_COOKIE = "vs_session";
export const SESSION_TTL_SECONDS = 7 * 24 * 3600;

const password = () => process.env.SITE_PASSWORD || "";

async function sign(message) {
  const key = await crypto.subtle.importKey(
    "raw", new TextEncoder().encode(password()), { name: "HMAC", hash: "SHA-256" }, false, ["sign"],
  );
  const sig = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(message));
  return Array.from(new Uint8Array(sig), (b) => b.toString(16).padStart(2, "0")).join("");
}

function equal(a, b) {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i += 1) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

export async function passwordMatches(given) {
  return password() !== "" && equal(await sign(given || ""), await sign(password()));
}

export async function createSession() {
  const exp = Math.floor(Date.now() / 1000) + SESSION_TTL_SECONDS;
  return `${exp}:${await sign(String(exp))}`;
}

export async function validSession(value) {
  if (!value || password() === "") return false;
  const [expStr, sig] = value.split(":");
  const exp = Number(expStr);
  if (!Number.isInteger(exp) || exp < Date.now() / 1000 || !sig) return false;
  return equal(sig, await sign(expStr));
}

// Only same-site paths may be used as the post-login destination.
export function safeNext(next) {
  if (typeof next !== "string" || !next.startsWith("/") || next.startsWith("//") || next.startsWith("/\\")) return "/";
  return next.startsWith("/login") || next.startsWith("/api/") ? "/" : next;
}
