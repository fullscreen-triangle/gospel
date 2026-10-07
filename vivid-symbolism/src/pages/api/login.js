// POST { password, next } -> session cookie and redirect.

import { SESSION_COOKIE, SESSION_TTL_SECONDS, createSession, passwordMatches, safeNext } from "@/lib/auth";

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).end();
  const next = safeNext(req.body?.next);
  if (!(await passwordMatches(String(req.body?.password || "")))) {
    await new Promise((r) => setTimeout(r, 800));
    return res.redirect(303, `/login?error=1&next=${encodeURIComponent(next)}`);
  }
  const secure = process.env.NODE_ENV === "production" ? "; Secure" : "";
  res.setHeader("Set-Cookie",
    `${SESSION_COOKIE}=${await createSession()}; Path=/; HttpOnly; SameSite=Lax; Max-Age=${SESSION_TTL_SECONDS}${secure}`);
  return res.redirect(303, next);
}
