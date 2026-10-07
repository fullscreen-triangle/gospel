// Every page and every file under public/ requires the password session.

import { NextResponse } from "next/server";

import { SESSION_COOKIE, validSession } from "@/lib/auth";

const OPEN = new Set(["/login", "/api/login", "/favicon.ico"]);

export async function middleware(req) {
  const { pathname, search } = req.nextUrl;
  if (OPEN.has(pathname) || (await validSession(req.cookies.get(SESSION_COOKIE)?.value))) {
    const res = NextResponse.next();
    res.headers.set("Cache-Control", "private, no-store");
    return res;
  }
  const url = req.nextUrl.clone();
  url.pathname = "/login";
  url.search = `?next=${encodeURIComponent(pathname + search)}`;
  return NextResponse.redirect(url);
}

export const config = {
  matcher: ["/((?!_next/static|_next/image).*)"],
};
