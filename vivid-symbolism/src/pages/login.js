import Head from "next/head";
import { useRouter } from "next/router";

export default function Login() {
  const { query } = useRouter();
  const next = typeof query.next === "string" ? query.next : "/";

  return (
    <>
      <Head>
        <title>Sign in</title>
        <meta name="robots" content="noindex, nofollow" />
      </Head>
      <div className="flex min-h-screen w-full items-center justify-center px-6 dark:text-light">
        <form
          method="POST"
          action="/api/login"
          className="w-full max-w-sm rounded-2xl border-2 border-dark bg-light p-8 dark:border-light dark:bg-dark"
        >
          <input type="hidden" name="next" value={next} />
          <label className="block text-sm font-semibold" htmlFor="password">Password</label>
          <input
            id="password" name="password" type="password" autoComplete="current-password" required autoFocus
            className="mt-2 w-full rounded-lg border-2 border-dark/40 bg-transparent px-3 py-2 font-medium
              outline-none focus:border-primary dark:border-light/40 dark:focus:border-primaryDark"
          />
          {query.error ? (
            <p className="mt-3 text-sm font-medium text-[#e34948]">Wrong password.</p>
          ) : null}
          <button
            type="submit"
            className="mt-5 w-full rounded-lg bg-dark px-4 py-2.5 font-semibold text-light
              hover:bg-primary dark:bg-light dark:text-dark dark:hover:bg-primaryDark"
          >
            Enter
          </button>
        </form>
      </div>
    </>
  );
}
