import { Html, Head, Main, NextScript } from "next/document";

/**
 * The site is dark, permanently.
 *
 * `dark` is hard-coded on <html> rather than applied by a script,
 * because there is no preference left to read. The class still exists
 * because `darkMode: "class"` in the Tailwind config keys every
 * `dark:` utility on it -- removing the class would silently fall the
 * whole site back to its light palette, so it stays, pinned.
 *
 * Setting it in the markup also means the served HTML is already dark.
 * The previous beforeInteractive script could only add the class after
 * the document existed, which is what produced the light flash on a
 * cold load.
 */
export default function Document() {
  return (
    <Html lang="en" className="dark">
      <Head />
      <body>
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}
