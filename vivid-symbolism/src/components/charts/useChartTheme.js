// Tiny hook to read the current dark/light mode from <html class>.
// Charts read this so axes and grid lines flip with the site theme.

import { useEffect, useState } from "react";

export function useChartTheme() {
  const [isDark, setIsDark] = useState(false);
  useEffect(() => {
    if (typeof window === "undefined") return undefined;
    const root = document.documentElement;
    const update = () => setIsDark(root.classList.contains("dark"));
    update();
    const obs = new MutationObserver(update);
    obs.observe(root, { attributes: true, attributeFilter: ["class"] });
    return () => obs.disconnect();
  }, []);
  return {
    fg: isDark ? "#f5f5f5" : "#1b1b1b",
    fgMuted: isDark ? "rgba(245,245,245,0.55)" : "rgba(27,27,27,0.55)",
    grid: isDark ? "rgba(245,245,245,0.18)" : "rgba(27,27,27,0.18)",
    accent: isDark ? "#58E6D9" : "#B63E96",
    accentSoft: isDark ? "rgba(88,230,217,0.25)" : "rgba(182,62,150,0.25)",
    bg: isDark ? "#1b1b1b" : "#f5f5f5",
  };
}
