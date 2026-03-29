"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useLocale } from "@/lib/i18n-context";

const NAV_KEYS = [
  { href: "/", key: "nav.dashboard" as const },
  { href: "/sites", key: "nav.sites" as const },
  { href: "/pipeline", key: "nav.pipeline" as const },
  { href: "/experiments", key: "nav.experiments" as const },
];

export function Navbar() {
  const pathname = usePathname();
  const { locale, setLocale, t } = useLocale();

  return (
    <nav className="sticky top-0 z-50 border-b border-slate-200 bg-white/80 backdrop-blur-md">
      <div className="mx-auto flex max-w-7xl items-center gap-6 px-4 py-3 md:px-8">
        <Link href="/" className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-teal-600 text-xs font-bold text-white">
            W
          </div>
          <span className="text-sm font-semibold text-slate-800">WBE-Agent</span>
        </Link>
        <div className="flex gap-1">
          {NAV_KEYS.map((item) => {
            const active =
              item.href === "/"
                ? pathname === "/"
                : pathname.startsWith(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`rounded-lg px-3 py-1.5 text-sm transition ${
                  active
                    ? "bg-teal-50 font-semibold text-teal-800"
                    : "text-slate-600 hover:bg-slate-50 hover:text-slate-800"
                }`}
              >
                {t(item.key)}
              </Link>
            );
          })}
        </div>
        {/* spacer */}
        <div className="flex-1" />
        {/* Language toggle */}
        <button
          onClick={() => setLocale(locale === "en" ? "zh" : "en")}
          className="flex items-center gap-1.5 rounded-lg border border-slate-200 px-2.5 py-1.5 text-xs text-slate-600 transition hover:bg-slate-50 hover:text-slate-800"
          aria-label="Toggle language"
        >
          <span className="text-sm">🌐</span>
          <span className="font-medium">{locale === "en" ? "中文" : "EN"}</span>
        </button>
      </div>
    </nav>
  );
}
