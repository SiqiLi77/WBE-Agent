"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import { t as translate, type DictKey, type Locale } from "@/lib/i18n";

type I18nCtx = {
  locale: Locale;
  setLocale: (l: Locale) => void;
  t: (key: DictKey) => string;
};

const Ctx = createContext<I18nCtx>({
  locale: "en",
  setLocale: () => {},
  t: (k) => k,
});

export function LocaleProvider({ children }: { children: ReactNode }) {
  // Always start with "en" so SSR and first client render match.
  const [locale, setLocaleState] = useState<Locale>("en");
  const [mounted, setMounted] = useState(false);

  // After hydration, read saved preference from localStorage.
  useEffect(() => {
    const saved = localStorage.getItem("wbe_locale") as Locale | null;
    if (saved === "en" || saved === "zh") setLocaleState(saved);
    setMounted(true);
  }, []);

  const setLocale = useCallback((l: Locale) => {
    setLocaleState(l);
    localStorage.setItem("wbe_locale", l);
  }, []);

  // Use the real locale only after mount; before that, always "en"
  // so server HTML and first client render are identical.
  const effectiveLocale = mounted ? locale : "en";

  const tFn = useCallback(
    (key: DictKey) => translate(key, effectiveLocale),
    [effectiveLocale],
  );

  return (
    <Ctx.Provider value={{ locale: effectiveLocale, setLocale, t: tFn }}>
      {children}
    </Ctx.Provider>
  );
}

export function useLocale() {
  return useContext(Ctx);
}
