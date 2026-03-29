import type { Metadata } from "next";
import { Sora, IBM_Plex_Mono } from "next/font/google";
import type { ReactNode } from "react";

import { ClientNavbar } from "@/components/client-navbar";
import { Providers } from "@/app/providers";
import "@/app/globals.css";

const sora = Sora({
  variable: "--font-sora",
  subsets: ["latin"],
  display: "swap",
});

const plexMono = IBM_Plex_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
  weight: ["400", "500"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "WBE-Agent Dashboard",
  description: "Web UI for wastewater anomaly detection and investigation workflows.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${sora.variable} ${plexMono.variable}`}>
        <Providers>
          <ClientNavbar />
          {children}
        </Providers>
      </body>
    </html>
  );
}
