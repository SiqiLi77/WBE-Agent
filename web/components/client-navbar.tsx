"use client";

import dynamic from "next/dynamic";

// Dynamically import Navbar with ssr:false to prevent hydration mismatch
// caused by locale context reading localStorage on the client.
const NavbarInner = dynamic(
  () => import("@/components/navbar").then((m) => m.Navbar),
  { ssr: false },
);

export function ClientNavbar() {
  return <NavbarInner />;
}
