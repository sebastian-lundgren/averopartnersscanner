import type { Metadata } from "next";
import Nav from "@/components/Nav";
import "./globals.css";

export const metadata: Metadata = {
  title: "Alarmskilt QC",
  description:
    "Review- og treningsverktøy for autoriserte bilder — unknown-first, menneske i løkken.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="no">
      <body>
        <Nav />
        <main>{children}</main>
      </body>
    </html>
  );
}
