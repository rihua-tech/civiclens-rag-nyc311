import type { Metadata } from "next";

import "./globals.css";

const title = "CivicLens | Grounded NYC 311 intelligence";
const description =
  "A non-production CivicLens portfolio interface for grounded NYC 311 documentation answers, approved analytics summaries, and validated provenance.";
const metadataBase = new URL(
  process.env.NEXT_PUBLIC_CIVICLENS_SITE_URL ?? "http://localhost:3000",
);

export const metadata: Metadata = {
  metadataBase,
  title,
  description,
  openGraph: {
    type: "website",
    title,
    description,
    images: [
      {
        url: "/og.png",
        width: 1731,
        height: 909,
        alt: "CivicLens — Ask the system. Inspect the evidence.",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title,
    description,
    images: ["/og.png"],
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
