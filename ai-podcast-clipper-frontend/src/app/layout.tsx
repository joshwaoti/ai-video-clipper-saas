import { type Metadata } from "next";
import { Geist } from "next/font/google";

import "~/styles/globals.css";

const geist = Geist({
  subsets: ["latin"],
  variable: "--font-geist",
});

export const metadata: Metadata = {
  title: "AI Podcast Clipper",
  description: "Clip your favorite podcast moments with AI",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={geist.variable}>
      <body>{children}</body>
    </html>
  );
}
