import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
    title: "Home Credit Risk Analysis",
    description: "AI-powered credit risk assessment chatbot",
    icons: {
        icon: [
            { url: "/hc_logo.png", type: "image/png" },
        ],
        apple: "/hc_logo.png",
    },
};

export default function RootLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <html lang="en">
            <head>
                <link rel="icon" href="/hc_logo.png" type="image/png" />
            </head>
            <body>{children}</body>
        </html>
    );
}
