import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Offline Chatbot - Pharmaceutical Assistant',
  description: 'An offline chatbot assistant for pharmaceutical batch records',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}

