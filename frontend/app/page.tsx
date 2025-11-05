'use client'

import { useState, useRef, useEffect } from 'react'
import ChatInterface from '@/components/ChatInterface'

export default function Home() {
  return (
    <main className="flex h-screen w-full flex-col bg-chat-gpt-bg">
      <ChatInterface />
    </main>
  )
}

