'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import MessageList from './MessageList'
import ChatInput from './ChatInput'
import Sidebar from './Sidebar'
import { Message } from '@/types/chat'

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)
  const [userId] = useState(() => `user_${Date.now()}`)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, scrollToBottom])

  const handleSendMessage = async (content: string) => {
    if (!content.trim() || isLoading) return

    const userMessage: Message = {
      id: `msg_${Date.now()}`,
      role: 'user',
      content: content.trim(),
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setIsLoading(true)

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: content.trim(),
          conversation_history: messages.map((m) => ({
            role: m.role,
            content: m.content,
          })),
          user_id: userId,
        }),
      })

      if (!response.ok) {
        throw new Error('Failed to get response')
      }

      const data = await response.json()
      
      const assistantMessage: Message = {
        id: `msg_${Date.now() + 1}`,
        role: 'assistant',
        content: data.response,
        timestamp: new Date(),
      }

      setMessages((prev) => [...prev, assistantMessage])
    } catch (error) {
      console.error('Error sending message:', error)
      const errorMessage: Message = {
        id: `msg_${Date.now() + 1}`,
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date(),
        isError: true,
      }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleNewChat = () => {
    setMessages([])
    setIsSidebarOpen(false)
  }

  const handleClearChat = () => {
    if (confirm('Are you sure you want to clear all messages?')) {
      setMessages([])
    }
  }

  return (
    <div className="flex h-full w-full overflow-hidden">
      <Sidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        onNewChat={handleNewChat}
        onClearChat={handleClearChat}
      />
      
      <div className="flex flex-1 flex-col">
        <header className="flex h-12 items-center border-b border-chat-gpt-border bg-chat-gpt-bg px-4">
          <button
            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            className="mr-3 rounded p-1.5 text-chat-gpt-text-secondary hover:bg-chat-gpt-bg-secondary"
            aria-label="Toggle sidebar"
          >
            <svg
              className="h-5 w-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 6h16M4 12h16M4 18h16"
              />
            </svg>
          </button>
          <h1 className="text-sm font-semibold text-chat-gpt-text">
            Pharmaceutical Assistant
          </h1>
        </header>

        <div className="flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            <div className="flex h-full items-center justify-center">
              <div className="max-w-2xl px-4 text-center">
                <h2 className="mb-4 text-4xl font-semibold text-chat-gpt-text">
                  Pharmaceutical Batch Record Assistant
                </h2>
                <p className="mb-8 text-lg text-chat-gpt-text-secondary">
                  Ask questions about lot numbers, batch records, and manufacturing details.
                </p>
                <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                  {[
                    'What is lot number L456?',
                    'Show me batch record for lot L123',
                    'When was lot L789 manufactured?',
                    'What are the ingredients in lot L456?',
                  ].map((suggestion, idx) => (
                    <button
                      key={idx}
                      onClick={() => handleSendMessage(suggestion)}
                      className="rounded-lg border border-chat-gpt-border bg-chat-gpt-bg-secondary p-3 text-left text-sm text-chat-gpt-text transition-colors hover:bg-chat-gpt-bg-tertiary"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <MessageList messages={messages} isLoading={isLoading} />
          )}
          <div ref={messagesEndRef} />
        </div>

        <ChatInput
          onSendMessage={handleSendMessage}
          isLoading={isLoading}
          disabled={isLoading}
        />
      </div>
    </div>
  )
}

