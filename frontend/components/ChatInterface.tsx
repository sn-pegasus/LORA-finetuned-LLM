'use client'

import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import MessageList from './MessageList'
import ChatInput from './ChatInput'
import Sidebar from './Sidebar'
import { Message, ChatSession } from '@/types/chat'

const SESSIONS_STORAGE_KEY = 'offline-chatbot:sessions'
const SESSION_LIMIT = 10

export default function ChatInterface() {
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)
  const [userId] = useState(() => `user_${Date.now()}`)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const [isModelReady, setIsModelReady] = useState(false)
  const [isBackendReachable, setIsBackendReachable] = useState<boolean | null>(null)

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  const activeSession = useMemo(() => sessions.find(s => s.id === activeSessionId) || null, [sessions, activeSessionId])
  const messages = activeSession?.messages ?? []

  // Load sessions from localStorage
  useEffect(() => {
    try {
      const raw = localStorage.getItem(SESSIONS_STORAGE_KEY)
      if (raw) {
        const parsed: ChatSession[] = JSON.parse(raw)
        // revive dates
        const revived = parsed.map(s => ({
          ...s,
          messages: (s.messages || []).map(m => ({ ...m, timestamp: new Date(m.timestamp) }))
        }))
        setSessions(revived)
        setActiveSessionId(revived[0]?.id || null)
      } else {
        // create initial empty session
        const first: ChatSession = {
          id: `sess_${Date.now()}`,
          title: 'New Chat',
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
          messages: [],
        }
        setSessions([first])
        setActiveSessionId(first.id)
      }
    } catch {
      // fallback to one new session
      const first: ChatSession = {
        id: `sess_${Date.now()}`,
        title: 'New Chat',
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        messages: [],
      }
      setSessions([first])
      setActiveSessionId(first.id)
    }
  }, [])

  // Persist sessions
  useEffect(() => {
    try {
      localStorage.setItem(SESSIONS_STORAGE_KEY, JSON.stringify(sessions))
    } catch {}
  }, [sessions])

  useEffect(() => {
    scrollToBottom()
  }, [messages, scrollToBottom])

  // Poll backend health until ready
  useEffect(() => {
    let timer: NodeJS.Timeout | null = null
    const check = async () => {
      try {
        const res = await fetch('/api/health', { cache: 'no-store' })
        if (!res.ok) throw new Error('not ok')
        const data = await res.json()
        setIsBackendReachable(true)
        setIsModelReady(Boolean(data?.model_loaded))
      } catch {
        setIsBackendReachable(false)
        setIsModelReady(false)
      } finally {
        timer = setTimeout(check, 2000)
      }
    }
    check()
    return () => {
      if (timer) clearTimeout(timer)
    }
  }, [])

  const handleSendMessage = async (content: string) => {
    if (!content.trim() || isLoading) return

    const userMessage: Message = {
      id: `msg_${Date.now()}`,
      role: 'user',
      content: content.trim(),
      timestamp: new Date(),
    }

    setSessions(prev => prev.map(s => s.id === activeSessionId ? {
      ...s,
      title: s.messages.length === 0 ? userMessage.content.slice(0, 40) || 'New Chat' : s.title,
      updatedAt: new Date().toISOString(),
      messages: [...s.messages, userMessage]
    } : s))
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

      setSessions(prev => prev.map(s => s.id === activeSessionId ? {
        ...s,
        updatedAt: new Date().toISOString(),
        messages: [...s.messages, assistantMessage]
      } : s))
    } catch (error) {
      console.error('Error sending message:', error)
      const errorMessage: Message = {
        id: `msg_${Date.now() + 1}`,
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date(),
        isError: true,
      }
      setSessions(prev => prev.map(s => s.id === activeSessionId ? {
        ...s,
        updatedAt: new Date().toISOString(),
        messages: [...s.messages, errorMessage]
      } : s))
    } finally {
      setIsLoading(false)
    }
  }

  const handleNewChat = () => {
    // Enforce session limit with confirmation to delete oldest
    if (sessions.length >= SESSION_LIMIT) {
      const oldest = [...sessions].sort((a,b) => new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime())[0]
      const created = new Date(oldest.createdAt)
      const confirmMsg = `You reached the chat limit (${SESSION_LIMIT}).\n\nOldest chat will be deleted:\n- Title: ${oldest.title || 'Untitled'}\n- Created: ${created.toLocaleString()}\n\nProceed?`
      const ok = confirm(confirmMsg)
      if (!ok) return
      setSessions(prev => prev.filter(s => s.id !== oldest.id))
    }
    const newSession: ChatSession = {
      id: `sess_${Date.now()}`,
      title: 'New Chat',
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      messages: [],
    }
    setSessions(prev => [newSession, ...prev])
    setActiveSessionId(newSession.id)
    setIsSidebarOpen(false)
  }

  const handleClearChat = () => {
    if (confirm('Are you sure you want to clear all messages?')) {
      if (!activeSessionId) return
      setSessions(prev => prev.map(s => s.id === activeSessionId ? { ...s, messages: [], updatedAt: new Date().toISOString() } : s))
    }
  }

  const handleSelectSession = (id: string) => {
    setActiveSessionId(id)
    setIsSidebarOpen(false)
  }

  const handleDeleteSession = (id: string) => {
    const target = sessions.find(s => s.id === id)
    if (!target) return
    const created = new Date(target.createdAt)
    const confirmMsg = `Delete chat?\n\nTitle: ${target.title || 'Untitled'}\nCreated: ${created.toLocaleString()}`
    if (!confirm(confirmMsg)) return

    setSessions(prev => prev.filter(s => s.id !== id))
    if (activeSessionId === id) {
      // switch to newest remaining or create a new one if none
      const remaining = sessions.filter(s => s.id !== id)
      if (remaining.length > 0) {
        setActiveSessionId(remaining[0].id)
      } else {
        const first: ChatSession = {
          id: `sess_${Date.now()}`,
          title: 'New Chat',
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
          messages: [],
        }
        setSessions([first])
        setActiveSessionId(first.id)
      }
    }
  }

  return (
    <div className="flex h-full w-full overflow-hidden">
      <Sidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        onNewChat={handleNewChat}
        onClearChat={handleClearChat}
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelectSession={handleSelectSession}
        onDeleteSession={handleDeleteSession}
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
          <div className="ml-auto flex items-center gap-2 text-xs">
            <span
              className={`inline-block h-2.5 w-2.5 rounded-full ${
                isBackendReachable === false
                  ? 'bg-red-500'
                  : isModelReady
                  ? 'bg-green-500'
                  : 'bg-yellow-500'
              }`}
              aria-hidden
            />
            <span className="text-chat-gpt-text-secondary">
              {isBackendReachable === false
                ? 'Backend offline'
                : isModelReady
                ? 'Model ready'
                : 'Loading model...'}
            </span>
          </div>
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
          disabled={isLoading || !isModelReady}
        />
      </div>
    </div>
  )
}

