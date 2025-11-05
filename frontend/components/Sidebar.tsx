'use client'

import { useEffect } from 'react'

interface SidebarProps {
  isOpen: boolean
  onClose: () => void
  onNewChat: () => void
  onClearChat: () => void
}

export default function Sidebar({
  isOpen,
  onClose,
  onNewChat,
  onClearChat,
}: SidebarProps) {
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden'
    } else {
      document.body.style.overflow = 'unset'
    }
    return () => {
      document.body.style.overflow = 'unset'
    }
  }, [isOpen])

  return (
    <>
      {isOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50 md:hidden"
          onClick={onClose}
        />
      )}
      <aside
        className={`fixed left-0 top-0 z-50 h-full w-64 transform border-r border-chat-gpt-border bg-chat-gpt-bg transition-transform duration-300 ease-in-out ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        } md:relative md:translate-x-0`}
      >
        <div className="flex h-full flex-col">
          <div className="flex h-12 items-center justify-between border-b border-chat-gpt-border px-4">
            <button
              onClick={onNewChat}
              className="flex items-center gap-2 rounded-lg border border-chat-gpt-border bg-transparent px-3 py-1.5 text-sm text-chat-gpt-text transition-colors hover:bg-chat-gpt-bg-secondary"
            >
              <svg
                className="h-4 w-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 4v16m8-8H4"
                />
              </svg>
              New Chat
            </button>
            <button
              onClick={onClose}
              className="rounded p-1.5 text-chat-gpt-text-secondary hover:bg-chat-gpt-bg-secondary md:hidden"
              aria-label="Close sidebar"
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
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-2">
            <div className="space-y-1">
              <button
                onClick={onClearChat}
                className="w-full rounded-lg px-3 py-2 text-left text-sm text-chat-gpt-text-secondary transition-colors hover:bg-chat-gpt-bg-secondary"
              >
                Clear Chat History
              </button>
            </div>
          </div>

          <div className="border-t border-chat-gpt-border p-4">
            <div className="rounded-lg bg-chat-gpt-bg-secondary p-3">
              <p className="text-xs text-chat-gpt-text-secondary">
                Offline Pharmaceutical Assistant
              </p>
              <p className="mt-1 text-xs text-chat-gpt-text-secondary">
                Powered by LLaMA 3.2
              </p>
            </div>
          </div>
        </div>
      </aside>
    </>
  )
}

