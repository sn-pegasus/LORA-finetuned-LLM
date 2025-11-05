'use client'

import { useState, useRef, useEffect, KeyboardEvent, useCallback } from 'react'

interface ChatInputProps {
  onSendMessage: (message: string) => void
  isLoading: boolean
  disabled?: boolean
}

export default function ChatInput({
  onSendMessage,
  isLoading,
  disabled = false,
}: ChatInputProps) {
  const [input, setInput] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Debounced submit handler
  const handleSubmit = useCallback(() => {
    if (input.trim() && !isLoading && !disabled) {
      onSendMessage(input)
      setInput('')
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto'
      }
    }
  }, [input, isLoading, disabled, onSendMessage])

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`
    }
  }, [input])

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  return (
    <div className="border-t border-chat-gpt-border bg-chat-gpt-bg">
      <div className="mx-auto max-w-3xl px-4 py-4">
        <div className="relative flex items-end rounded-lg border border-chat-gpt-border bg-chat-gpt-bg-secondary shadow-lg">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Message Pharmaceutical Assistant..."
            disabled={disabled || isLoading}
            rows={1}
            className="max-h-[200px] w-full resize-none bg-transparent px-4 py-3 text-chat-gpt-text placeholder:text-chat-gpt-text-secondary focus:outline-none disabled:cursor-not-allowed disabled:opacity-50"
            style={{
              overflow: 'hidden',
            }}
          />
          <button
            onClick={handleSubmit}
            disabled={!input.trim() || isLoading || disabled}
            className="mb-2 mr-2 rounded-lg bg-chat-gpt-accent p-2 text-white transition-colors hover:bg-chat-gpt-accent/90 disabled:cursor-not-allowed disabled:opacity-50"
            aria-label="Send message"
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
                d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
              />
            </svg>
          </button>
        </div>
        <p className="mt-2 text-center text-xs text-chat-gpt-text-secondary">
          Press Enter to send, Shift+Enter for new line
        </p>
      </div>
    </div>
  )
}

