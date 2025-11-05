'use client'

import { memo, useMemo } from 'react'
import MessageBubble from './MessageBubble'
import { Message } from '@/types/chat'
import LoadingDots from './LoadingDots'

interface MessageListProps {
  messages: Message[]
  isLoading: boolean
}

const MessageList = memo(function MessageList({ messages, isLoading }: MessageListProps) {
  // Memoize visible messages for virtualization (showing last 50 messages)
  const visibleMessages = useMemo(() => {
    return messages.slice(-50)
  }, [messages])

  return (
    <div className="mx-auto w-full max-w-3xl px-4 py-6">
      {visibleMessages.map((message) => (
        <MessageBubble key={message.id} message={message} />
      ))}
      {isLoading && (
        <div className="mb-4 flex items-start gap-4">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-chat-gpt-accent">
            <svg
              className="h-5 w-5 text-white"
              fill="currentColor"
              viewBox="0 0 24 24"
            >
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" />
            </svg>
          </div>
          <div className="flex-1 rounded-lg bg-chat-gpt-bg-secondary px-4 py-3">
            <LoadingDots />
          </div>
        </div>
      )}
    </div>
  )
})

export default MessageList

