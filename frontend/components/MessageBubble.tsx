'use client'

import { memo } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Message } from '@/types/chat'

interface MessageBubbleProps {
  message: Message
}

const MessageBubble = memo(function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user'
  const isError = message.isError

  return (
    <div
      className={`mb-4 flex items-start gap-4 ${
        isUser ? 'flex-row-reverse' : ''
      }`}
    >
      <div
        className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full ${
          isUser
            ? 'bg-chat-gpt-bg-tertiary'
            : isError
            ? 'bg-red-600'
            : 'bg-chat-gpt-accent'
        }`}
      >
        {isUser ? (
          <svg
            className="h-5 w-5 text-chat-gpt-text"
            fill="currentColor"
            viewBox="0 0 24 24"
          >
            <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z" />
          </svg>
        ) : (
          <svg
            className="h-5 w-5 text-white"
            fill="currentColor"
            viewBox="0 0 24 24"
          >
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" />
          </svg>
        )}
      </div>
      <div
        className={`flex-1 rounded-lg px-4 py-3 ${
          isUser
            ? 'bg-chat-gpt-bg-secondary text-chat-gpt-text'
            : isError
            ? 'bg-red-900/20 text-red-400'
            : 'bg-chat-gpt-bg-secondary text-chat-gpt-text'
        }`}
      >
        <div className="prose prose-invert max-w-none">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            className="text-chat-gpt-text whitespace-pre-wrap"
            components={{
              p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
              code: ({ children, className }) => {
                const isInline = !className
                return isInline ? (
                  <code className="rounded bg-chat-gpt-bg-tertiary px-1.5 py-0.5 text-sm text-chat-gpt-accent">
                    {children}
                  </code>
                ) : (
                  <code className="block rounded-lg bg-chat-gpt-bg-tertiary p-4 text-sm">
                    {children}
                  </code>
                )
              },
              pre: ({ children }) => (
                <pre className="mb-2 overflow-x-auto rounded-lg bg-chat-gpt-bg-tertiary p-4">
                  {children}
                </pre>
              ),
              ul: ({ children }) => (
                <ul className="mb-2 ml-4 list-disc space-y-1">{children}</ul>
              ),
              ol: ({ children }) => (
                <ol className="mb-2 ml-4 list-decimal space-y-1">{children}</ol>
              ),
              li: ({ children }) => <li className="text-chat-gpt-text">{children}</li>,
              a: ({ href, children }) => (
                <a
                  href={href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-chat-gpt-accent underline hover:text-chat-gpt-accent/80"
                >
                  {children}
                </a>
              ),
            }}
          >
            {message.content}
          </ReactMarkdown>
        </div>
      </div>
    </div>
  )
})

export default MessageBubble

