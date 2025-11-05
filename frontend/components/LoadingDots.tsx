'use client'

export default function LoadingDots() {
  return (
    <div className="flex gap-1.5">
      <div className="h-2 w-2 animate-pulse rounded-full bg-chat-gpt-text-secondary"></div>
      <div
        className="h-2 w-2 animate-pulse rounded-full bg-chat-gpt-text-secondary"
        style={{ animationDelay: '0.2s' }}
      ></div>
      <div
        className="h-2 w-2 animate-pulse rounded-full bg-chat-gpt-text-secondary"
        style={{ animationDelay: '0.4s' }}
      ></div>
    </div>
  )
}

