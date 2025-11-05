export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  isError?: boolean
}

export interface ChatSession {
  id: string
  title: string
  createdAt: string
  updatedAt: string
  messages: Message[]
}

export interface ChatRequest {
  message: string
  conversation_history?: Array<{
    role: string
    content: string
  }>
  user_id?: string
}

export interface ChatResponse {
  response: string
  latency_ms?: number
}

