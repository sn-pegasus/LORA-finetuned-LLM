export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  isError?: boolean
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

