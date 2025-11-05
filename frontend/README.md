# Offline Chatbot Frontend

A modern, ChatGPT-like UI built with Next.js, React, and TypeScript for the offline pharmaceutical chatbot.

## Features

- 🎨 ChatGPT-inspired dark theme UI
- ⚡ Optimized rendering with React memoization
- 📱 Responsive design for mobile and desktop
- 💬 Real-time chat interface
- 🎯 Markdown support for formatted responses
- 🔄 Loading states and error handling
- 📊 Message virtualization for performance

## Setup

1. Install dependencies:
```bash
npm install
# or
yarn install
```

2. Create a `.env.local` file (optional):
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

3. Run the development server:
```bash
npm run dev
# or
yarn dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Building for Production

```bash
npm run build
npm start
```

## Project Structure

```
frontend/
├── app/
│   ├── globals.css       # Global styles
│   ├── layout.tsx        # Root layout
│   └── page.tsx          # Main page
├── components/
│   ├── ChatInterface.tsx # Main chat container
│   ├── ChatInput.tsx     # Input component
│   ├── MessageBubble.tsx # Individual message
│   ├── MessageList.tsx   # Message container
│   ├── Sidebar.tsx       # Side navigation
│   └── LoadingDots.tsx   # Loading animation
└── types/
    └── chat.ts           # TypeScript types
```

## Optimizations

- **React.memo**: Components are memoized to prevent unnecessary re-renders
- **Message Virtualization**: Only renders last 50 messages for performance
- **Lazy Loading**: Components load on demand
- **Debounced Input**: Prevents excessive API calls

## Technologies

- Next.js 14 (App Router)
- React 18
- TypeScript
- Tailwind CSS
- React Markdown

