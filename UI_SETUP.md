# UI Setup Guide - ChatGPT-like Interface

This guide will help you set up and run the new Next.js frontend UI for your offline chatbot.

## Prerequisites

1. **Node.js**: Version 18 or higher
   - Download from: https://nodejs.org/
   - Verify installation: `node --version`

2. **Python Backend**: The FastAPI server must be running
   - See backend setup below

## Quick Start

### Step 1: Install Frontend Dependencies

Navigate to the frontend directory:
```bash
cd frontend
npm install
```

### Step 2: Install Backend Dependencies

From the project root, install Python dependencies:
```bash
pip install -r requirements.txt
```

### Step 3: Start the Backend Server

From the project root:
```bash
python api_server.py
```

The backend will start on `http://localhost:8000` and load the model (this takes 60-90 seconds on first run).

### Step 4: Start the Frontend

In a new terminal, from the frontend directory:
```bash
cd frontend
npm run dev
```

The frontend will start on `http://localhost:3000`.

### Step 5: Open in Browser

Open your browser and navigate to:
```
http://localhost:3000
```

## Configuration

### Backend API URL

By default, the frontend connects to `http://localhost:8000`. To change this:

1. Create a `.env.local` file in the `frontend` directory:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

2. Or modify `frontend/components/ChatInterface.tsx`:
```typescript
const API_URL = 'http://your-api-url:8000'
```

## Features

### ChatGPT-like UI

- **Dark Theme**: Matches ChatGPT's color scheme
- **Message Bubbles**: Clean, modern message display
- **Sidebar**: Navigation and chat history
- **Responsive**: Works on desktop and mobile
- **Markdown Support**: Formatted responses with code blocks, lists, etc.

### Optimizations

- **Fast Rendering**: Memoized components for performance
- **Message Virtualization**: Only renders visible messages
- **Smooth Scrolling**: Auto-scrolls to latest message
- **Loading States**: Visual feedback during API calls

## Troubleshooting

### Frontend won't start

**Error**: `Cannot find module 'next'`
- **Solution**: Run `npm install` in the frontend directory

**Error**: Port 3000 already in use
- **Solution**: Change port: `npm run dev -- -p 3001`

### Backend Connection Issues

**Error**: `Failed to fetch` or connection refused
- **Solution**: Make sure `api_server.py` is running on port 8000
- Check: `curl http://localhost:8000/health`

**Error**: `Model not loaded`
- **Solution**: Wait for the model to finish loading (60-90 seconds)
- Check backend logs for loading progress

### Model Loading Issues

**Error**: Model takes too long to load
- **Solution**: This is normal on first run. Subsequent runs are faster
- Ensure you have enough RAM (8GB+ recommended)

**Error**: Out of memory
- **Solution**: Close other applications
- Consider using a smaller model or CPU-only mode

## Development

### Making Changes

1. **Frontend Changes**: Edit files in `frontend/` directory
   - Changes auto-reload with Next.js hot reload
   
2. **Backend Changes**: Edit `api_server.py`
   - Restart the server to apply changes

### Building for Production

**Frontend**:
```bash
cd frontend
npm run build
npm start
```

**Backend**:
```bash
# Use a production WSGI server like gunicorn
pip install gunicorn
gunicorn api_server:app -w 4 -k uvicorn.workers.UvicornWorker
```

## Architecture

### Frontend (Next.js)
- **Framework**: Next.js 14 with App Router
- **Styling**: Tailwind CSS
- **Language**: TypeScript
- **Components**: React 18

### Backend (FastAPI)
- **Framework**: FastAPI
- **Model**: LLaMA 3.2-3B with LoRA adapter
- **Server**: Uvicorn
- **API**: REST endpoints for chat

## API Endpoints

- `GET /health` - Health check
- `POST /api/chat` - Send message and get response
- `POST /api/chat/stream` - Streaming response (SSE)
- `POST /api/feedback` - Submit feedback

## Next Steps

1. **Customize UI**: Edit components in `frontend/components/`
2. **Add Features**: Implement new functionality in the frontend
3. **Optimize Model**: Fine-tune the model for better responses
4. **Deploy**: Set up production deployment

## Support

For issues or questions:
- Check the backend logs for errors
- Verify all dependencies are installed
- Ensure model files are in the correct location
- Check network connectivity between frontend and backend

---

**Note**: The first startup will take longer as the model loads into memory. Subsequent requests are faster.

