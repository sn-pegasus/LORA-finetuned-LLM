# Quick Start - ChatGPT-like UI

## 🚀 Fast Setup (5 minutes)

### 1. Install Frontend Dependencies
```bash
cd frontend
npm install
```

### 2. Install Backend Dependencies
```bash
# From project root
pip install -r requirements.txt
```

### 3. Start Backend Server
```bash
# From project root
python api_server.py
```
⏱️ Wait 60-90 seconds for model to load (first time only)

### 4. Start Frontend
```bash
# In a new terminal, from frontend directory
cd frontend
npm run dev
```

### 5. Open Browser
Navigate to: **http://localhost:3000**

## ✅ You're Done!

The UI includes:
- ✅ ChatGPT-like dark theme
- ✅ Real-time chat interface
- ✅ Markdown support
- ✅ Mobile responsive
- ✅ Optimized performance

## 📝 What Changed?

**New Files:**
- `api_server.py` - FastAPI backend server
- `frontend/` - Complete Next.js application
- `UI_SETUP.md` - Detailed setup guide

**Updated Files:**
- `requirements.txt` - Added FastAPI dependencies

**Existing Functionality:**
- ✅ All existing features preserved
- ✅ `chat.py` still works (CLI interface)
- ✅ Continuous learning system intact

## 🎨 Features

- **Modern UI**: ChatGPT-inspired design
- **Fast**: Optimized with React memoization
- **Responsive**: Works on all devices
- **Type-Safe**: Full TypeScript support

## 🔧 Troubleshooting

**Port 3000 in use?**
```bash
npm run dev -- -p 3001
```

**Backend not connecting?**
- Check `api_server.py` is running
- Verify port 8000 is available
- Check backend logs for errors

**Model loading issues?**
- First load takes 60-90 seconds (normal)
- Ensure 8GB+ RAM available
- Check Python dependencies installed

## 📚 More Info

See `UI_SETUP.md` for detailed documentation.

---

**Note**: The original `chat.py` CLI interface still works! The UI is an additional option.

