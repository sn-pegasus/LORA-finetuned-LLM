import { NextRequest, NextResponse } from 'next/server'

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000'

export async function GET(_request: NextRequest) {
  try {
    const res = await fetch(`${BACKEND_URL}/health`, { cache: 'no-store' })
    if (!res.ok) {
      const text = await res.text()
      return NextResponse.json({ error: 'Backend health failed', details: text }, { status: res.status })
    }
    const data = await res.json()
    return NextResponse.json(data)
  } catch (error) {
    return NextResponse.json({ error: 'Health check error', details: error instanceof Error ? error.message : 'Unknown' }, { status: 503 })
  }
}


