import { useState, useEffect, useRef, useCallback } from 'react'

const WS_URL = 'ws://localhost:8000/ws'
const MAX_HISTORY = 60

export function useFloodWebSocket() {
  const [connected, setConnected] = useState(false)
  const [data, setData] = useState(null)
  const [history, setHistory] = useState([])
  const [connectionStatus, setConnectionStatus] = useState('disconnected')

  const wsRef = useRef(null)
  const reconnectTimeoutRef = useRef(null)
  const reconnectAttempts = useRef(0)

  const addToHistory = useCallback((reading) => {
    setHistory(prev => {
      const newHistory = [...prev, reading]
      if (newHistory.length > MAX_HISTORY) {
        return newHistory.slice(-MAX_HISTORY)
      }
      return newHistory
    })
  }, [])

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    setConnectionStatus('connecting')
    console.log('[WebSocket] Connecting to', WS_URL)

    try {
      wsRef.current = new WebSocket(WS_URL)

      wsRef.current.onopen = () => {
        console.log('[WebSocket] Connected')
        setConnected(true)
        setConnectionStatus('connected')
        reconnectAttempts.current = 0
      }

      wsRef.current.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data)
          handleMessage(message)
        } catch (err) {
          console.error('[WebSocket] Parse error:', err)
        }
      }

      wsRef.current.onclose = () => {
        console.log('[WebSocket] Disconnected')
        setConnected(false)
        setConnectionStatus('disconnected')
        scheduleReconnect()
      }

      wsRef.current.onerror = (error) => {
        console.error('[WebSocket] Error:', error)
        setConnectionStatus('error')
      }
    } catch (err) {
      console.error('[WebSocket] Connection error:', err)
      setConnectionStatus('error')
      scheduleReconnect()
    }
  }, [])

  const scheduleReconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }
    const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 16000)
    reconnectAttempts.current += 1
    console.log(`[WebSocket] Reconnecting in ${delay}ms...`)
    reconnectTimeoutRef.current = setTimeout(connect, delay)
  }, [connect])

  const handleMessage = useCallback((message) => {
    if (message.type === 'connected' || message.type === 'reading') {
      const reading = message.data || message
      setData(reading)
      addToHistory({
        timestamp: new Date().toLocaleTimeString(),
        raw: reading.processed?.rawWaterLevel || reading.state?.waterLevel || 0,
        smoothed: reading.processed?.smoothedWaterLevel || reading.state?.smoothedLevel || 0,
        confidence: reading.processed?.confidence || reading.state?.confidence || 0,
        rateOfChange: reading.processed?.rateOfChange || reading.state?.rateOfChange || 0,
        risk: reading.processed?.risk || reading.state?.risk || 'SAFE',
        readingsProcessed: reading.state?.readingsProcessed || 0,
        bufferSize: reading.state?.bufferSize || 0,
        nodeId: reading.node?.id || reading.state?.nodeId || 'NODE-001'
      })
    }
  }, [addToHistory])

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    setConnected(false)
    setConnectionStatus('disconnected')
  }, [])

  const sendCommand = useCallback((action, params = {}) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action, ...params }))
    }
  }, [])

  useEffect(() => {
    connect()
    return () => disconnect()
  }, [connect, disconnect])

  return {
    connected,
    connectionStatus,
    data,
    history,
    sendCommand,
    reconnect: connect,
    disconnect
  }
}
