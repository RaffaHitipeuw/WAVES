import { useState, useEffect, useRef } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts'
import {
  Droplets,
  Activity,
  AlertTriangle,
  Radio,
  Clock,
  Database,
  Cpu,
  Wifi,
  WifiOff,
  Play,
  Square,
  RefreshCw,
  CheckCircle,
  TrendingUp,
  TrendingDown,
  Minus,
  Video,
  ActivitySquare,
  AlertOctagon
} from 'lucide-react'
import { useFloodWebSocket } from './hooks/useFloodWebSocket'

const API_BASE = 'http://localhost:8000'

const RISK_CONFIG = {
  SAFE: {
    color: 'text-risk-safe',
    bgColor: 'bg-risk-safe/10 border-risk-safe/30',
    label: 'SAFE',
    message: 'Water level is within normal parameters.',
    alertClass: 'safe'
  },
  WATCH: {
    color: 'text-risk-watch',
    bgColor: 'bg-risk-watch/10 border-risk-watch/30',
    label: 'WATCH',
    message: 'Water level is elevated. Monitoring intensified.',
    alertClass: 'watch'
  },
  WARNING: {
    color: 'text-risk-warning',
    bgColor: 'bg-risk-warning/10 border-risk-warning/30',
    label: 'WARNING',
    message: 'Rapid water-level increase detected. Prepare for flooding.',
    alertClass: 'warning'
  },
  CRITICAL: {
    color: 'text-risk-critical',
    bgColor: 'bg-risk-critical/10 border-risk-critical/30',
    label: 'CRITICAL',
    message: 'Critical water level. Immediate action required.',
    alertClass: 'critical'
  }
}

const MEASUREMENT_STATUS = {
  'VALID': { color: 'text-risk-safe', label: 'VALID', icon: CheckCircle },
  'CALIBRATING': { color: 'text-risk-watch', label: 'CALIBRATING', icon: Activity },
  'LOW_CONFIDENCE': { color: 'text-risk-warning', label: 'LOW CONFIDENCE', icon: AlertTriangle },
  'UNCERTAIN': { color: 'text-risk-warning', label: 'UNCERTAIN', icon: AlertTriangle },
  'NO_VALID_WATERLINE': { color: 'text-text-muted', label: 'NO WATERLINE', icon: AlertOctagon },
  'NO_DETECTION': { color: 'text-text-muted', label: 'NO DETECTION', icon: AlertOctagon },
  'SUDDEN_CHANGE': { color: 'text-risk-critical', label: 'UNSTABLE', icon: AlertTriangle }
}

function formatNumber(num, decimals = 1) {
  if (num === null || num === undefined || isNaN(num)) return '--'
  return num.toFixed(decimals)
}

function formatTime(timestamp) {
  if (!timestamp) return '--:--:--'
  try {
    return new Date(timestamp).toLocaleTimeString()
  } catch {
    return '--:--:--'
  }
}

function StatusBar({ status, nodeId, mode, progress }) {
  return (
    <div className="status-bar">
      <div className="status-bar-logo">
        <div className="w-9 h-9 bg-accent-cyan/20 border border-accent-cyan/50 rounded flex items-center justify-center">
          <Droplets size={18} className="text-accent-cyan" />
        </div>
        <div>
          <div className="status-bar-title">HYDROSIGNAL</div>
          <div className="status-bar-subtitle">Flood Early Warning System</div>
        </div>
      </div>
      <div className="flex items-center gap-6">
        {mode === 'video' && progress !== undefined && (
          <div className="text-xs text-text-muted">
            <span className="text-accent-cyan font-medium">{Math.round(progress * 100)}%</span>
            <span className="ml-1">processed</span>
          </div>
        )}
        <div className="status-indicator">
          <div className={`status-dot ${status}`} />
          <span className="text-xs text-text-muted uppercase tracking-wider">
            {status === 'connected' ? 'LIVE' : status.toUpperCase()}
          </span>
        </div>
        <div className="text-xs text-text-muted">
          <span className="text-text-secondary font-semibold">{nodeId}</span>
        </div>
      </div>
    </div>
  )
}

function VideoMonitor({ videoRef, measurement, isPlaying, mode }) {
  const statusConfig = MEASUREMENT_STATUS[measurement?.measurementStatus] || MEASUREMENT_STATUS['NO_DETECTION']
  const StatusIcon = statusConfig.icon

  return (
    <div className="video-monitor">
      <video
        ref={videoRef}
        src="/asset.mp4"
        className="absolute inset-0 w-full h-full object-contain"
        loop
        muted
        playsInline
      />
      <div className="video-overlay" />
      <div className="video-timestamp">
        <div className="video-timestamp-badge">
          <div className="flex items-center gap-2">
            {mode === 'video' ? <Video size={12} className="text-accent-cyan" /> : <ActivitySquare size={12} className="text-accent-cyan" />}
            <span className="video-timestamp-text">
              {new Date().toLocaleTimeString()}
            </span>
          </div>
        </div>
        {measurement?.frameIndex !== undefined && (
          <div className="video-timestamp-badge">
            <span className="video-timestamp-text">
              Frame {measurement.frameIndex}
            </span>
          </div>
        )}
      </div>
      {!isPlaying && (
        <div className="absolute inset-0 flex items-center justify-center bg-bg-primary/90">
          <div className="text-center">
            <Droplets size={48} className="text-text-muted mx-auto mb-4" />
            <div className="text-text-muted text-sm">VIDEO SENSOR</div>
            <div className="text-text-muted/50 text-xs mt-1">Click START to begin</div>
          </div>
        </div>
      )}
      <div className="video-hud">
        <div className="video-hud-item">
          <div className="video-hud-label">Water Level</div>
          <div className="video-hud-value text-text-primary">
            <span className="number-update">
              {measurement?.isValid ? formatNumber(measurement.waterLevel) : '--'}
            </span>
            <span className="video-hud-unit">cm</span>
          </div>
        </div>
        <div className="video-hud-item">
          <div className="video-hud-label">Confidence</div>
          <div className={`video-hud-value ${measurement?.isValid ? 'text-accent-cyan' : 'text-text-muted'}`}>
            {measurement ? Math.round(measurement.confidence * 100) : '--'}
            <span className="video-hud-unit">%</span>
          </div>
        </div>
        <div className="video-hud-item">
          <div className="video-hud-label">Rate</div>
          <div className={`video-hud-value ${
            !measurement?.rateOfChange ? 'text-text-muted' :
            measurement.rateOfChange >= 0 ? 'text-risk-warning' : 'text-risk-safe'
          }`}>
            {measurement?.rateOfChange != null ? `${measurement.rateOfChange >= 0 ? '+' : ''}${formatNumber(measurement.rateOfChange)}` : '--'}
            <span className="video-hud-unit">cm/m</span>
          </div>
        </div>
        <div className="video-hud-item">
          <div className="video-hud-label">Detection</div>
          <div className={`video-hud-value ${statusConfig.color}`}>
            <StatusIcon size={14} className="inline mr-1" />
            {statusConfig.label}
          </div>
        </div>
      </div>
    </div>
  )
}

function PrimaryDisplay({ measurement, risk }) {
  const riskConfig = RISK_CONFIG[risk] || RISK_CONFIG.SAFE
  const waterLevel = measurement?.waterLevel
  const isValid = measurement?.isValid !== false
  const statusConfig = MEASUREMENT_STATUS[measurement?.measurementStatus] || MEASUREMENT_STATUS['NO_DETECTION']
  const StatusIcon = statusConfig.icon

  return (
    <div className="primary-display">
      <div className="primary-label">Predicted Water Level</div>
      <div className="primary-value-container">
        <span className={`primary-value ${!isValid ? 'text-text-muted' : riskConfig.color}`}>
          {isValid ? formatNumber(waterLevel) : '--'}
        </span>
        <span className="primary-unit">cm</span>
      </div>
      {!isValid && (
        <div className={`mt-3 flex items-center justify-center gap-2 ${statusConfig.color}`}>
          <StatusIcon size={14} />
          <span className="text-xs">{measurement?.measurementStatus?.replace(/_/g, ' ')}</span>
        </div>
      )}
    </div>
  )
}

function MetricCard({ label, value, unit, trend, children }) {
  return (
    <div className="metric-card">
      <div className="metric-card-label">{label}</div>
      <div className="metric-card-value">{value}</div>
      {unit && <div className="metric-card-sub">{unit}</div>}
      {trend !== undefined && trend !== null && (
        <div className={`metric-card-sub ${trend > 0 ? 'text-risk-warning' : trend < 0 ? 'text-risk-safe' : 'text-text-muted'}`}>
          {trend > 0 ? '↑ Rising' : trend < 0 ? '↓ Falling' : '→ Stable'}
        </div>
      )}
      {children}
    </div>
  )
}

function ConfidenceCard({ measurement }) {
  const percentage = measurement ? Math.round((measurement.confidence || 0) * 100) : 0
  const isValid = measurement?.isValid !== false

  const getColorClass = () => {
    if (!isValid) return 'bg-text-muted'
    if (percentage >= 90) return 'bg-risk-safe'
    if (percentage >= 70) return 'bg-risk-watch'
    if (percentage >= 50) return 'bg-risk-warning'
    return 'bg-risk-critical'
  }

  return (
    <div className="metric-card">
      <div className="metric-card-label">Confidence</div>
      <div className="flex items-baseline gap-2">
        <span className={`metric-card-value ${isValid ? 'text-accent-cyan' : 'text-text-muted'}`}>
          {percentage}
        </span>
        <span className="text-sm text-text-muted">%</span>
      </div>
      <div className="confidence-bar-container">
        <div className="confidence-bar">
          <div
            className={`confidence-fill ${getColorClass()}`}
            style={{ width: `${isValid ? percentage : 30}%` }}
          />
        </div>
      </div>
    </div>
  )
}

function RateOfRisePanel({ measurement, history }) {
  const rateOfChange = measurement?.rateOfChange || 0
  const trend = measurement?.trend || 'UNKNOWN'

  let trendColor = 'text-text-secondary'
  let trendIndicator = '→'

  if (trend === 'RISING' || trend === 'RISING_FAST') {
    trendColor = rateOfChange > 0 ? 'text-risk-warning' : 'text-text-secondary'
    trendIndicator = '↑'
  } else if (trend === 'FALLING' || trend === 'FALLING_FAST') {
    trendColor = 'text-risk-safe'
    trendIndicator = '↓'
  } else if (trend === 'STABLE') {
    trendColor = 'text-text-secondary'
    trendIndicator = '→'
  } else if (trend === 'NO_DETECTION' || trend === 'UNCERTAIN') {
    trendColor = 'text-text-muted'
    trendIndicator = '?'
  }

  return (
    <div className="rate-panel">
      <div className="panel-title mb-3">
        <Activity size={12} />
        Rate of Rise
      </div>
      <div className="flex items-end justify-between">
        <div>
          <div className={`rate-value ${rateOfChange >= 0 ? 'text-risk-warning' : 'text-risk-safe'}`}>
            {rateOfChange >= 0 ? '+' : ''}{formatNumber(rateOfChange)}
          </div>
          <div className="text-sm text-text-muted mt-1">cm per minute</div>
        </div>
        <div className="text-right">
          <div className={`rate-trend-indicator ${trendColor}`}>
            <span className="text-2xl">{trendIndicator}</span>
          </div>
          <div className={`rate-trend ${trendColor} mt-1`}>{trend.replace(/_/g, ' ')}</div>
        </div>
      </div>
    </div>
  )
}

function WaterLevelChart({ history, measurement }) {
  const thresholds = { watch: 30, warning: 50, critical: 70 }

  const allLevels = history.map(h => [h.raw, h.smoothed]).flat().filter(v => v !== null && !isNaN(v))
  const minLevel = allLevels.length > 0 ? Math.max(0, Math.min(...allLevels, 20) - 10) : 0
  const maxLevel = allLevels.length > 0 ? Math.max(...allLevels, 50) + 10 : 100

  return (
    <div className="chart-panel">
      <div className="panel-header">
        <div className="panel-title">
          <Activity size={12} />
          Water Level / Time
        </div>
      </div>
      <div className="chart-container">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={history} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f1f2a" vertical={false} />
            <XAxis
              dataKey="timestamp"
              stroke="#52525b"
              fontSize={10}
              tickLine={false}
              axisLine={false}
              interval="preserveStartEnd"
              tick={{ fill: '#71717a' }}
            />
            <YAxis
              stroke="#52525b"
              fontSize={10}
              tickLine={false}
              axisLine={false}
              domain={[minLevel, maxLevel]}
              tick={{ fill: '#71717a' }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#12121a',
                border: '1px solid #2a2a3a',
                borderRadius: '6px',
                fontSize: '11px'
              }}
              labelStyle={{ color: '#a1a1aa' }}
            />
            <ReferenceLine y={thresholds.watch} stroke="#eab308" strokeDasharray="4 4" strokeOpacity={0.4} />
            <ReferenceLine y={thresholds.warning} stroke="#f97316" strokeDasharray="4 4" strokeOpacity={0.4} />
            <ReferenceLine y={thresholds.critical} stroke="#ef4444" strokeDasharray="4 4" strokeOpacity={0.4} />
            <Line
              type="monotone"
              dataKey="raw"
              stroke="#22d3ee"
              strokeWidth={1.5}
              dot={false}
              name="Raw"
              connectNulls={false}
              animationDuration={300}
            />
            <Line
              type="monotone"
              dataKey="smoothed"
              stroke="#3b82f6"
              strokeWidth={2}
              strokeDasharray="4 2"
              dot={false}
              name="Smoothed"
              connectNulls={false}
              animationDuration={300}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="chart-legend">
        <div className="chart-legend-item">
          <div className="chart-legend-line bg-accent-cyan" />
          <span>RAW</span>
        </div>
        <div className="chart-legend-item">
          <div className="chart-legend-line bg-accent-blue" style={{ borderTopWidth: 2, borderTopStyle: 'dashed' }} />
          <span>SMOOTHED</span>
        </div>
        <div className="flex-1" />
        <div className="flex items-center gap-1">
          <div className="w-3 h-0 border-t border-dashed border-risk-watch opacity-60" />
          <span>WATCH</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-0 border-t border-dashed border-risk-warning opacity-60" />
          <span>WARN</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-0 border-t border-dashed border-risk-critical opacity-60" />
          <span>CRIT</span>
        </div>
      </div>
    </div>
  )
}

function SystemStatus({ measurement, data, mode }) {
  const state = data?.state || {}
  const isValid = measurement?.isValid !== false

  const statusItems = [
    { label: 'Connection', value: 'CONNECTED', status: 'active' },
    { label: 'Processing', value: mode === 'video' ? 'CV PIPELINE' : 'SIMULATOR', status: 'active' },
    { label: 'Detection', value: isValid ? 'VALID' : 'INVALID', status: isValid ? 'active' : 'warning' },
    { label: 'Last Update', value: state.lastReadingAt ? formatTime(state.lastReadingAt) : '--:--:--', status: 'active' },
    { label: 'Readings', value: state.readingsProcessed || 0, status: 'active' }
  ]

  return (
    <div className="system-status">
      <div className="panel-title mb-3">
        <Cpu size={12} />
        System Status
      </div>
      {statusItems.map((item, index) => (
        <div key={index} className="status-row">
          <span className="status-row-label">{item.label}</span>
          <span className={`status-row-value ${item.status === 'active' ? 'text-risk-safe' : 'text-risk-warning'}`}>
            {item.status === 'active' && <CheckCircle size={12} className="inline mr-1.5" />}
            {item.value}
          </span>
        </div>
      ))}
    </div>
  )
}

function RiskAlert({ measurement, data }) {
  const risk = data?.state?.risk || measurement?.risk || 'SAFE'
  const riskConfig = RISK_CONFIG[risk] || RISK_CONFIG.SAFE
  const isValid = measurement?.isValid !== false

  if (risk === 'SAFE' && !isValid) return null

  return (
    <div className={`risk-alert ${riskConfig.alertClass}`}>
      <div className="risk-alert-header">
        <div className="risk-alert-icon">
          <AlertTriangle size={20} />
        </div>
        <div>
          <div className={`risk-alert-title ${riskConfig.color}`}>
            {riskConfig.label}
          </div>
          <div className="risk-alert-message">
            {riskConfig.message}
          </div>
        </div>
      </div>
    </div>
  )
}

function Controls({ onStart, onStop, onReset, onModeChange, isRunning, mode, onVideoSelect }) {
  return (
    <div className="controls-panel">
      <div className="panel-title mb-3">
        <Radio size={12} />
        Processing Controls
      </div>
      <div className="controls-row">
        {!isRunning ? (
          <button onClick={onStart} className="btn btn-primary">
            <Play size={14} />
            Start
          </button>
        ) : (
          <button onClick={onStop} className="btn btn-danger">
            <Square size={14} />
            Stop
          </button>
        )}
        <button onClick={onReset} className="btn btn-secondary">
          <RefreshCw size={14} />
          Reset
        </button>
        <select
          value={mode}
          onChange={(e) => onModeChange(e.target.value)}
          className="btn-select"
          disabled={isRunning}
        >
          <option value="video">Video CV</option>
          <option value="simulator">Simulator</option>
        </select>
      </div>
      {mode === 'video' && (
        <div className="mt-3 text-xs text-text-muted">
          Using computer vision pipeline on video frames
        </div>
      )}
      {mode === 'simulator' && (
        <div className="mt-3 text-xs text-risk-watch">
          Warning: Using synthetic data from simulator
        </div>
      )}
    </div>
  )
}

function Thresholds() {
  const thresholds = [
    { label: 'WATCH', value: '30 cm', color: 'text-risk-watch' },
    { label: 'WARNING', value: '50 cm', color: 'text-risk-warning' },
    { label: 'CRITICAL', value: '70 cm', color: 'text-risk-critical' }
  ]

  return (
    <div className="panel p-4">
      <div className="panel-title mb-3">
        <Database size={12} />
        Thresholds
      </div>
      <div className="thresholds-list">
        {thresholds.map((t, i) => (
          <div key={i} className="threshold-row">
            <span className={`threshold-label ${t.color}`}>{t.label}</span>
            <span className="threshold-value">{t.value}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function App() {
  const { connected, connectionStatus, data, history } = useFloodWebSocket()
  const [isRunning, setIsRunning] = useState(false)
  const [mode, setMode] = useState('video')
  const [systemMode, setSystemMode] = useState('video')
  const videoRef = useRef(null)
  const [isVideoPlaying, setIsVideoPlaying] = useState(false)

  const nodeId = data?.node?.id || data?.state?.nodeId || 'NODE-001'
  const measurement = data?.measurement
  const state = data?.state || {}

  const waterLevel = measurement?.waterLevel ?? state.waterLevel ?? 0
  const smoothed = measurement?.smoothedLevel ?? state.smoothedLevel ?? 0
  const confidence = measurement?.confidence ?? state.confidence ?? 0
  const rateOfChange = measurement?.rateOfChange ?? state.rateOfChange ?? 0
  const risk = state.risk || 'SAFE'
  const riskConfig = RISK_CONFIG[risk] || RISK_CONFIG.SAFE
  const isValid = measurement?.isValid !== false
  const progress = data?.video?.progress

  useEffect(() => {
    if (videoRef.current && connected) {
      videoRef.current.play().then(() => {
        setIsVideoPlaying(true)
      }).catch(() => {
        setIsVideoPlaying(false)
      })
    }
  }, [connected])

  const handleStart = async () => {
    try {
      if (mode !== systemMode) {
        await fetch(`${API_BASE}/api/mode?mode=${mode}`, { method: 'POST' })
        setSystemMode(mode)
      }
      const response = await fetch(`${API_BASE}/api/start`, { method: 'POST' })
      const result = await response.json()
      if (result.status === 'started' || result.status === 'already_running') {
        setIsRunning(true)
        if (videoRef.current) {
          videoRef.current.play().catch(() => {})
          setIsVideoPlaying(true)
        }
      }
    } catch (err) {
      console.error('Start error:', err)
    }
  }

  const handleStop = async () => {
    try {
      await fetch(`${API_BASE}/api/stop`, { method: 'POST' })
      setIsRunning(false)
    } catch (err) {
      console.error('Stop error:', err)
    }
  }

  const handleReset = async () => {
    try {
      await fetch(`${API_BASE}/api/reset`, { method: 'POST' })
    } catch (err) {
      console.error('Reset error:', err)
    }
  }

  const handleModeChange = async (newMode) => {
    setMode(newMode)
  }

  return (
    <div className="app-shell">
      <div className="app-container">
        <StatusBar
          status={connectionStatus}
          nodeId={nodeId}
          mode={systemMode}
          progress={progress}
        />
        <div className="mb-6">
          <VideoMonitor
            videoRef={videoRef}
            measurement={measurement}
            isPlaying={isVideoPlaying}
            mode={systemMode}
          />
        </div>
        <div className="mb-6">
          <PrimaryDisplay measurement={measurement} risk={risk} />
        </div>
        <div className="metric-grid mb-6">
          <MetricCard
            label="Smoothed"
            value={measurement?.smoothedLevel != null ? formatNumber(measurement.smoothedLevel) : '--'}
            unit="cm"
            trend={rateOfChange}
          />
          <MetricCard
            label="Rate of Rise"
            value={rateOfChange != null ? `${rateOfChange >= 0 ? '+' : ''}${formatNumber(rateOfChange)}` : '--'}
            unit="cm/min"
            trend={rateOfChange}
          />
          <ConfidenceCard measurement={measurement} />
          <MetricCard
            label="Risk Level"
            value={riskConfig.label}
            trend={null}
          >
            <div className="mt-2">
              <span className={`risk-badge ${risk.toLowerCase()}`}>
                {riskConfig.label}
              </span>
            </div>
          </MetricCard>
        </div>
        <div className="mb-6">
          <WaterLevelChart history={history} measurement={measurement} />
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
          <RateOfRisePanel measurement={measurement} history={history} />
          <SystemStatus measurement={measurement} data={data} mode={systemMode} />
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
          <Controls
            onStart={handleStart}
            onStop={handleStop}
            onReset={handleReset}
            onModeChange={handleModeChange}
            isRunning={isRunning}
            mode={mode}
          />
          <Thresholds />
        </div>
        {(risk !== 'SAFE' || !isValid) && (
          <div className="mb-6 animate-fade-in">
            <RiskAlert measurement={measurement} data={data} />
          </div>
        )}
        <div className="app-footer">
          <div className="footer-text">
            HydroSignal v2.0 • Real-time Flood Monitoring • Computer Vision Engine
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
