import { useState, useRef, useEffect } from 'react'

const SIGNAL_COLORS = {
  edge: '#22d3ee',
  color: '#a78bfa',
  texture: '#34d399'
}

const STATE_COLORS = {
  NO_DETECTION: { color: '#71717a', bg: 'rgba(113,113,122,0.1)' },
  CANDIDATE: { color: '#f59e0b', bg: 'rgba(245,158,11,0.1)' },
  DETECTED: { color: '#3b82f6', bg: 'rgba(59,130,246,0.1)' },
  STABLE: { color: '#22c55e', bg: 'rgba(34,197,94,0.1)' },
  UNSTABLE: { color: '#ef4444', bg: 'rgba(239,68,68,0.1)' },
  REJECTED: { color: '#dc2626', bg: 'rgba(220,38,38,0.1)' },
  CALIBRATION_INVALID: { color: '#9333ea', bg: 'rgba(147,51,234,0.1)' },
  UNCERTAIN: { color: '#f97316', bg: 'rgba(249,115,22,0.1)' },
  SIMULATOR: { color: '#06b6d4', bg: 'rgba(6,182,212,0.1)' }
}

const EVIDENCE_LABELS = {
  detection: 'Detection',
  temporal: 'Temporal Stability',
  stability: 'Buffer Stability',
  calibration: 'Calibration',
  lighting: 'Lighting',
  plausibility: 'Plausibility'
}

export function EvidenceBreakdown({ evidence }) {
  if (!evidence || Object.keys(evidence).length === 0) {
    return null
  }
  return (
    <div className="debug-section">
      <div className="debug-section-title">Evidence Components</div>
      <div className="evidence-grid">
        {Object.entries(evidence).map(([key, value]) => {
          const percentage = Math.round(value * 100)
          const color = percentage >= 80 ? '#22c55e' : percentage >= 50 ? '#f59e0b' : '#ef4444'
          return (
            <div key={key} className="evidence-item">
              <div className="evidence-label">{EVIDENCE_LABELS[key] || key}</div>
              <div className="evidence-bar-container">
                <div className="evidence-bar" style={{ width: `${percentage}%`, backgroundColor: color }} />
              </div>
              <div className="evidence-value" style={{ color }}>{percentage}%</div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export function DetectionStatePanel({ diagnostics }) {
  if (!diagnostics) return null
  const state = diagnostics.state || 'UNKNOWN'
  const stateStyle = STATE_COLORS[state] || STATE_COLORS.NO_DETECTION
  return (
    <div className="debug-section">
      <div className="debug-section-title">Detection State</div>
      <div className="state-badge" style={{ color: stateStyle.color, backgroundColor: stateStyle.bg, borderColor: stateStyle.color }}>
        {state.replace(/_/g, ' ')}
      </div>
    </div>
  )
}

export function CandidatePanel({ candidates }) {
  if (!candidates || candidates.length === 0) {
    return (
      <div className="debug-section">
        <div className="debug-section-title">Candidate Detections</div>
        <div className="no-candidates">No candidates detected</div>
      </div>
    )
  }
  return (
    <div className="debug-section">
      <div className="debug-section-title">Candidate Detections ({candidates.length})</div>
      <div className="candidates-list">
        {candidates.map((c, i) => (
          <div key={i} className={`candidate-row ${c.selected ? 'candidate-selected' : ''}`}>
            <div className="candidate-method">{c.method.toUpperCase()}</div>
            <div className="candidate-details">
              <span>y={c.waterline_y?.toFixed(1)}</span>
              <span>conf={Math.round(c.confidence * 100)}%</span>
              <span>q={Math.round(c.quality * 100)}%</span>
              {c.selected && <span className="candidate-selected-label">SELECTED</span>}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export function DiagnosticsPanel({ diagnostics }) {
  if (!diagnostics) return null
  const { state, reasons = [], permitted_inferences = [], blocked_inferences = [], calibration_status, calibration_valid, calibration_method } = diagnostics
  return (
    <div className="debug-section">
      <div className="debug-section-title">Why Accepted / Rejected</div>
      {reasons.length === 0 ? (
        <div className="diag-ok">No rejection reasons — measurement accepted</div>
      ) : (
        <div className="reasons-list">
          {reasons.map((r, i) => (
            <div key={i} className="reason-item">
              <span className="reason-bullet">•</span>
              {r}
            </div>
          ))}
        </div>
      )}
      {calibration_status && (
        <div className="calibration-info">
          <span>Calibration: {calibration_method}</span>
          <span className={calibration_valid ? 'text-risk-safe' : 'text-risk-warning'}>
            {calibration_valid ? 'VALID' : 'INVALID'} — {calibration_status}
          </span>
        </div>
      )}
      {permitted_inferences.length > 0 && (
        <div className="inference-section">
          <div className="inference-label permitted">Permitted:</div>
          <div className="inference-list">{permitted_inferences.join(', ')}</div>
        </div>
      )}
      {blocked_inferences.length > 0 && (
        <div className="inference-section">
          <div className="inference-label blocked">Blocked:</div>
          <div className="inference-list">{blocked_inferences.join(', ')}</div>
        </div>
      )}
    </div>
  )
}

export function SignalVisualization({ signals, frameWidth = 300, frameHeight = 200 }) {
  if (!signals) return null
  const { edge, color, texture, roi, brightness } = signals
  const panels = []
  if (edge) panels.push({ key: 'edge', label: 'Edge', data: edge, color: SIGNAL_COLORS.edge })
  if (color) panels.push({ key: 'color', label: 'Color', data: color, color: SIGNAL_COLORS.color })
  if (texture) panels.push({ key: 'texture', label: 'Texture', data: texture, color: SIGNAL_COLORS.texture })
  if (panels.length === 0) return null
  const panelW = Math.min(frameWidth, 280)
  const panelH = 60
  return (
    <div className="debug-section">
      <div className="debug-section-title">Signal Profiles</div>
      {roi && (
        <div className="roi-info">ROI: y[{roi.y_min}–{roi.y_max}] x[{roi.x_min}–{roi.x_max}] | brightness: {brightness?.toFixed(0)}</div>
      )}
      <div className="signals-container">
        {panels.map(({ key, label, data, color }) => {
          if (!data || !data.data || data.data.length === 0) return null
          const dataLen = data.data.length - 1
          const absMax = Math.max(1, ...data.data.map(Math.abs))
          const points = data.data.map((v, i) =>
            `${(i / dataLen) * panelW},${panelH - (v / absMax) * panelH * 0.9}`
          ).join(' ')
          const peakX = dataLen > 0 ? (data.peak_idx / dataLen) * panelW : 0
          return (
            <div key={key} className="signal-panel">
              <div className="signal-label" style={{ color }}>
                {label} | peak={data.peak_value?.toFixed(1)} idx={data.peak_idx}
              </div>
              <svg width={panelW} height={panelH} className="signal-svg">
                <polyline
                  points={points}
                  fill="none"
                  stroke={color}
                  strokeWidth="1.5"
                />
                {data.peak_idx !== undefined && (
                  <line
                    x1={peakX} y1={0}
                    x2={peakX} y2={panelH}
                    stroke={color}
                    strokeWidth="1"
                    strokeDasharray="2,2"
                    opacity="0.5"
                  />
                )}
              </svg>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export function TemporalHistoryPanel({ temporal, frameIndex }) {
  if (!temporal) return null
  const history = temporal.detection_history || []
  if (history.length < 2) return null
  const graphH = 80
  const graphW = 280
  const validIndices = history.map((detected, i) => detected ? i : -1).filter(i => i >= 0)
  if (validIndices.length < 2) {
    return (
      <div className="debug-section">
        <div className="debug-section-title">Temporal History</div>
        <div className="no-history">Insufficient detection history</div>
        <div className="history-stats">
          <span>valid: {temporal.valid_detections}</span>
          <span>invalid: {temporal.invalid_detections}</span>
          <span>rate: {(temporal.detection_rate || 0).toFixed(2)}</span>
        </div>
      </div>
    )
  }
  const lastN = history.slice(-30)
  const svgPoints = lastN.map((detected, i) => {
    const x = (i / (lastN.length - 1)) * graphW
    const y = detected ? graphH - 4 : graphH / 2
    return `${x},${y}`
  }).join(' ')
  const currentX = graphW
  return (
    <div className="debug-section">
      <div className="debug-section-title">
        Temporal History ({history.length} frames)
      </div>
      <div className="history-stats">
        <span className={temporal.trend === 'STABLE' ? 'text-risk-safe' : temporal.trend === 'NO_DETECTION' ? 'text-text-muted' : 'text-risk-warning'}>
          {temporal.trend}
        </span>
        <span>rate: {(temporal.rate_of_change || 0).toFixed(2)} px/s</span>
        <span>valid: {temporal.valid_detections}</span>
        <span>invalid: {temporal.invalid_detections}</span>
        <span>det_rate: {(temporal.detection_rate || 0).toFixed(2)}</span>
      </div>
      <svg width={graphW} height={graphH} className="history-svg">
        {lastN.map((detected, i) => (
          <circle
            key={i}
            cx={(i / (lastN.length - 1)) * graphW}
            cy={detected ? graphH - 4 : graphH / 2}
            r={detected ? 3 : 4}
            fill={detected ? '#22d3ee' : '#ef4444'}
            opacity={detected ? 0.8 : 0.5}
          />
        ))}
        <line x1={currentX} y1={0} x2={currentX} y2={graphH} stroke="#fff" strokeWidth="1" opacity="0.3" />
      </svg>
    </div>
  )
}

export function FrameControls({ isPaused, onStep, onReset, onTogglePause, overlays, onToggleOverlay, isVideoMode }) {
  return (
    <div className="debug-section">
      <div className="debug-section-title">Frame Controls</div>
      <div className="controls-row">
        <button onClick={onTogglePause} className={`debug-btn ${isPaused ? 'btn-primary' : 'btn-secondary'}`}>
          {isPaused ? 'Resume' : 'Pause'}
        </button>
        <button onClick={onStep} className="debug-btn btn-secondary" disabled={!isPaused}>
          Step
        </button>
        <button onClick={onReset} className="debug-btn btn-secondary">
          Reset
        </button>
      </div>
      {isVideoMode && (
        <div className="overlay-toggles">
          <div className="overlay-toggle-label">Overlays:</div>
          {Object.entries(overlays).map(([key, enabled]) => (
            <button
              key={key}
              onClick={() => onToggleOverlay(key)}
              className={`debug-btn overlay-btn ${enabled ? 'btn-primary' : 'btn-secondary'}`}
            >
              {key}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

export function VideoOverlay({ data, overlays, frameWidth, frameHeight }) {
  const { detection, candidates, signals } = data || {}
  const roi = signals?.roi
  const lines = []
  if (overlays.waterline && detection?.waterline_y) {
    const y = detection.waterline_y
    lines.push({ type: 'waterline', y, color: '#22d3ee', label: `WL ${y.toFixed(1)}px` })
  }
  if (overlays.candidates && candidates?.length > 0) {
    candidates.forEach((c, i) => {
      lines.push({
        type: 'candidate',
        y: c.waterline_y,
        color: c.selected ? '#22c55e' : '#f59e0b',
        label: `${c.method.toUpperCase()} ${Math.round(c.confidence * 100)}%`
      })
    })
  }
  if (overlays.roi && roi) {
    lines.push({
      type: 'roi',
      x1: roi.x_min, y1: roi.y_min,
      x2: roi.x_max, y2: roi.y_max,
      color: '#a78bfa',
      label: `ROI [${roi.x_min},${roi.y_min}]-${roi.x_max},${roi.y_max}`
    })
  }
  if (lines.length === 0) return null
  const svgW = frameWidth || 320
  const svgH = frameHeight || 200
  return (
    <svg
      width={svgW}
      height={svgH}
      className="video-overlay-svg"
      style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }}
    >
      {lines.map((line, i) => {
        if (line.type === 'roi') {
          return (
            <g key={i}>
              <rect
                x={line.x1} y={line.y1}
                width={line.x2 - line.x1} height={line.y2 - line.y1}
                fill="none" stroke={line.color} strokeWidth="1.5" strokeDasharray="4,3"
              />
              <text x={line.x1 + 4} y={line.y1 + 14} fill={line.color} fontSize="10">
                {line.label}
              </text>
            </g>
          )
        }
        return (
          <g key={i}>
            <line
              x1={0} y1={line.y} x2={svgW} y2={line.y}
              stroke={line.color} strokeWidth={line.type === 'waterline' ? 2 : 1}
              strokeDasharray={line.type === 'candidate' ? '4,4' : 'none'}
            />
            <text x={8} y={line.y - 4} fill={line.color} fontSize="11" fontWeight={line.type === 'waterline' ? 'bold' : 'normal'}>
              {line.label}
            </text>
          </g>
        )
      })}
    </svg>
  )
}

export function CVDebugPanel({ data, isPaused, onStep, onReset, onTogglePause, overlays, onToggleOverlay, isVideoMode }) {
  const { diagnostics, evidence, candidates, temporal, signals, detection, frame_index } = data || {}
  return (
    <div className="cv-debug-panel">
      <div className="debug-panel-header">
        <span>CV INSTRUMENTATION</span>
        {frame_index !== undefined && <span className="frame-index">Frame {frame_index}</span>}
      </div>
      <div className="debug-panels">
        <div className="debug-col">
          <DetectionStatePanel diagnostics={diagnostics} />
          <EvidenceBreakdown evidence={evidence} />
          <CandidatePanel candidates={candidates} />
          <DiagnosticsPanel diagnostics={diagnostics} />
        </div>
        <div className="debug-col">
          <SignalVisualization signals={signals} />
          <TemporalHistoryPanel temporal={temporal} frameIndex={frame_index} />
          <FrameControls
            isPaused={isPaused}
            onStep={onStep}
            onReset={onReset}
            onTogglePause={onTogglePause}
            overlays={overlays}
            onToggleOverlay={onToggleOverlay}
            isVideoMode={isVideoMode}
          />
        </div>
      </div>
    </div>
  )
}
