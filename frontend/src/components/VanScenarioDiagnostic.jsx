/**
 * VanScenarioDiagnostic.jsx
 *
 *专门用来回答一个问题:
 * "Ketika scene fisik berubah drastis (van terbawa air),
 *  apakah sistem observasi benar-benar respond?"
 *
 * Panel ini TIDAK mengutak-atik confidence.
 * Hanya menampilkan raw metrics supaya kita bisa lihat
 * apakah measurement system mengamati perubahan,
 * atau hanya confidence-nya yang di-stamp 80% tanpa dasar.
 */
import { useState, useEffect, useRef } from 'react'

/**
 * Tampilkan perubahan SCENE vs perubahan METRIC secara berdampingan.
 * Ini kunci untuk diagnose: van terbawa air
 *
 * Columns:
 * - FRAME INDEX
 * - WATERLINE_RAW (from detection)
 * - WATERLINE_SMOOTHED (from temporal)
 * - DETECTION_SCORE (from evidence.detection)
 * - TEMPORAL_SCORE (from evidence.temporal)
 * - STABILITY_SCORE (from evidence.stability)
 * - PLUMBING_SCORE (calibration * lighting)
 * - MEASUREMENT_CONF (final)
 * - RISK
 * - STATE
 */
export function MetricTimeSeries({ history, maxRows = 20 }) {
  if (!history || history.length === 0) {
    return (
      <div className="debug-section van-section">
        <div className="debug-section-title van-title">
          METRIC TIME SERIES
          <span className="van-subtitle">— Does system RESPOND to scene changes?</span>
        </div>
        <div className="van-empty">No history recorded yet. Running...</div>
      </div>
    )
  }

  const rows = history.slice(-maxRows)

  return (
    <div className="debug-section van-section">
      <div className="debug-section-title van-title">
        METRIC TIME SERIES
        <span className="van-subtitle">— Does system RESPOND to scene changes?</span>
      </div>
      <div className="van-table-wrapper">
        <table className="van-table">
          <thead>
            <tr>
              <th>#</th>
              <th>WL_raw</th>
              <th>WL_smooth</th>
              <th>det</th>
              <th>temp</th>
              <th>stab</th>
              <th>calib</th>
              <th>CONF</th>
              <th>RISK</th>
              <th>STATE</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => {
              const isLast = i === rows.length - 1
              const wlRaw = row.waterline_y_raw ?? row.waterline_y ?? 0
              const wlSmooth = row.waterline_y_smooth ?? row.smoothed ?? 0
              const wlDelta = Math.abs(wlRaw - wlSmooth)

              // Detect if scene changed: big jump in raw waterline
              const prevRow = i > 0 ? rows[i - 1] : null
              const rawDelta = prevRow
                ? Math.abs((row.waterline_y_raw ?? row.waterline_y ?? 0) - (prevRow.waterline_y_raw ?? prevRow.waterline_y ?? 0))
                : 0

              return (
                <tr
                  key={row.frame ?? i}
                  className={[
                    isLast ? 'van-row-current' : '',
                    rawDelta > 10 ? 'van-row-scene-change' : '',
                    row.state === 'NO_DETECTION' ? 'van-row-no-detection' : ''
                  ].join(' ')}
                >
                  <td className="van-td-frame">{row.frame ?? i}</td>
                  <td className="van-td-number">
                    {wlRaw > 0 ? wlRaw.toFixed(1) : '—'}
                    {rawDelta > 10 && <span className="van-delta">+{rawDelta.toFixed(1)}</span>}
                  </td>
                  <td className="van-td-number">
                    {wlSmooth > 0 ? wlSmooth.toFixed(1) : '—'}
                    {wlDelta > 5 && <span className="van-gap">gap:{wlDelta.toFixed(1)}</span>}
                  </td>
                  <td className="van-td-score">{pct(row.detection_score ?? row.evidence?.detection)}</td>
                  <td className="van-td-score">{pct(row.temporal_score ?? row.evidence?.temporal)}</td>
                  <td className="van-td-score">{pct(row.stability_score ?? row.evidence?.stability)}</td>
                  <td className="van-td-score">{pct(row.calibration_score ?? row.evidence?.calibration)}</td>
                  <td className="van-td-conf">
                    <span className={confColor(row.confidence ?? row.measurement_confidence)}>
                      {pct(row.confidence ?? row.measurement_confidence)}
                    </span>
                  </td>
                  <td className={`van-td-risk van-risk-${(row.risk ?? 'SAFE').toLowerCase()}`}>
                    {row.risk ?? 'SAFE'}
                  </td>
                  <td className="van-td-state">{row.state ?? '—'}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <div className="van-legend">
        <span className="van-legend-item"><span className="van-row-scene-change" style={{display:'inline-block',width:12,height:12}}></span> Scene changed (&gt;10px jump)</span>
        <span className="van-legend-item"><span className="van-row-no-detection" style={{display:'inline-block',width:12,height:12}}></span> No detection</span>
        <span className="van-legend-item"><span className="van-row-current" style={{display:'inline-block',width:12,height:12}}></span> Current frame</span>
        <span className="van-legend-item"><span className="van-gap"></span> gap = |raw - smooth|</span>
      </div>
    </div>
  )
}

/**
 * Komparasi METRIC vs PHYSICAL CHANGE
 *
 * Tampilkan:
 * 1. Physical metric: raw waterline delta over last N frames
 * 2. Derived metric: rate_of_change
 * 3. Measurement confidence — apakah ikut berubah?
 * 4. Evidence scores — apakah ikut berubah?
 *
 * Pertanyaan: Kalau physical berubah tapi confidence tetep,
 * berarti confidence itu nggak sensitif.
 */
export function MetricDeltaAnalysis({ current, history }) {
  if (!current || !history || history.length < 5) {
    return null
  }

  // Physical change: raw waterline delta
  const lastN = Math.min(10, history.length)
  const recent = history.slice(-lastN)

  const rawYValues = recent
    .map(r => r.waterline_y_raw ?? r.waterline_y ?? null)
    .filter(v => v !== null)

  const smoothYValues = recent
    .map(r => r.waterline_y_smooth ?? r.smoothed ?? null)
    .filter(v => v !== null)

  const rawDelta = rawYValues.length >= 2
    ? rawYValues[rawYValues.length - 1] - rawYValues[0]
    : null

  const smoothDelta = smoothYValues.length >= 2
    ? smoothYValues[smoothYValues.length - 1] - smoothYValues[0]
    : null

  // Evidence scores over time
  const detScores = recent.map(r => r.detection_score ?? r.evidence?.detection ?? null).filter(v => v !== null)
  const tempScores = recent.map(r => r.temporal_score ?? r.evidence?.temporal ?? null).filter(v => v !== null)
  const stabScores = recent.map(r => r.stability_score ?? r.evidence?.stability ?? null).filter(v => v !== null)

  const avgDet = detScores.length > 0 ? detScores.reduce((a, b) => a + b, 0) / detScores.length : null
  const avgTemp = tempScores.length > 0 ? tempScores.reduce((a, b) => a + b, 0) / tempScores.length : null
  const avgStab = stabScores.length > 0 ? stabScores.reduce((a, b) => a + b, 0) / stabScores.length : null

  const detRange = detScores.length >= 2
    ? Math.max(...detScores) - Math.min(...detScores)
    : 0
  const tempRange = tempScores.length >= 2
    ? Math.max(...tempScores) - Math.min(...tempScores)
    : 0

  // Key diagnostic: did evidence SCORES change when physical changed?
  const physicalChanged = rawDelta !== null && Math.abs(rawDelta) > 10
  const evidenceResponded = physicalChanged && (detRange > 0.1 || tempRange > 0.1)

  const currentConf = current.confidence ?? current.measurement_confidence ?? 0
  const evidenceAvg = (avgDet ?? 0) * 0.35 + (avgTemp ?? 0) * 0.35 + (avgStab ?? 0) * 0.3

  return (
    <div className="debug-section van-section">
      <div className="debug-section-title van-title">
        PHYSICAL vs METRIC
        <span className="van-subtitle">— Is evidence tied to observation?</span>
      </div>

      <div className="van-delta-grid">
        <div className="van-delta-card van-card-physical">
          <div className="van-card-label">PHYSICAL</div>
          <div className="van-card-metric">
            {rawDelta !== null ? (
              <>
                <span className={rawDelta < -5 ? 'van-change-down' : rawDelta > 5 ? 'van-change-up' : ''}>
                  {rawDelta > 0 ? '+' : ''}{rawDelta.toFixed(1)}
                </span>
                <span className="van-card-unit">px raw (last {lastN} frames)</span>
              </>
            ) : '—'}
          </div>
          <div className="van-card-sub">
            smoothed: {smoothDelta !== null ? `${smoothDelta > 0 ? '+' : ''}${smoothDelta.toFixed(1)}` : '—'}
          </div>
          <div className={`van-card-status ${physicalChanged ? 'van-status-changed' : 'van-status-stable'}`}>
            {physicalChanged ? 'SCENE CHANGED' : 'STABLE'}
          </div>
        </div>

        <div className="van-delta-card van-card-metric-eval">
          <div className="van-card-label">EVIDENCE SCORES</div>
          <div className="van-card-metric">
            <span>det: {avgDet !== null ? pct(avgDet) : '—'}</span>
            <span>temp: {avgTemp !== null ? pct(avgTemp) : '—'}</span>
            <span>stab: {avgStab !== null ? pct(avgStab) : '—'}</span>
          </div>
          <div className="van-card-sub">
            det range: {detRange.toFixed(3)} | temp range: {tempRange.toFixed(3)}
          </div>
          <div className={`van-card-status ${evidenceResponded ? 'van-status-responsive' : 'van-status-blind'}`}>
            {evidenceResponded ? 'EVIDENCE RESPONDED' : 'EVIDENCE FLAT'}
          </div>
        </div>

        <div className="van-delta-card van-card-confidence">
          <div className="van-card-label">MEASUREMENT CONF</div>
          <div className="van-card-metric">
            <span className={confColor(currentConf)}>{pct(currentConf)}</span>
          </div>
          <div className="van-card-sub">
            derived from evidence: ~{pct(evidenceAvg)}
          </div>
          <div className="van-card-status van-status-info">
            RAW OUTPUT
          </div>
        </div>

        <div className="van-delta-card van-card-sensitivity">
          <div className="van-card-label">SENSITIVITY</div>
          <div className="van-card-sensitivity-text">
            {physicalChanged && !evidenceResponded ? (
              <span className="van-insensitive">
                PHYSICAL CHANGED but EVIDENCE SCORES did NOT respond.
                Confidence may be capped or insensitive.
              </span>
            ) : physicalChanged && evidenceResponded ? (
              <span className="van-sensitive">
                EVIDENCE SCORES responded to physical change.
                System is observing scene changes.
              </span>
            ) : !physicalChanged ? (
              <span className="van-neutral">
                Scene stable. No physical change detected.
              </span>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  )
}

/**
 * Evidence Component Breakdown
 * Tampilkan semua komponen score secara flat, tidak digabung.
 */
export function EvidenceComponents({ evidence, measurement_confidence, risk_confidence }) {
  if (!evidence) return null

  const components = [
    { key: 'detection', label: 'Detection Score', raw: evidence.detection },
    { key: 'temporal', label: 'Temporal Score', raw: evidence.temporal },
    { key: 'stability', label: 'Buffer Stability', raw: evidence.stability },
    { key: 'calibration', label: 'Calibration', raw: evidence.calibration },
    { key: 'lighting', label: 'Lighting', raw: evidence.lighting },
    { key: 'plausibility', label: 'Plausibility', raw: evidence.plausibility },
  ]

  const PRODUCT = components.reduce((acc, c) => acc * (c.raw ?? 1), 1)

  return (
    <div className="debug-section van-section">
      <div className="debug-section-title van-title">
        EVIDENCE BREAKDOWN
        <span className="van-subtitle">— What actually forms confidence?</span>
      </div>
      <div className="van-evidence-grid">
        {components.map(({ key, label, raw }) => {
          const pctVal = raw != null ? Math.round(raw * 100) : 0
          const color = pctVal >= 80 ? '#22c55e' : pctVal >= 50 ? '#f59e0b' : '#ef4444'
          return (
            <div key={key} className="van-evidence-row">
              <span className="van-evidence-label">{label}</span>
              <div className="van-evidence-bar-wrap">
                <div className="van-evidence-bar" style={{ width: `${pctVal}%`, backgroundColor: color }} />
              </div>
              <span className="van-evidence-value" style={{ color }}>
                {pctVal}%
              </span>
            </div>
          )
        })}
      </div>
      <div className="van-evidence-product">
        Raw product (all ×): <span className="van-product-val">{PRODUCT.toFixed(3)}</span>
        <span className="van-evidence-note">
          Final CONF: <span className={confColor(measurement_confidence)}>{pct(measurement_confidence)}</span>
          {' | risk_conf: '}
          <span>{pct(risk_confidence)}</span>
        </span>
      </div>
    </div>
  )
}

/**
 * Raw vs Smoothed Comparison
 * Tampilkan bedanya waterline_raw dan waterline_smoothed.
 * Kalau gap besar, smoothed terlalu lambat.
 */
export function RawVsSmoothed({ current, history }) {
  if (!current) return null

  const raw = current.waterline_y_raw ?? current.detection?.waterline_y ?? null
  const smooth = current.waterline_y_smooth ?? current.smoothed ?? current.temporal?.waterline_y ?? null
  const rawConf = current.detection?.confidence ?? null

  const gap = raw !== null && smooth !== null ? Math.abs(raw - smooth) : null
  const gapSeverity = gap !== null
    ? gap < 3 ? 'small'
    : gap < 10 ? 'medium'
    : 'large'
    : 'none'

  // Historical gap trend
  const histGaps = (history || []).slice(-15).map(r => {
    const rRaw = r.waterline_y_raw ?? r.waterline_y ?? null
    const rSmooth = r.waterline_y_smooth ?? r.smoothed ?? null
    return rRaw !== null && rSmooth !== null ? Math.abs(rRaw - rSmooth) : null
  }).filter(g => g !== null)

  const avgGap = histGaps.length > 0
    ? histGaps.reduce((a, b) => a + b, 0) / histGaps.length
    : null

  const maxGap = histGaps.length > 0 ? Math.max(...histGaps) : null

  return (
    <div className="debug-section van-section">
      <div className="debug-section-title van-title">
        RAW vs SMOOTHED
        <span className="van-subtitle">— Does smoothing lag behind reality?</span>
      </div>
      <div className="van-raw-grid">
        <div className="van-raw-item">
          <span className="van-raw-label">Raw (from detection)</span>
          <span className="van-raw-value">
            {raw !== null ? raw.toFixed(1) : '—'}
            <span className="van-raw-unit">px</span>
          </span>
          <span className="van-raw-sub">det_conf: {rawConf !== null ? pct(rawConf) : '—'}</span>
        </div>
        <div className="van-raw-item">
          <span className="van-raw-label">Smoothed (from temporal)</span>
          <span className="van-raw-value">
            {smooth !== null ? smooth.toFixed(1) : '—'}
            <span className="van-raw-unit">px</span>
          </span>
          <span className="van-raw-sub">temporal_conf: {pct(current.temporal?.confidence)}</span>
        </div>
        <div className={`van-raw-item van-raw-gap van-gap-${gapSeverity}`}>
          <span className="van-raw-label">Gap |raw - smooth|</span>
          <span className="van-raw-value">
            {gap !== null ? gap.toFixed(1) : '—'}
            <span className="van-raw-unit">px</span>
          </span>
          <span className="van-raw-sub">
            avg: {avgGap !== null ? avgGap.toFixed(1) : '—'} | max: {maxGap !== null ? maxGap.toFixed(1) : '—'}
          </span>
        </div>
      </div>
      {gapSeverity === 'large' && (
        <div className="van-warning">
          LARGE GAP: smoothed level lags significantly behind raw detection.
          During rapid scene change (van), this may cause delayed risk response.
        </div>
      )}
    </div>
  )
}

// Helper functions
function pct(value) {
  if (value == null) return '—'
  return Math.round(value * 100) + '%'
}

function confColor(conf) {
  if (conf == null) return '#fff'
  if (conf >= 0.7) return '#22c55e'
  if (conf >= 0.4) return '#f59e0b'
  return '#ef4444'
}

// Compose the full diagnostic panel
export function VanScenarioDiagnostic({ data, history = [], isVideoMode }) {
  if (!data) {
    return (
      <div className="cv-debug-panel van-scenario">
        <div className="debug-panel-header van-header">
          <span>VAN SCENARIO DIAGNOSTIC</span>
          <span className="van-badge">RAW METRICS</span>
        </div>
        <div className="van-no-data">No data received yet</div>
      </div>
    )
  }

  // Extract what we need from the data structure
  const evidence = data.evidence || {}
  const temporal = data.temporal || {}
  const detection = data.detection || {}
  const diagnostics = data.diagnostics || {}
  const measurement = data.measurement || {}

  const current = {
    frame: data.frame_index,
    waterline_y_raw: detection.waterline_y,
    waterline_y_smooth: temporal.waterline_y,
    detection_score: evidence.detection,
    temporal_score: evidence.temporal,
    stability_score: evidence.stability,
    calibration_score: evidence.calibration,
    lighting_score: evidence.lighting,
    plausibility_score: evidence.plausibility,
    confidence: measurement.confidence ?? data.risk_confidence,
    measurement_confidence: measurement.confidence,
    risk_confidence: data.risk_confidence,
    risk: data.risk,
    state: diagnostics.state,
    smoothed: temporal.waterline_y,
  }

  return (
    <div className="cv-debug-panel van-scenario">
      <div className="debug-panel-header van-header">
        <span>VAN SCENARIO DIAGNOSTIC</span>
        <span className="van-badge">RAW METRICS</span>
        <span className="van-frame-badge">Frame {data.frame_index}</span>
      </div>

      <div className="van-panels">
        <MetricTimeSeries history={history} maxRows={25} />
        <MetricDeltaAnalysis current={current} history={history} />
        <EvidenceComponents
          evidence={evidence}
          measurement_confidence={measurement.confidence}
          risk_confidence={data.risk_confidence}
        />
        <RawVsSmoothed current={current} history={history} />
      </div>
    </div>
  )
}
