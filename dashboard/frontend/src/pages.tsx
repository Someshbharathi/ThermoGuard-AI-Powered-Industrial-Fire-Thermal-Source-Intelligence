import { useEffect, useState } from 'react'
import { Activity, AlertTriangle, BarChart3, Flame, LoaderCircle, Radio, Search, Settings, Sparkles } from 'lucide-react'
import { ClassificationBadge, EmptyModule, RiskBadge } from './components/ui'
import { OverviewDashboard } from './components/dashboard'
import { api } from './services/api'
import type { DashboardPayload } from './types'

export function OverviewPage() {
  const [data, setData] = useState<DashboardPayload | null>(null)
  const [error, setError] = useState('')
  useEffect(() => { api.getOverview().then(setData).catch((cause: Error) => setError(cause.message)) }, [])
  if (error) return <div className="data-state"><AlertTriangle size={22} /><h2>Data service unavailable</h2><p>{error}. Start the FastAPI service and refresh this view.</p></div>
  if (!data) return <div className="data-state"><LoaderCircle className="spin" size={24} /><h2>Loading verified outputs</h2><p>Reading the final fusion risk dataset.</p></div>
  return <OverviewDashboard data={data} />
}

export function EventsPage() {
  const [search, setSearch] = useState('')
  const [risk, setRisk] = useState('')
  const [items, setItems] = useState<import('./types').EventRecord[]>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)
  useEffect(() => {
    const params = new URLSearchParams({ limit: '100' })
    if (search) params.set('search', search)
    if (risk) params.set('risk', risk)
    api.getEvents(`?${params.toString()}`).then((result) => { setItems(result.items); setTotal(result.total) }).finally(() => setLoading(false))
  }, [search, risk])
  return <div className="events-page"><div className="module-kicker"><Activity size={14} /> VERIFIED EVENT CATALOGUE</div><div className="events-toolbar"><div><h2>Model-scored thermal events</h2><p>{total.toLocaleString()} records from the final risk output</p></div><div className="event-filters"><label><Search size={15} /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search events..." /></label><select value={risk} onChange={(event) => setRisk(event.target.value)} aria-label="Filter by risk"><option value="">All risk levels</option><option value="CRITICAL">Critical</option><option value="HIGH">High</option><option value="MEDIUM">Medium</option><option value="LOW">Low</option></select></div></div><div className="panel full-events"><div className="table-scroll"><table><thead><tr><th>Event ID</th><th>Classification</th><th>Risk</th><th>Score</th><th>Confidence</th><th>FRP</th><th>Persistence</th><th>Coordinates</th><th>Status</th></tr></thead><tbody>{loading ? <tr><td colSpan={9}>Loading verified events...</td></tr> : items.map((event) => <tr key={event.id}><td className="event-id">{event.id}</td><td><ClassificationBadge value={event.classification} /></td><td><RiskBadge level={event.risk} /></td><td>{event.score.toFixed(2)}</td><td>{event.modelConfidence?.toFixed(2) ?? '-'}%</td><td>{event.frp} MW</td><td>{event.persistence}</td><td>{event.location}</td><td>{event.status}</td></tr>)}</tbody></table></div><div className="table-note">Source: results/risk/thermoguard_risk_scores.csv · filters query the API</div></div></div>
}

const modules = {
  map: { title: 'Live Map', icon: Radio, description: 'Explore thermal anomalies, industrial context, and persistence clusters across the analysis area.' },
  events: { title: 'Events', icon: Flame, description: 'Review, filter, and investigate the normalized thermal event catalogue.' },
  alerts: { title: 'Alerts', icon: AlertTriangle, description: 'Prioritize high-confidence signals and track response workflows.' },
  analytics: { title: 'Analytics', icon: BarChart3, description: 'Compare risk, persistence, classification, and model performance.' },
  insights: { title: 'AI Insights', icon: Sparkles, description: 'Surface explainable model findings with SHAP-backed intelligence.' },
  system: { title: 'System', icon: Settings, description: 'Monitor data sources, model versions, and dashboard integration health.' },
} as const

export function ModulePage({ module }: { module: keyof typeof modules }) { const item = modules[module]; const Icon = item.icon; return <div className="module-page"><div className="module-kicker"><Activity size={14} /> COMMAND MODULE / PHASE 1</div><EmptyModule icon={<Icon size={25} />} title={item.title} description={item.description} /></div> }
