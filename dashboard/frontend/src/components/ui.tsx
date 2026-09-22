import type { ReactNode } from 'react'
import type { Classification, RiskLevel } from '../types'

export function Card({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <section className={`panel ${className}`}>{children}</section>
}

export function SectionHeader({ eyebrow, title, action }: { eyebrow?: string; title: string; action?: ReactNode }) {
  return <div className="section-header"><div>{eyebrow && <span className="eyebrow">{eyebrow}</span>}<h2>{title}</h2></div>{action}</div>
}

export function RiskBadge({ level, score }: { level: RiskLevel; score?: number }) {
  return <span className={`badge risk-${level.toLowerCase()}`}>{level}{score !== undefined && <strong>{score.toFixed(1)}</strong>}</span>
}

export function ClassificationBadge({ value }: { value: Classification }) {
  return <span className="classification-badge"><i />{value}</span>
}

export function EmptyModule({ icon, title, description }: { icon: ReactNode; title: string; description: string }) {
  return <div className="module-empty"><div className="empty-icon">{icon}</div><h2>{title}</h2><p>{description}</p><span className="ready-label">MODULE READY FOR INTEGRATION</span></div>
}
