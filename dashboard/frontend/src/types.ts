export type RiskLevel = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'

export type Classification =
  | 'Industrial Fire'
  | 'Gas Flare'
  | 'Agricultural Burning'
  | 'Wildfire'
  | 'Mining Activity'
  | 'Other / Unknown'

export interface EventRecord {
  id: string
  classification: Classification
  risk: RiskLevel
  score: number
  frp: number
  persistence: string
  activeDays?: number
  detectionCount?: number
  location: string
  status: string
  latitude?: number
  longitude?: number
  modelConfidence?: number
  explanation?: string
  detectedAt?: string
  satelliteDate?: string
  nearestFacility?: string
  distanceKm?: number
  probabilities?: Record<string, number>
  riskFactors?: Record<string, number>
}

export interface AlertRecord { id: string; eventId: string; level: RiskLevel; title: string; location: string; event: EventRecord }

export interface ShapExplanation { id: string; classification: string; confidence: number; positive: string; negative: string; source: string }

export interface DashboardPayload {
  source: string
  rowCount: number
  kpis: Array<{ label: string; value: string; detail: string; trend: string; tone: string }>
  timeline: { labels: string[]; values: number[] }
  classificationDistribution: Array<{ label: string; value: number; color: string }>
  riskDistribution: Array<{ label: string; value: number; count: string; color: string }>
  events: EventRecord[]
  mapPoints: Array<{ id: string; latitude: number; longitude: number; risk: RiskLevel; classification: string; activeDays?: number; frp?: number }>
}
