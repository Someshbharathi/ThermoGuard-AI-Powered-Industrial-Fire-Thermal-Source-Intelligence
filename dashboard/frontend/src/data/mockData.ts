import type { EventRecord } from '../types'

// DEMO DATA - replace with FastAPI responses when integration begins.
export const kpis = [
  { label: 'Total thermal events', value: '1,130,374', detail: '2024 India FIRMS observations', trend: '+8.4%', tone: 'cyan' },
  { label: 'Industrial candidates', value: '69,025', detail: 'AI candidate detections', trend: '+12.1%', tone: 'amber' },
  { label: 'High / critical risk', value: '82', detail: 'Current analysis set', trend: 'Requires review', tone: 'red' },
  { label: 'Persistent sources', value: '13,600', detail: 'Long-duration thermal activity', trend: '+4.7%', tone: 'green' },
]

export const thermalTimeline = [34, 42, 39, 55, 47, 61, 58, 76, 71, 85, 79, 93, 88, 104, 98, 118, 110, 126, 121, 138, 133, 149, 143, 160]
export const timelineLabels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
export const classificationDistribution = [
  { label: 'Industrial', value: 42, color: '#ef7d32' },
  { label: 'Mining', value: 24, color: '#d6a34d' },
  { label: 'Natural', value: 19, color: '#55b8a4' },
  { label: 'Other / Unknown', value: 15, color: '#64748b' },
]
export const riskDistribution = [
  { label: 'Low', value: 49, count: '33,830', color: '#55b8a4' },
  { label: 'Medium', value: 28, count: '19,321', color: '#d6a34d' },
  { label: 'High', value: 17, count: '11,727', color: '#ef7d32' },
  { label: 'Critical', value: 6, count: '4,147', color: '#df5d5d' },
]
export const recentEvents: EventRecord[] = [
  { id: 'TG-24-08142', classification: 'Industrial Fire', risk: 'CRITICAL', score: 91.8, frp: 74.2, persistence: '18 days', location: 'Korba, Chhattisgarh', status: 'Review' },
  { id: 'TG-24-07988', classification: 'Gas Flare', risk: 'HIGH', score: 78.4, frp: 56.8, persistence: '31 days', location: 'Jamnagar, Gujarat', status: 'Escalated' },
  { id: 'TG-24-07731', classification: 'Mining Activity', risk: 'MEDIUM', score: 54.2, frp: 32.1, persistence: '12 days', location: 'Singrauli, Madhya Pradesh', status: 'Monitored' },
  { id: 'TG-24-07404', classification: 'Agricultural Burning', risk: 'LOW', score: 22.6, frp: 18.3, persistence: '2 days', location: 'Bathinda, Punjab', status: 'Cleared' },
]
export const demoMapPoints = [
  { x: 29, y: 27, type: 'industrial', label: 'DEMO-01' },
  { x: 46, y: 48, type: 'thermal', label: 'DEMO-02' },
  { x: 55, y: 68, type: 'thermal', label: 'DEMO-03' },
  { x: 67, y: 40, type: 'industrial', label: 'DEMO-04' },
  { x: 72, y: 58, type: 'thermal', label: 'DEMO-05' },
  { x: 37, y: 75, type: 'thermal', label: 'DEMO-06' },
]
