import type { AlertRecord, DashboardPayload, EventRecord, ShapExplanation } from '../types'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

export async function apiRequest<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options,
  })
  if (!response.ok) throw new Error(`API request failed: ${response.status}`)
  return response.json() as Promise<T>
}

export const api = {
  getOverview: () => apiRequest<DashboardPayload>('/api/overview'),
  getEvents: (query = '') => apiRequest<{ items: EventRecord[]; total: number }>(`/api/events${query}`),
  getEvent: (id: string) => apiRequest<EventRecord>(`/api/events/${id}`),
  getAlerts: () => apiRequest<{ items: AlertRecord[] }>('/api/alerts'),
  getShapExplanation: (id: string) => apiRequest<ShapExplanation>(`/api/shap/${id}`),
  getAnalytics: () => apiRequest<unknown>('/api/analytics'),
}
