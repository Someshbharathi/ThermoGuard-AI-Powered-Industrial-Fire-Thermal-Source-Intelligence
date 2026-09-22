import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { Factory, Satellite } from 'lucide-react'
import { AlertsPage, LightLayout, ModernEventsPage, OverviewModern, PersistentPage, PlaceholderPage, ReportsPage, RiskPage } from './modern'

function App() {
  return <BrowserRouter><Routes><Route element={<LightLayout />}><Route path="/" element={<OverviewModern />} /><Route path="/events" element={<ModernEventsPage />} /><Route path="/persistent" element={<PersistentPage />} /><Route path="/industrial" element={<PlaceholderPage title="Industrial Areas" icon={Factory} description="Explore model-associated industrial signals and infrastructure proximity across the analysis area." />} /><Route path="/satellite" element={<PlaceholderPage title="Satellite Imagery" icon={Satellite} description="Inspect Sentinel-2 context for selected thermal records when imagery service integration is enabled." />} /><Route path="/risk" element={<RiskPage />} /><Route path="/alerts" element={<AlertsPage />} /><Route path="/reports" element={<ReportsPage />} /><Route path="*" element={<OverviewModern />} /></Route></Routes></BrowserRouter>
}

export default App
