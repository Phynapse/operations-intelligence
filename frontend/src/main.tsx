import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import App from './App';
import DashboardPage from './pages/DashboardPage';
import AskPage from './pages/AskPage';
import AnalysisDetailPage from './pages/AnalysisDetailPage';
import DatasetsPage from './pages/DatasetsPage';
import './index.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />}>
          <Route index element={<DashboardPage />} />
          <Route path="ask" element={<AskPage />} />
          <Route path="analysis/:id" element={<AnalysisDetailPage />} />
          <Route path="datasets" element={<DatasetsPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </StrictMode>
);
