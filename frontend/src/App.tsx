import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import MainLayout from './layouts/MainLayout';
import RiskAssessment from './pages/RiskAssessment';
import Monitoring from './pages/Monitoring';
import ModelInfo from './pages/ModelInfo';
import SystemStatus from './pages/SystemStatus';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<MainLayout />}>
          <Route index element={<RiskAssessment />} />
          <Route path="monitoring" element={<Monitoring />} />
          <Route path="model" element={<ModelInfo />} />
          <Route path="system" element={<SystemStatus />} />
        </Route>
      </Routes>
    </Router>
  );
}

export default App;
