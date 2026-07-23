import React, { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import LandingPage from './components/logic/LandingPage';
import Homepage from './components/logic/homepage';
import Dashboard from './components/logic/Dashboard';
import PredictForm from '../PredictForm'
import Recent from './components/logic/Recent'
import StatementUpload from './components/logic/StatementUpload'
const RouteTitleUpdater = () => {
  const location = useLocation();

  useEffect(() => {
    const routeToTitle = {
      '/': 'ShieldPay - AI Fraud Protection',
      '/dashboard': 'ShieldPay - Dashboard',
      '/send-money': 'ShieldPay - Send Money',
      '/transactions': 'ShieldPay - Transactions',
      '/statements': 'ShieldPay - Statements',
      '/beneficiaries': 'ShieldPay - Beneficiaries',
      '/settings': 'ShieldPay - Settings',
      '/help-support': 'ShieldPay - Help & Support',
    };

    const title = routeToTitle[location.pathname] || 'ShieldPay';
    document.title = title;
  }, [location]);

  return null; // This component does not render anything
};

const App = () => {
  return (
    <Router>
      <RouteTitleUpdater />
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/send-money" element={<Homepage />} />
        <Route path="/transactions" element={<Recent />} />
        <Route path="/statements" element={<StatementUpload />} />
        <Route path="/beneficiaries" element={<Homepage />} />
        <Route path="/settings" element={<PredictForm />} />
        <Route path="/help-support" element={<Homepage />} />
      </Routes>
    </Router>
    
  );
};

export default App;
