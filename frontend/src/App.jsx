import React from 'react';
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import { LineChart } from 'lucide-react';
import Landing from './pages/Landing';
import TickerView from './pages/TickerView';
import './App.css';

function Header() {
  return (
    <header className="app-header">
      <Link to="/" className="app-header__brand">
        <LineChart size={20} />
        <span>finance-research</span>
      </Link>
      <span className="app-header__tag">research tool · not investment advice</span>
    </header>
  );
}

function App() {
  return (
    <BrowserRouter>
      <Header />
      <main className="app-main">
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/t/:ticker" element={<TickerView />} />
          <Route path="*" element={<Landing />} />
        </Routes>
      </main>
    </BrowserRouter>
  );
}

export default App;
