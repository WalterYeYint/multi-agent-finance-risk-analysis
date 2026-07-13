import React from 'react';
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import { LineChart, Cpu } from 'lucide-react';
import Landing from './pages/Landing';
import TickerView from './pages/TickerView';
import { useModel } from './hooks/useModel';
import './App.css';

/** N9: badge showing which LLM is producing analyses, so users can gauge trust. */
// eslint-disable-next-line no-unused-vars -- temporarily unmounted in Header; drop this line when re-enabling
function ModelBadge() {
  const { model } = useModel();
  if (!model?.label) return null;
  return (
    <span
      className="model-badge"
      title={`Analyses are produced by ${model.label}`}
    >
      <Cpu size={13} />
      <span>{model.label}</span>
    </span>
  );
}

function Header() {
  return (
    <header className="app-header">
      <Link to="/" className="app-header__brand">
        <LineChart size={20} />
        <span>finance-research</span>
      </Link>
      <span className="app-header__spacer" />
      {/* Temporarily hidden — re-enable by uncommenting. */}
      {/* <ModelBadge /> */}
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
