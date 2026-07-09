import React from 'react';
import ReactDOM from 'react-dom/client';
import axios from 'axios';
import './index.css';
import App from './App';

// I1: when the backend enforces API-key auth (API_KEYS set), the deployed
// frontend must present a key to start an analysis. Baked in at build time via
// REACT_APP_API_KEY; unset in local dev (backend auth is off by default), so
// this is a no-op there. Note: a key shipped in a public SPA bundle is not a
// true secret — its value is enabling per-key rate limiting/revocation and
// blocking casual/anonymous abuse, not confidentiality.
if (process.env.REACT_APP_API_KEY) {
  axios.defaults.headers.common['X-API-Key'] = process.env.REACT_APP_API_KEY;
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
