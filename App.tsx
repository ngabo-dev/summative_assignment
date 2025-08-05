import React from 'react';

export default function App() {
  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif', backgroundColor: '#f8f9fa' }}>
      <div style={{ maxWidth: '800px', margin: '0 auto', backgroundColor: 'white', padding: '40px', borderRadius: '8px', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}>
        <h1 style={{ color: '#2c3e50', marginBottom: '20px' }}>🚀 Flower Classification ML Platform</h1>
        
        <div style={{ backgroundColor: '#e8f5e8', border: '1px solid #28a745', padding: '20px', borderRadius: '6px', marginBottom: '30px' }}>
          <h2 style={{ color: '#155724', margin: '0 0 10px 0', fontSize: '18px' }}>✅ Vite Setup Complete</h2>
          <p style={{ color: '#155724', margin: '0' }}>Your application has been successfully converted to Vite for optimal performance.</p>
        </div>

        <div style={{ backgroundColor: '#fff3cd', border: '1px solid #ffc107', padding: '20px', borderRadius: '6px', marginBottom: '30px' }}>
          <h2 style={{ color: '#856404', margin: '0 0 15px 0', fontSize: '18px' }}>⚡ Start Development Server</h2>
          <div style={{ backgroundColor: '#2c3e50', color: '#ecf0f1', padding: '15px', borderRadius: '4px', fontFamily: 'monospace', fontSize: '14px', marginBottom: '15px' }}>
            npm run dev
          </div>
          <p style={{ color: '#856404', margin: '0' }}>
            Run the command above to start the Vite development server and access your ML dashboard.
          </p>
        </div>

        <div style={{ backgroundColor: '#d1ecf1', border: '1px solid #17a2b8', padding: '20px', borderRadius: '6px', marginBottom: '30px' }}>
          <h2 style={{ color: '#0c5460', margin: '0 0 15px 0', fontSize: '18px' }}>🔧 Backend Setup</h2>
          <p style={{ color: '#0c5460', marginBottom: '10px' }}>Don't forget to start your Flask backend:</p>
          <div style={{ backgroundColor: '#2c3e50', color: '#ecf0f1', padding: '10px', borderRadius: '4px', fontFamily: 'monospace', fontSize: '14px' }}>
            cd backend<br />
            pip install -r requirements.txt<br />
            python app.py
          </div>
        </div>

        <div style={{ backgroundColor: '#f8d7da', border: '1px solid #dc3545', padding: '20px', borderRadius: '6px' }}>
          <h2 style={{ color: '#721c24', margin: '0 0 10px 0', fontSize: '18px' }}>📁 File Location</h2>
          <p style={{ color: '#721c24', margin: '0' }}>
            The main application is now located at: <code style={{ backgroundColor: '#f5c6cb', padding: '2px 6px', borderRadius: '3px' }}>/src/App.tsx</code>
          </p>
        </div>
      </div>
    </div>
  );
}