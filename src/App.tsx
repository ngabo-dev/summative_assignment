import React, { useState, useEffect } from 'react';
import { Sidebar } from '../components/Sidebar';
import { Dashboard } from '../components/Dashboard';
import { SinglePrediction } from '../components/SinglePrediction';
import { BulkUpload } from '../components/BulkUpload';
import { DataInsights } from '../components/DataInsights';
import { ModelManagement } from '../components/ModelManagement';
import { SystemMonitoring } from '../components/SystemMonitoring';
import { Button } from '../components/ui/button';
import { Sun, Moon, AlertTriangle, RefreshCw, Wifi, WifiOff } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '../components/ui/alert';

type Page = 'dashboard' | 'single-prediction' | 'bulk-upload' | 'data-insights' | 'model-management' | 'system-monitoring';

// API configuration
const API_BASE_URL = 'http://localhost:5000/api';

export default function App() {
  const [currentPage, setCurrentPage] = useState<Page>('dashboard');
  const [darkMode, setDarkMode] = useState(false);
  const [apiStatus, setApiStatus] = useState<'online' | 'offline' | 'checking'>('checking');
  const [lastHealthCheck, setLastHealthCheck] = useState<Date>(new Date());

  // Check API health on component mount
  useEffect(() => {
    checkApiHealth();
    // Check API health every 30 seconds
    const interval = setInterval(checkApiHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const checkApiHealth = async () => {
    setApiStatus('checking');
    
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout
      
      const response = await fetch(`${API_BASE_URL}/health`, {
        signal: controller.signal,
        headers: {
          'Accept': 'application/json',
        }
      });
      
      clearTimeout(timeoutId);
      
      if (response.ok) {
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
          const healthData = await response.json();
          setApiStatus('online');
          // Only log when successfully connecting (not every health check)
          if (apiStatus !== 'online') {
            console.log('✅ Flask backend connected:', healthData);
          }
        } else {
          setApiStatus('offline');
        }
      } else {
        setApiStatus('offline');
      }
      
      setLastHealthCheck(new Date());
      
    } catch (error) {
      setApiStatus('offline');
      setLastHealthCheck(new Date());
      
      // Log connection errors for debugging
      if (apiStatus !== 'offline') {
        console.error('❌ Flask backend connection failed:', error instanceof Error ? error.message : 'Unknown error');
      }
    }
  };

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
    document.documentElement.classList.toggle('dark');
  };

  const renderPage = () => {
    // Pass API configuration to all components
    const pageProps = {
      apiBaseUrl: API_BASE_URL,
      apiStatus
    };

    switch (currentPage) {
      case 'dashboard':
        return <Dashboard {...pageProps} />;
      case 'single-prediction':
        return <SinglePrediction {...pageProps} />;
      case 'bulk-upload':
        return <BulkUpload {...pageProps} />;
      case 'data-insights':
        return <DataInsights {...pageProps} />;
      case 'model-management':
        return <ModelManagement {...pageProps} />;
      case 'system-monitoring':
        return <SystemMonitoring {...pageProps} />;
      default:
        return <Dashboard {...pageProps} />;
    }
  };

  const getStatusColor = () => {
    switch (apiStatus) {
      case 'online': return 'bg-green-500';
      case 'offline': return 'bg-red-500';
      case 'checking': return 'bg-yellow-500 animate-pulse';
    }
  };

  const getStatusText = () => {
    switch (apiStatus) {
      case 'online': return 'Connected';
      case 'offline': return 'Disconnected';
      case 'checking': return 'Connecting...';
    }
  };

  const getStatusIcon = () => {
    switch (apiStatus) {
      case 'online': return <Wifi className="w-4 h-4 text-green-600" />;
      case 'offline': return <WifiOff className="w-4 h-4 text-red-600" />;
      case 'checking': return <RefreshCw className="w-4 h-4 text-yellow-600 animate-spin" />;
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="flex">
        <Sidebar currentPage={currentPage} onPageChange={setCurrentPage} />
        
        <main className="flex-1 ml-64">
          <header className="bg-card border-b border-border px-6 py-4 flex justify-between items-center">
            <div>
              <h1 className="text-2xl font-medium">Flower Classification ML Platform</h1>
              <div className="flex items-center gap-4">
                <p className="text-muted-foreground">Production ML Dashboard</p>
                <div className="flex items-center gap-2">
                  {getStatusIcon()}
                  <span className="text-sm text-muted-foreground">
                    {getStatusText()}
                  </span>
                  {apiStatus !== 'checking' && (
                    <button
                      onClick={checkApiHealth}
                      className="text-muted-foreground hover:text-foreground p-1"
                      title="Refresh connection status"
                    >
                      <RefreshCw className="w-3 h-3" />
                    </button>
                  )}
                </div>
              </div>
            </div>
            
            <Button
              variant="outline"
              size="sm"
              onClick={toggleDarkMode}
              className="flex items-center gap-2"
            >
              {darkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
              {darkMode ? 'Light' : 'Dark'}
            </Button>
          </header>
          
          <div className="p-6">
            {apiStatus === 'offline' && (
              <Alert className="mb-6 border-red-200 bg-red-50 dark:border-red-900 dark:bg-red-900/20">
                <AlertTriangle className="h-4 w-4 text-red-600" />
                <AlertTitle className="text-red-800 dark:text-red-200">
                  Backend Connection Required
                </AlertTitle>
                <AlertDescription className="text-red-700 dark:text-red-300">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <div className="font-medium mb-2">🔧 Setup Instructions</div>
                      <div className="text-sm space-y-1">
                        <div>• Navigate to the <code className="bg-red-200 dark:bg-red-900 px-1 rounded">backend/</code> directory</div>
                        <div>• Install dependencies: <code className="bg-red-200 dark:bg-red-900 px-1 rounded">pip install -r requirements.txt</code></div>
                        <div>• Start the Flask server: <code className="bg-red-200 dark:bg-red-900 px-1 rounded">python app.py</code></div>
                        <div>• Ensure the server is running on port 5000</div>
                      </div>
                    </div>
                    <div>
                      <div className="font-medium mb-2">🔍 Troubleshooting</div>
                      <div className="text-sm space-y-1">
                        <div>• Check if Flask server is running</div>
                        <div>• Verify port 5000 is not blocked</div>
                        <div>• Check console for connection errors</div>
                        <div>• Ensure CORS is properly configured</div>
                      </div>
                    </div>
                  </div>
                  <div className="flex justify-between items-center mt-4 pt-3 border-t border-red-300 dark:border-red-700 text-xs">
                    <span>Last attempt: {lastHealthCheck.toLocaleTimeString()}</span>
                    <button 
                      onClick={checkApiHealth}
                      className="text-red-800 dark:text-red-200 hover:underline font-medium"
                    >
                      Retry Connection
                    </button>
                  </div>
                </AlertDescription>
              </Alert>
            )}
            
            {renderPage()}
          </div>
        </main>
      </div>
    </div>
  );
}