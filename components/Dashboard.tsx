import React, { useState, useEffect } from 'react';
import { Card } from './ui/card';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { PerformanceChart } from './PerformanceChart';
import { PredictionChart } from './PredictionChart'; 
import { Activity, TrendingUp, Zap, Users, Server, CheckCircle, AlertTriangle, RefreshCw } from 'lucide-react';

interface DashboardMetrics {
  model_status: {
    status: string;
    uptime: number;
    version: string;
    last_trained: string;
  };
  performance: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
  };
  system: {
    cpu_usage: number;
    memory_usage: number;
    network_io: number;
    requests_per_sec: number;
    active_containers: number;
    success_rate: number;
    avg_latency: number;
  };
  predictions_24h: {
    total: number;
    by_class: {
      roses: number;
      tulips: number;
      sunflowers: number;
    };
  };
}

interface DashboardProps {
  apiBaseUrl: string;
  apiStatus: 'online' | 'offline' | 'checking';
}

export function Dashboard({ apiBaseUrl, apiStatus }: DashboardProps) {
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (apiStatus === 'online') {
      loadDashboardMetrics();
      
      // Refresh metrics every 30 seconds when API is online
      const interval = setInterval(loadDashboardMetrics, 30000);
      return () => clearInterval(interval);
    } else {
      setMetrics(null);
      setIsLoading(false);
      setError('Backend connection required');
    }
  }, [apiStatus, apiBaseUrl]);

  const loadDashboardMetrics = async () => {
    if (apiStatus !== 'online') return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${apiBaseUrl}/dashboard/metrics`, {
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('application/json')) {
        throw new Error('Invalid response format - expected JSON');
      }
      
      const data = await response.json();
      setMetrics(data);
      setError(null);
    } catch (error) {
      console.error('Error loading dashboard metrics:', error);
      setError(error instanceof Error ? error.message : 'Failed to load metrics');
      setMetrics(null);
    } finally {
      setIsLoading(false);
    }
  };

  if (apiStatus === 'offline') {
    return (
      <div className="space-y-6">
        <div className="text-center py-12">
          <AlertTriangle className="w-16 h-16 text-red-500 mx-auto mb-4" />
          <h3 className="text-xl font-medium mb-2">Backend Connection Required</h3>
          <p className="text-muted-foreground mb-6 max-w-md mx-auto">
            The dashboard requires a connection to the Flask backend server to display real-time ML metrics and system status.
          </p>
          <div className="space-y-2 text-sm text-muted-foreground max-w-md mx-auto">
            <p>1. Navigate to the <code className="bg-muted px-1 rounded">backend/</code> directory</p>
            <p>2. Run <code className="bg-muted px-1 rounded">pip install -r requirements.txt</code></p>
            <p>3. Start the server with <code className="bg-muted px-1 rounded">python app.py</code></p>
          </div>
        </div>
      </div>
    );
  }

  if (isLoading && !metrics) {
    return (
      <div className="space-y-6">
        <div className="text-center">
          <h2 className="text-2xl font-medium mb-2">Dashboard Overview</h2>
          <p className="text-muted-foreground">Loading ML system metrics...</p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {Array.from({ length: 4 }).map((_, i) => (
            <Card key={i} className="p-6">
              <div className="animate-pulse">
                <div className="h-4 bg-muted rounded w-1/2 mb-2"></div>
                <div className="h-8 bg-muted rounded w-3/4 mb-2"></div>
                <div className="h-3 bg-muted rounded w-1/3"></div>
              </div>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  if (error || !metrics) {
    return (
      <div className="text-center py-12">
        <AlertTriangle className="w-12 h-12 text-red-500 mx-auto mb-4" />
        <h3 className="text-lg font-medium mb-2">Failed to Load Dashboard</h3>
        <p className="text-muted-foreground mb-4">
          {error || 'Unable to connect to the ML system'}
        </p>
        <button 
          onClick={loadDashboardMetrics}
          className="inline-flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90"
        >
          <RefreshCw className="w-4 h-4" />
          Retry
        </button>
      </div>
    );
  }

  const StatusIndicator = ({ status }: { status: string }) => {
    const isHealthy = status === 'healthy';
    return (
      <div className="flex items-center gap-2">
        <div className={`w-2 h-2 rounded-full ${isHealthy ? 'bg-green-500' : 'bg-red-500'}`} />
        <span className="capitalize text-sm">{status}</span>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-medium mb-2">Dashboard Overview</h2>
          <p className="text-muted-foreground">Real-time ML system performance and metrics</p>
        </div>
        
        <Badge variant="outline" className="text-green-600 border-green-600">
          <CheckCircle className="w-3 h-3 mr-1" />
          Live Data
        </Badge>
      </div>

      {/* Model Status */}
      <Card className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-medium">Model Status</h3>
          <StatusIndicator status={metrics.model_status.status} />
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div>
            <p className="text-sm text-muted-foreground mb-1">Current Version</p>
            <p className="font-medium">{metrics.model_status.version}</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground mb-1">Uptime</p>
            <p className="font-medium">{metrics.model_status.uptime}%</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground mb-1">Last Trained</p>
            <p className="font-medium">{metrics.model_status.last_trained}</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground mb-1">Status</p>
            <Badge className="bg-green-100 text-green-800 hover:bg-green-100">
              <CheckCircle className="w-3 h-3 mr-1" />
              Active
            </Badge>
          </div>
        </div>
      </Card>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card className="p-6">
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm text-muted-foreground">Model Accuracy</p>
            <TrendingUp className="w-4 h-4 text-green-500" />
          </div>
          <p className="text-2xl font-medium mb-1">{metrics.performance.accuracy}%</p>
          <p className="text-xs text-green-600">↑ Active model performance</p>
        </Card>

        <Card className="p-6">
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm text-muted-foreground">Daily Predictions</p>
            <Activity className="w-4 h-4 text-blue-500" />
          </div>
          <p className="text-2xl font-medium mb-1">{metrics.predictions_24h.total.toLocaleString()}</p>
          <p className="text-xs text-blue-600">Total processed today</p>
        </Card>

        <Card className="p-6">
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm text-muted-foreground">Avg Response Time</p>
            <Zap className="w-4 h-4 text-yellow-500" />
          </div>
          <p className="text-2xl font-medium mb-1">{Math.round(metrics.system.avg_latency)}ms</p>
          <p className="text-xs text-green-600">Optimal performance</p>
        </Card>

        <Card className="p-6">
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm text-muted-foreground">Success Rate</p>
            <CheckCircle className="w-4 h-4 text-green-500" />
          </div>
          <p className="text-2xl font-medium mb-1">{metrics.system.success_rate}%</p>
          <p className="text-xs text-green-600">Excellent reliability</p>
        </Card>
      </div>

      {/* Performance Overview */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="p-6">
          <h3 className="font-medium mb-4">Model Performance</h3>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Accuracy</span>
                <span>{metrics.performance.accuracy}%</span>
              </div>
              <Progress value={metrics.performance.accuracy} />
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Precision</span>
                <span>{metrics.performance.precision}%</span>
              </div>
              <Progress value={metrics.performance.precision} />
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Recall</span>
                <span>{metrics.performance.recall}%</span>
              </div>
              <Progress value={metrics.performance.recall} />
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>F1 Score</span>
                <span>{metrics.performance.f1_score}%</span>
              </div>
              <Progress value={metrics.performance.f1_score} />
            </div>
          </div>
        </Card>

        <Card className="p-6">
          <h3 className="font-medium mb-4">System Resources</h3>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>CPU Usage</span>
                <span>{Math.round(metrics.system.cpu_usage)}%</span>
              </div>
              <Progress value={metrics.system.cpu_usage} />
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Memory Usage</span>
                <span>{Math.round(metrics.system.memory_usage)}%</span>
              </div>
              <Progress value={metrics.system.memory_usage} />
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Network I/O</span>
                <span>{Math.round(metrics.system.network_io)} MB/s</span>
              </div>
              <Progress value={metrics.system.network_io} />
            </div>
            <div className="flex items-center justify-between pt-2">
              <div className="flex items-center gap-2">
                <Server className="w-4 h-4 text-muted-foreground" />
                <span className="text-sm">Active Containers</span>
              </div>
              <Badge variant="outline">{metrics.system.active_containers}</Badge>
            </div>
          </div>
        </Card>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <PerformanceChart />
        <PredictionChart />
      </div>

      {/* Quick Stats */}
      <Card className="p-6">
        <h3 className="font-medium mb-4">Today's Predictions by Class</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="w-12 h-12 bg-chart-1/20 rounded-full flex items-center justify-center mx-auto mb-2">
              <div className="w-6 h-6 bg-chart-1 rounded-full"></div>
            </div>
            <p className="font-medium text-lg">{metrics.predictions_24h.by_class.roses}</p>
            <p className="text-sm text-muted-foreground">Roses</p>
          </div>
          <div className="text-center">
            <div className="w-12 h-12 bg-chart-2/20 rounded-full flex items-center justify-center mx-auto mb-2">
              <div className="w-6 h-6 bg-chart-2 rounded-full"></div>
            </div>
            <p className="font-medium text-lg">{metrics.predictions_24h.by_class.tulips}</p>
            <p className="text-sm text-muted-foreground">Tulips</p>
          </div>
          <div className="text-center">
            <div className="w-12 h-12 bg-chart-3/20 rounded-full flex items-center justify-center mx-auto mb-2">
              <div className="w-6 h-6 bg-chart-3 rounded-full"></div>
            </div>
            <p className="font-medium text-lg">{metrics.predictions_24h.by_class.sunflowers}</p>
            <p className="text-sm text-muted-foreground">Sunflowers</p>
          </div>
        </div>
      </Card>
    </div>
  );
}