import React, { useState, useEffect } from 'react';
import { Card } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { ScrollArea } from './ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { 
  Activity, 
  AlertTriangle, 
  CheckCircle, 
  Clock, 
  Download, 
  RefreshCw, 
  Server, 
  Users,
  Zap,
  Database,
  Globe,
  AlertCircle
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';

interface LogEntry {
  id: string;
  timestamp: string;
  level: 'info' | 'warning' | 'error' | 'success';
  category: 'api' | 'system' | 'training' | 'upload';
  message: string;
  details?: string;
}

export function SystemMonitoring() {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isLiveMode, setIsLiveMode] = useState(true);

  // Mock real-time system metrics
  const [systemMetrics, setSystemMetrics] = useState({
    cpu: 67,
    memory: 45,
    network: 23,
    requests: 156
  });

  // Performance data for charts
  const performanceData = [
    { time: '00:00', cpu: 45, memory: 62, network: 12, requests: 89 },
    { time: '00:15', cpu: 52, memory: 58, network: 18, requests: 124 },
    { time: '00:30', cpu: 48, memory: 61, network: 15, requests: 98 },
    { time: '00:45', cpu: 67, memory: 45, network: 23, requests: 156 },
    { time: '01:00', cpu: 54, memory: 52, network: 19, requests: 143 },
  ];

  const requestData = [
    { time: '00:00', successful: 234, failed: 12, total: 246 },
    { time: '00:15', successful: 312, failed: 8, total: 320 },
    { time: '00:30', successful: 267, failed: 15, total: 282 },
    { time: '00:45', successful: 389, failed: 6, total: 395 },
    { time: '01:00', successful: 445, failed: 11, total: 456 },
  ];

  // Mock logs data
  const mockLogs: LogEntry[] = [
    {
      id: '1',
      timestamp: '2024-01-08 14:32:15',
      level: 'success',
      category: 'api',
      message: 'Prediction request processed successfully',
      details: 'Image: rose_001.jpg, Confidence: 98.7%, Response time: 145ms'
    },
    {
      id: '2',
      timestamp: '2024-01-08 14:31:45',
      level: 'info',
      category: 'system',
      message: 'Auto-scaling triggered: Container instance added',
      details: 'New container: ml-worker-04, CPU threshold exceeded (75%)'
    },
    {
      id: '3',
      timestamp: '2024-01-08 14:30:22',
      level: 'success',
      category: 'training',
      message: 'Model retraining completed successfully',
      details: 'Version: v2.1.4-beta, Accuracy improved: 94.2% → 95.1%'
    },
    {
      id: '4',
      timestamp: '2024-01-08 14:29:18',
      level: 'warning',
      category: 'api',
      message: 'High latency detected on prediction endpoint',
      details: 'Average response time: 280ms (threshold: 200ms)'
    },
    {
      id: '5',
      timestamp: '2024-01-08 14:28:33',
      level: 'info',
      category: 'upload',
      message: 'Bulk upload processed',
      details: '75 tulip images uploaded by DataTeam, Processing time: 2.3s'
    },
    {
      id: '6',
      timestamp: '2024-01-08 14:27:45',
      level: 'error',
      category: 'api',
      message: 'Prediction request failed',
      details: 'Error: Invalid image format (image_corrupted.jpg), User: test@example.com'
    },
    {
      id: '7',
      timestamp: '2024-01-08 14:26:12',
      level: 'success',
      category: 'system',
      message: 'Health check passed',
      details: 'All services operational, Response time: 45ms'
    },
    {
      id: '8',
      timestamp: '2024-01-08 14:25:39',
      level: 'info',
      category: 'training',
      message: 'Dataset backup completed',
      details: 'Backup size: 2.4 GB, Location: s3://ml-backups/20240108/'
    }
  ];

  useEffect(() => {
    setLogs(mockLogs);

    // Simulate real-time updates
    if (isLiveMode) {
      const interval = setInterval(() => {
        // Update system metrics with some randomness
        setSystemMetrics(prev => ({
          cpu: Math.max(30, Math.min(90, prev.cpu + (Math.random() - 0.5) * 10)),
          memory: Math.max(20, Math.min(80, prev.memory + (Math.random() - 0.5) * 8)),
          network: Math.max(5, Math.min(50, prev.network + (Math.random() - 0.5) * 6)),
          requests: Math.max(50, Math.min(300, prev.requests + (Math.random() - 0.5) * 20))
        }));

        // Add new log entry occasionally
        if (Math.random() < 0.3) {
          const newLog: LogEntry = {
            id: Date.now().toString(),
            timestamp: new Date().toLocaleString(),
            level: Math.random() > 0.8 ? 'warning' : Math.random() > 0.9 ? 'error' : 'info',
            category: ['api', 'system', 'training', 'upload'][Math.floor(Math.random() * 4)] as any,
            message: 'New system event detected',
            details: 'Real-time monitoring update'
          };
          
          setLogs(prev => [newLog, ...prev.slice(0, 19)]); // Keep only latest 20 logs
        }
      }, 5000);

      return () => clearInterval(interval);
    }
  }, [isLiveMode]);

  const getLevelIcon = (level: string) => {
    switch (level) {
      case 'success': return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'warning': return <AlertTriangle className="w-4 h-4 text-yellow-500" />;
      case 'error': return <AlertCircle className="w-4 h-4 text-red-500" />;
      default: return <Clock className="w-4 h-4 text-blue-500" />;
    }
  };

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'success': return 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400';
      case 'warning': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400';
      case 'error': return 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400';
      default: return 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400';
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'api': return <Globe className="w-4 h-4" />;
      case 'system': return <Server className="w-4 h-4" />;
      case 'training': return <Activity className="w-4 h-4" />;
      case 'upload': return <Download className="w-4 h-4" />;
      default: return <Clock className="w-4 h-4" />;
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-medium">System Monitoring</h2>
          <p className="text-muted-foreground">Real-time system performance and event logging</p>
        </div>
        
        <div className="flex items-center gap-2">
          <Button
            variant={isLiveMode ? "default" : "outline"}
            size="sm"
            onClick={() => setIsLiveMode(!isLiveMode)}
          >
            <Activity className="w-4 h-4 mr-1" />
            {isLiveMode ? 'Live' : 'Paused'}
          </Button>
          <Button variant="outline" size="sm">
            <RefreshCw className="w-4 h-4 mr-1" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Real-time Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">CPU Usage</p>
              <p className="text-2xl font-medium">{systemMetrics.cpu}%</p>
            </div>
            <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900/20 rounded-lg flex items-center justify-center">
              <Activity className="w-6 h-6 text-blue-600" />
            </div>
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Memory</p>
              <p className="text-2xl font-medium">{systemMetrics.memory}%</p>
            </div>
            <div className="w-12 h-12 bg-purple-100 dark:bg-purple-900/20 rounded-lg flex items-center justify-center">
              <Database className="w-6 h-6 text-purple-600" />
            </div>
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Network I/O</p>
              <p className="text-2xl font-medium">{systemMetrics.network} MB/s</p>
            </div>
            <div className="w-12 h-12 bg-green-100 dark:bg-green-900/20 rounded-lg flex items-center justify-center">
              <Zap className="w-6 h-6 text-green-600" />
            </div>
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Requests/min</p>
              <p className="text-2xl font-medium">{systemMetrics.requests}</p>
            </div>
            <div className="w-12 h-12 bg-orange-100 dark:bg-orange-900/20 rounded-lg flex items-center justify-center">
              <Users className="w-6 h-6 text-orange-600" />
            </div>
          </div>
        </Card>
      </div>

      {/* Charts and Logs Tabs */}
      <Tabs defaultValue="performance" className="space-y-4">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="requests">API Requests</TabsTrigger>
          <TabsTrigger value="logs">Event Logs</TabsTrigger>
        </TabsList>

        <TabsContent value="performance">
          <Card className="p-6">
            <h3 className="font-medium mb-4">System Performance (Last Hour)</h3>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis 
                  dataKey="time" 
                  className="text-muted-foreground"
                  fontSize={12}
                />
                <YAxis 
                  className="text-muted-foreground"
                  fontSize={12}
                />
                <Tooltip 
                  contentStyle={{
                    backgroundColor: 'var(--card)',
                    border: '1px solid var(--border)',
                    borderRadius: '8px',
                    fontSize: '12px'
                  }}
                />
                <Line 
                  type="monotone" 
                  dataKey="cpu" 
                  stroke="hsl(var(--chart-1))" 
                  strokeWidth={2}
                  name="CPU %"
                />
                <Line 
                  type="monotone" 
                  dataKey="memory" 
                  stroke="hsl(var(--chart-2))" 
                  strokeWidth={2}
                  name="Memory %"
                />
                <Line 
                  type="monotone" 
                  dataKey="requests" 
                  stroke="hsl(var(--chart-3))" 
                  strokeWidth={2}
                  name="Requests/min"
                />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </TabsContent>

        <TabsContent value="requests">
          <Card className="p-6">
            <h3 className="font-medium mb-4">API Request Analytics</h3>
            <ResponsiveContainer width="100%" height={400}>
              <AreaChart data={requestData}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis 
                  dataKey="time" 
                  className="text-muted-foreground"
                  fontSize={12}
                />
                <YAxis 
                  className="text-muted-foreground"
                  fontSize={12}
                />
                <Tooltip 
                  contentStyle={{
                    backgroundColor: 'var(--card)',
                    border: '1px solid var(--border)',
                    borderRadius: '8px',
                    fontSize: '12px'
                  }}
                />
                <Area 
                  type="monotone" 
                  dataKey="successful" 
                  stackId="1"
                  stroke="hsl(var(--chart-1))" 
                  fill="hsl(var(--chart-1))"
                  name="Successful"
                />
                <Area 
                  type="monotone" 
                  dataKey="failed" 
                  stackId="1"
                  stroke="hsl(var(--chart-5))" 
                  fill="hsl(var(--chart-5))"
                  name="Failed"
                />
              </AreaChart>
            </ResponsiveContainer>
          </Card>
        </TabsContent>

        <TabsContent value="logs">
          <Card className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-medium">Event Logs</h3>
              <div className="flex items-center gap-2">
                {isLiveMode && (
                  <Badge variant="secondary" className="text-xs">
                    <Activity className="w-3 h-3 mr-1" />
                    Live Updates
                  </Badge>
                )}
                <Button variant="outline" size="sm">
                  Export Logs
                </Button>
              </div>
            </div>
            
            <ScrollArea className="h-96">
              <div className="space-y-2">
                {logs.map((log) => (
                  <div key={log.id} className="p-3 border rounded-lg hover:bg-muted/50 transition-colors">
                    <div className="flex items-start gap-3">
                      <div className="flex items-center gap-2 mt-0.5">
                        {getLevelIcon(log.level)}
                        {getCategoryIcon(log.category)}
                      </div>
                      
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between mb-1">
                          <div className="flex items-center gap-2">
                            <span className="font-medium text-sm">{log.message}</span>
                            <Badge className={`text-xs ${getLevelColor(log.level)}`}>
                              {log.level}
                            </Badge>
                            <Badge variant="outline" className="text-xs">
                              {log.category}
                            </Badge>
                          </div>
                          <span className="text-xs text-muted-foreground whitespace-nowrap">
                            {log.timestamp}
                          </span>
                        </div>
                        
                        {log.details && (
                          <p className="text-xs text-muted-foreground mt-1">
                            {log.details}
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </ScrollArea>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Container Status */}
      <Card className="p-6">
        <h3 className="font-medium mb-4">Container Status</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 border rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="font-medium">ml-worker-01</span>
              <Badge className="bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400">
                <CheckCircle className="w-3 h-3 mr-1" />
                Healthy
              </Badge>
            </div>
            <div className="text-sm text-muted-foreground space-y-1">
              <div>CPU: 52% | Memory: 67%</div>
              <div>Uptime: 23h 45m</div>
              <div>Requests: 1,247</div>
            </div>
          </div>

          <div className="p-4 border rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="font-medium">ml-worker-02</span>
              <Badge className="bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400">
                <CheckCircle className="w-3 h-3 mr-1" />
                Healthy
              </Badge>
            </div>
            <div className="text-sm text-muted-foreground space-y-1">
              <div>CPU: 48% | Memory: 61%</div>
              <div>Uptime: 18h 12m</div>
              <div>Requests: 978</div>
            </div>
          </div>

          <div className="p-4 border rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="font-medium">ml-worker-03</span>
              <Badge className="bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400">
                <AlertTriangle className="w-3 h-3 mr-1" />
                Warning
              </Badge>
            </div>
            <div className="text-sm text-muted-foreground space-y-1">
              <div>CPU: 89% | Memory: 76%</div>
              <div>Uptime: 2h 34m</div>
              <div>Requests: 2,156</div>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}