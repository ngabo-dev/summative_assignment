import { useState } from 'react';
import { Card } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from './ui/alert_dialog';
import { 
  CheckCircle, 
  Clock, 
  Download, 
  Play, 
  RotateCcw, 
  Settings, 
  TrendingUp,
  Database,
  Cpu,
  Calendar
} from 'lucide-react';

interface ModelVersion {
  id: string;
  version: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  trainingTime: string;
  datasetSize: number;
  modelSize: string;
  epochCount: number;
  status: 'deployed' | 'archived' | 'testing';
  createdAt: string;
  deployedAt?: string;
}
// Ensure the file extension is .tsx and tsconfig.json has "jsx": "react-jsx" or "react"

export function ModelManagement() {
  const [deployingVersion, setDeployingVersion] = useState<string | null>(null);

  const models: ModelVersion[] = [
    {
      id: '1',
      version: 'v2.1.3',
      accuracy: 94.2,
      precision: 92.8,
      recall: 95.1,
      f1Score: 93.9,
      trainingTime: '2.3 hours',
      datasetSize: 2847,
      modelSize: '45.2 MB',
      epochCount: 50,
      status: 'deployed',
      createdAt: '2024-01-03 14:30',
      deployedAt: '2024-01-03 16:45'
    },
    {
      id: '2',
      version: 'v2.1.2',
      accuracy: 92.1,
      precision: 90.5,
      recall: 93.2,
      f1Score: 91.8,
      trainingTime: '2.1 hours',
      datasetSize: 2672,
      modelSize: '44.8 MB',
      epochCount: 45,
      status: 'archived',
      createdAt: '2024-01-01 09:15',
      deployedAt: '2024-01-01 11:30'
    },
    {
      id: '3',
      version: 'v2.1.1',
      accuracy: 89.7,
      precision: 88.2,
      recall: 91.1,
      f1Score: 89.6,
      trainingTime: '1.9 hours',
      datasetSize: 2456,
      modelSize: '43.9 MB',
      epochCount: 40,
      status: 'archived',
      createdAt: '2023-12-28 16:20',
      deployedAt: '2023-12-28 18:45'
    },
    {
      id: '4',
      version: 'v2.1.4-beta',
      accuracy: 95.1,
      precision: 94.2,
      recall: 96.3,
      f1Score: 95.2,
      trainingTime: '2.8 hours',
      datasetSize: 3024,
      modelSize: '46.7 MB',
      epochCount: 60,
      status: 'testing',
      createdAt: '2024-01-05 10:12'
    }
  ];

  const handleDeploy = async (version: string) => {
    setDeployingVersion(version);
    
    // Simulate deployment process
    setTimeout(() => {
      setDeployingVersion(null);
      // In a real app, you would update the model status here
    }, 3000);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'deployed': return 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400';
      case 'testing': return 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400';
      case 'archived': return 'bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400';
      default: return 'bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'deployed': return <CheckCircle className="w-4 h-4" />;
      case 'testing': return <Clock className="w-4 h-4" />;
      case 'archived': return <Download className="w-4 h-4" />;
      default: return <Settings className="w-4 h-4" />;
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-medium mb-2">Model Management</h2>
        <p className="text-muted-foreground">Manage model versions, deployments, and performance tracking</p>
      </div>

      {/* Current Deployment Status */}
      <Card className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-medium">Current Deployment</h3>
          <Badge className={getStatusColor('deployed')}>
            <CheckCircle className="w-3 h-3 mr-1" />
            Production
          </Badge>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="text-center p-4 bg-muted/50 rounded-lg">
            <div className="text-2xl font-medium">v2.1.3</div>
            <div className="text-sm text-muted-foreground">Active Version</div>
          </div>
          <div className="text-center p-4 bg-muted/50 rounded-lg">
            <div className="text-2xl font-medium">94.2%</div>
            <div className="text-sm text-muted-foreground">Accuracy</div>
          </div>
          <div className="text-center p-4 bg-muted/50 rounded-lg">
            <div className="text-2xl font-medium">145ms</div>
            <div className="text-sm text-muted-foreground">Avg Latency</div>
          </div>
          <div className="text-center p-4 bg-muted/50 rounded-lg">
            <div className="text-2xl font-medium">99.97%</div>
            <div className="text-sm text-muted-foreground">Uptime</div>
          </div>
        </div>
      </Card>

      {/* Model Versions */}
      <Card className="p-6">
        <h3 className="font-medium mb-4">Model Versions</h3>
        <div className="space-y-4">
          {models.map((model) => (
            <div key={model.id} className="border rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-3">
                  <h4 className="font-medium">{model.version}</h4>
                  <Badge className={getStatusColor(model.status)}>
                    {getStatusIcon(model.status)}
                    <span className="ml-1 capitalize">{model.status}</span>
                  </Badge>
                  {model.status === 'testing' && (
                    <Badge variant="outline" className="text-xs">
                      <TrendingUp className="w-3 h-3 mr-1" />
                      +0.9% accuracy
                    </Badge>
                  )}
                </div>
                
                <div className="flex gap-2">
                  {model.status !== 'deployed' && (
                    <AlertDialog>
                      <AlertDialogTrigger asChild>
                        <Button size="sm" variant="outline">
                          <Play className="w-4 h-4 mr-1" />
                          Deploy
                        </Button>
                      </AlertDialogTrigger>
                      <AlertDialogContent>
                        <AlertDialogHeader>
                          <AlertDialogTitle>Deploy Model {model.version}</AlertDialogTitle>
                          <AlertDialogDescription>
                            This will replace the current production model (v2.1.3) with {model.version}. 
                            This action will cause a brief service interruption during the deployment process.
                          </AlertDialogDescription>
                        </AlertDialogHeader>
                        <AlertDialogFooter>
                          <AlertDialogCancel>Cancel</AlertDialogCancel>
                          <AlertDialogAction onClick={() => handleDeploy(model.version)}>
                            Deploy Model
                          </AlertDialogAction>
                        </AlertDialogFooter>
                      </AlertDialogContent>
                    </AlertDialog>
                  )}
                  
                  {model.status === 'deployed' && (
                    <Button size="sm" variant="outline">
                      <RotateCcw className="w-4 h-4 mr-1" />
                      Rollback
                    </Button>
                  )}
                </div>
              </div>

              {deployingVersion === model.version && (
                <div className="mb-3 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                  <div className="flex justify-between text-sm mb-1">
                    <span>Deploying {model.version}...</span>
                    <span>67%</span>
                  </div>
                  <Progress value={67} />
                  <p className="text-xs text-muted-foreground mt-1">
                    Updating model weights and restarting containers...
                  </p>
                </div>
              )}

              <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-4 text-sm">
                <div>
                  <div className="text-muted-foreground">Accuracy</div>
                  <div className="font-medium">{model.accuracy}%</div>
                </div>
                <div>
                  <div className="text-muted-foreground">Precision</div>
                  <div className="font-medium">{model.precision}%</div>
                </div>
                <div>
                  <div className="text-muted-foreground">Recall</div>
                  <div className="font-medium">{model.recall}%</div>
                </div>
                <div>
                  <div className="text-muted-foreground">F1-Score</div>
                  <div className="font-medium">{model.f1Score}%</div>
                </div>
                <div>
                  <div className="text-muted-foreground">Training Time</div>
                  <div className="font-medium">{model.trainingTime}</div>
                </div>
                <div>
                  <div className="text-muted-foreground">Dataset Size</div>
                  <div className="font-medium">{model.datasetSize.toLocaleString()}</div>
                </div>
                <div>
                  <div className="text-muted-foreground">Model Size</div>
                  <div className="font-medium">{model.modelSize}</div>
                </div>
                <div>
                  <div className="text-muted-foreground">Epochs</div>
                  <div className="font-medium">{model.epochCount}</div>
                </div>
              </div>

              <div className="mt-3 flex items-center gap-4 text-xs text-muted-foreground">
                <div className="flex items-center gap-1">
                  <Calendar className="w-3 h-3" />
                  <span>Created: {model.createdAt}</span>
                </div>
                {model.deployedAt && (
                  <div className="flex items-center gap-1">
                    <CheckCircle className="w-3 h-3" />
                    <span>Deployed: {model.deployedAt}</span>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Model Comparison */}
      <Card className="p-6">
        <h3 className="font-medium mb-4">Performance Comparison</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b">
                <th className="text-left py-2">Version</th>
                <th className="text-left py-2">Accuracy</th>
                <th className="text-left py-2">Precision</th>
                <th className="text-left py-2">Recall</th>
                <th className="text-left py-2">F1-Score</th>
                <th className="text-left py-2">Model Size</th>
                <th className="text-left py-2">Training Time</th>
              </tr>
            </thead>
            <tbody>
              {models.slice(0, 3).map((model) => (
                <tr key={model.id} className="border-b">
                  <td className="py-2">
                    <div className="flex items-center gap-2">
                      {model.version}
                      {model.status === 'deployed' && (
                        <Badge variant="secondary" className="text-xs">Current</Badge>
                      )}
                    </div>
                  </td>
                  <td className="py-2">{model.accuracy}%</td>
                  <td className="py-2">{model.precision}%</td>
                  <td className="py-2">{model.recall}%</td>
                  <td className="py-2">{model.f1Score}%</td>
                  <td className="py-2">{model.modelSize}</td>
                  <td className="py-2">{model.trainingTime}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* System Resources */}
      <Card className="p-6">
        <h3 className="font-medium mb-4">System Resources</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Cpu className="w-4 h-4" />
              <span className="font-medium">CPU Usage</span>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Current</span>
                <span>67%</span>
              </div>
              <Progress value={67} />
            </div>
            <div className="text-xs text-muted-foreground">
              8 cores allocated, 2.4 GHz average
            </div>
          </div>

          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Database className="w-4 h-4" />
              <span className="font-medium">Memory Usage</span>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Current</span>
                <span>45%</span>
              </div>
              <Progress value={45} />
            </div>
            <div className="text-xs text-muted-foreground">
              18.2 GB / 32 GB allocated
            </div>
          </div>

          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Settings className="w-4 h-4" />
              <span className="font-medium">Container Status</span>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Active Containers</span>
                <span>3/5</span>
              </div>
              <Progress value={60} />
            </div>
            <div className="text-xs text-muted-foreground">
              Auto-scaling enabled, min: 2, max: 8
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}