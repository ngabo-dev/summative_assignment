import React, { useState } from 'react';
import { Card } from './ui/card';
import { Button } from './ui/button';
import { Progress } from './ui/progress';
import { Badge } from './ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { 
  Upload, 
  FolderOpen, 
  CheckCircle, 
  XCircle, 
  Clock, 
  Play,
  TrendingUp,
  FileImage
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface UploadProgress {
  total: number;
  success: number;
  failed: number;
  processing: boolean;
}

interface RetrainingProgress {
  stage: string;
  progress: number;
  isActive: boolean;
  eta: string;
}

export function BulkUpload() {
  const [selectedClass, setSelectedClass] = useState<string>('');
  const [uploadProgress, setUploadProgress] = useState<UploadProgress | null>(null);
  const [retrainingProgress, setRetrainingProgress] = useState<RetrainingProgress | null>(null);
  const [files, setFiles] = useState<FileList | null>(null);

  const flowerClasses = ['rose', 'tulip', 'sunflower'];

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFiles(e.target.files);
    }
  };

  const handleBulkUpload = () => {
    if (!files || !selectedClass) return;

    setUploadProgress({
      total: files.length,
      success: 0,
      failed: 0,
      processing: true
    });

    // Simulate upload progress
    let success = 0;
    let failed = 0;
    const interval = setInterval(() => {
      const shouldFail = Math.random() < 0.1; // 10% chance of failure
      if (shouldFail) {
        failed++;
      } else {
        success++;
      }

      setUploadProgress({
        total: files.length,
        success,
        failed,
        processing: success + failed < files.length
      });

      if (success + failed >= files.length) {
        clearInterval(interval);
      }
    }, 200);
  };

  const handleRetrain = () => {
    const stages = [
      { stage: 'Preparing dataset', duration: 2000 },
      { stage: 'Data augmentation', duration: 3000 },
      { stage: 'Model training', duration: 8000 },
      { stage: 'Validation', duration: 2000 },
      { stage: 'Model optimization', duration: 1500 }
    ];

    let currentStage = 0;
    const startTime = Date.now();

    const processStage = () => {
      if (currentStage >= stages.length) {
        setRetrainingProgress(null);
        return;
      }

      const stage = stages[currentStage];
      setRetrainingProgress({
        stage: stage.stage,
        progress: 0,
        isActive: true,
        eta: `${Math.ceil((stage.duration * (stages.length - currentStage)) / 1000)}s`
      });

      const stageInterval = setInterval(() => {
        const elapsed = Date.now() - startTime - stages.slice(0, currentStage).reduce((acc, s) => acc + s.duration, 0);
        const progress = Math.min(100, (elapsed / stage.duration) * 100);
        
        setRetrainingProgress(prev => prev ? {
          ...prev,
          progress,
          eta: `${Math.ceil((stage.duration - elapsed + stages.slice(currentStage + 1).reduce((acc, s) => acc + s.duration, 0)) / 1000)}s`
        } : null);

        if (progress >= 100) {
          clearInterval(stageInterval);
          currentStage++;
          setTimeout(processStage, 100);
        }
      }, 100);
    };

    processStage();
  };

  const performanceComparison = [
    { metric: 'Accuracy', previous: 92.1, current: 94.2 },
    { metric: 'Precision', previous: 90.5, current: 92.8 },
    { metric: 'Recall', previous: 93.2, current: 95.1 },
    { metric: 'F1-Score', previous: 91.8, current: 93.9 }
  ];

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-medium mb-2">Bulk Upload & Retraining</h2>
        <p className="text-muted-foreground">Upload multiple images and retrain the model with new data</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Upload Section */}
        <Card className="p-6">
          <h3 className="font-medium mb-4">Bulk Image Upload</h3>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">Select Flower Class</label>
              <Select value={selectedClass} onValueChange={setSelectedClass}>
                <SelectTrigger>
                  <SelectValue placeholder="Choose flower type" />
                </SelectTrigger>
                <SelectContent>
                  {flowerClasses.map((cls) => (
                    <SelectItem key={cls} value={cls}>
                      {cls.charAt(0).toUpperCase() + cls.slice(1)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Select Images</label>
              <div className="relative">
                <input
                  type="file"
                  multiple
                  accept="image/*"
                  onChange={handleFileSelect}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  title="Select images to upload"
                />
                <div className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-6 text-center hover:border-muted-foreground/50 transition-colors">
                  <FolderOpen className="mx-auto h-8 w-8 text-muted-foreground mb-2" />
                  <p className="text-sm">Click to select multiple images</p>
                  {files && (
                    <p className="text-xs text-muted-foreground mt-1">
                      {files.length} files selected
                    </p>
                  )}
                </div>
              </div>
            </div>

            <Button 
              onClick={handleBulkUpload}
              disabled={!files || !selectedClass || uploadProgress?.processing}
              className="w-full"
            >
              <Upload className="w-4 h-4 mr-2" />
              Upload Images
            </Button>

            {uploadProgress && (
              <div className="space-y-3 p-4 bg-muted/50 rounded-lg">
                <div className="flex justify-between text-sm">
                  <span>Upload Progress</span>
                  <span>{uploadProgress.success + uploadProgress.failed}/{uploadProgress.total}</span>
                </div>
                <Progress value={((uploadProgress.success + uploadProgress.failed) / uploadProgress.total) * 100} />
                
                <div className="grid grid-cols-3 gap-2 text-xs">
                  <div className="flex items-center gap-1">
                    <CheckCircle className="w-3 h-3 text-green-500" />
                    <span>Success: {uploadProgress.success}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <XCircle className="w-3 h-3 text-red-500" />
                    <span>Failed: {uploadProgress.failed}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <Clock className="w-3 h-3 text-blue-500" />
                    <span>Remaining: {uploadProgress.total - uploadProgress.success - uploadProgress.failed}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </Card>

        {/* Retraining Section */}
        <Card className="p-6">
          <h3 className="font-medium mb-4">Model Retraining</h3>
          
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div className="space-y-1">
                <div className="text-muted-foreground">Training Images</div>
                <div className="font-medium">2,847 total</div>
              </div>
              <div className="space-y-1">
                <div className="text-muted-foreground">Last Trained</div>
                <div className="font-medium">2 hours ago</div>
              </div>
              <div className="space-y-1">
                <div className="text-muted-foreground">Current Model</div>
                <Badge variant="secondary">v2.1.3</Badge>
              </div>
              <div className="space-y-1">
                <div className="text-muted-foreground">Accuracy</div>
                <div className="font-medium">94.2%</div>
              </div>
            </div>

            <Button 
              onClick={handleRetrain}
              disabled={retrainingProgress?.isActive}
              className="w-full"
              variant={retrainingProgress?.isActive ? "secondary" : "default"}
            >
              <Play className="w-4 h-4 mr-2" />
              {retrainingProgress?.isActive ? 'Retraining in Progress' : 'Start Retraining'}
            </Button>

            {retrainingProgress && (
              <div className="space-y-3 p-4 bg-muted/50 rounded-lg">
                <div className="flex justify-between text-sm">
                  <span>{retrainingProgress.stage}</span>
                  <span>ETA: {retrainingProgress.eta}</span>
                </div>
                <Progress value={retrainingProgress.progress} />
                <div className="text-xs text-muted-foreground">
                  Training with augmented dataset...
                </div>
              </div>
            )}
          </div>
        </Card>
      </div>

      {/* Performance Comparison */}
      <Card className="p-6">
        <h3 className="font-medium mb-4">Model Performance Comparison</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {performanceComparison.map((metric, index) => (
            <div key={index} className="text-center p-4 border rounded-lg">
              <div className="text-sm text-muted-foreground mb-1">{metric.metric}</div>
              <div className="space-y-1">
                <div className="flex items-center justify-center gap-2">
                  <span className="text-xs text-muted-foreground">Previous:</span>
                  <span className="font-medium">{metric.previous}%</span>
                </div>
                <div className="flex items-center justify-center gap-2">
                  <span className="text-xs text-muted-foreground">Current:</span>
                  <span className="font-medium text-green-600">{metric.current}%</span>
                  <TrendingUp className="w-3 h-3 text-green-500" />
                </div>
              </div>
              <div className="text-xs text-green-600 mt-1">
                +{(metric.current - metric.previous).toFixed(1)}%
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Recent Uploads */}
      <Card className="p-6">
        <h3 className="font-medium mb-4">Recent Upload History</h3>
        <div className="space-y-3">
          <div className="flex items-center justify-between py-2 border-b">
            <div className="flex items-center gap-3">
              <FileImage className="w-4 h-4 text-blue-500" />
              <div>
                <div className="font-medium text-sm">75 Tulip Images</div>
                <div className="text-xs text-muted-foreground">Uploaded by Admin</div>
              </div>
            </div>
            <div className="text-right">
              <div className="text-sm">6 hours ago</div>
              <Badge variant="secondary" className="text-xs">Processed</Badge>
            </div>
          </div>
          
          <div className="flex items-center justify-between py-2 border-b">
            <div className="flex items-center gap-3">
              <FileImage className="w-4 h-4 text-red-500" />
              <div>
                <div className="font-medium text-sm">42 Rose Images</div>
                <div className="text-xs text-muted-foreground">Uploaded by DataTeam</div>
              </div>
            </div>
            <div className="text-right">
              <div className="text-sm">1 day ago</div>
              <Badge variant="secondary" className="text-xs">Processed</Badge>
            </div>
          </div>
          
          <div className="flex items-center justify-between py-2">
            <div className="flex items-center gap-3">
              <FileImage className="w-4 h-4 text-yellow-500" />
              <div>
                <div className="font-medium text-sm">28 Sunflower Images</div>
                <div className="text-xs text-muted-foreground">Uploaded by MLTeam</div>
              </div>
            </div>
            <div className="text-right">
              <div className="text-sm">2 days ago</div>
              <Badge variant="secondary" className="text-xs">Processed</Badge>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}