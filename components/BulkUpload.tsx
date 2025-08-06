import React, { useState, useEffect } from 'react';
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
  FileImage,
  AlertCircle,
  Loader2
} from 'lucide-react';

interface RetrainProgress {
  stage: string;
  progress: number;
  isActive: boolean;
  eta?: string;
  error?: string;
}

interface ApiStats {
  system: {
    uptime: string;
    last_trained: string;
    prediction_count: number;
  };
  model: {
    loaded: boolean;
    path: string;
  };
}

interface RetrainResponse {
  success: boolean;
  message: string;
  new_classes: string[];
  model_path: string;
  timestamp: string;
  images_processed: number;
}

export function BulkUpload() {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [selectedClass, setSelectedClass] = useState<string>('');
  const [retrainingProgress, setRetrainingProgress] = useState<RetrainProgress | null>(null);
  const [apiStats, setApiStats] = useState<ApiStats | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const flowerClasses = ['rose', 'tulip', 'sunflower'];
  const API_BASE_URL = 'http://localhost:8000'; // Adjust as needed

  // Fetch API stats on component mount
  useEffect(() => {
    fetchApiStats();
  }, []);

  const fetchApiStats = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/stats`);
      if (response.ok) {
        const stats = await response.json();
        setApiStats(stats);
      }
    } catch (error) {
      console.error('Failed to fetch API stats:', error);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const filesArray = Array.from(e.target.files);
      setSelectedFiles(filesArray);
      setError(null);
      setSuccessMessage(null);
    }
  };

  const handleRetrain = async () => {
    if (!selectedFiles.length || !selectedClass) {
      setError('Please select files and a flower class');
      return;
    }

    setIsLoading(true);
    setError(null);
    setSuccessMessage(null);
    
    setRetrainingProgress({
      stage: 'Preparing images for upload...',
      progress: 0,
      isActive: true
    });

    try {
      // Prepare form data
      const formData = new FormData();
      
      // Add all selected files
      selectedFiles.forEach((file, index) => {
        formData.append('images', file);
      });

      // Add labels (same class for all images)
      selectedFiles.forEach(() => {
        formData.append('labels', selectedClass);
      });

      setRetrainingProgress({
        stage: 'Uploading images and starting retraining...',
        progress: 20,
        isActive: true
      });

      // Call the retrain endpoint
      const response = await fetch(`${API_BASE_URL}/api/train/retrain`, {
        method: 'POST',
        body: formData,
      });

      const result: RetrainResponse = await response.json();

      if (response.ok && result.success) {
        setRetrainingProgress({
          stage: 'Retraining completed successfully!',
          progress: 100,
          isActive: false
        });

        setSuccessMessage(
          `Successfully retrained model with ${result.images_processed} images. ${result.message}`
        );

        // Clear form
        setSelectedFiles([]);
        setSelectedClass('');
        
        // Refresh stats
        setTimeout(() => {
          fetchApiStats();
          setRetrainingProgress(null);
        }, 2000);

      } else {
        throw new Error(result.message || 'Retraining failed');
      }

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      setError(`Retraining failed: ${errorMessage}`);
      setRetrainingProgress({
        stage: 'Retraining failed',
        progress: 0,
        isActive: false,
        error: errorMessage
      });
    } finally {
      setIsLoading(false);
    }
  };

  const removeFile = (index: number) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-medium mb-2">Model Retraining</h2>
        <p className="text-muted-foreground">Upload flower images to retrain and improve the model</p>
      </div>

      {/* Error Message */}
      {error && (
        <Card className="p-4 border-red-200 bg-red-50">
          <div className="flex items-center gap-2 text-red-700">
            <AlertCircle className="w-4 h-4" />
            <span className="text-sm font-medium">{error}</span>
          </div>
        </Card>
      )}

      {/* Success Message */}
      {successMessage && (
        <Card className="p-4 border-green-200 bg-green-50">
          <div className="flex items-center gap-2 text-green-700">
            <CheckCircle className="w-4 h-4" />
            <span className="text-sm font-medium">{successMessage}</span>
          </div>
        </Card>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Upload Section */}
        <Card className="p-6">
          <h3 className="font-medium mb-4">Upload Training Images</h3>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">Select Flower Class</label>
              <Select value={selectedClass} onValueChange={setSelectedClass} disabled={isLoading}>
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
                  disabled={isLoading}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed"
                  title="Select images to upload"
                />
                <div className={`border-2 border-dashed rounded-lg p-6 text-center transition-colors ${
                  isLoading 
                    ? 'border-muted-foreground/25 bg-muted/25' 
                    : 'border-muted-foreground/25 hover:border-muted-foreground/50'
                }`}>
                  <FolderOpen className="mx-auto h-8 w-8 text-muted-foreground mb-2" />
                  <p className="text-sm">Click to select multiple images</p>
                  {selectedFiles.length > 0 && (
                    <p className="text-xs text-muted-foreground mt-1">
                      {selectedFiles.length} files selected
                    </p>
                  )}
                </div>
              </div>
            </div>

            {/* Selected Files List */}
            {selectedFiles.length > 0 && (
              <div className="max-h-32 overflow-y-auto space-y-1 p-2 border rounded">
                {selectedFiles.map((file, index) => (
                  <div key={index} className="flex items-center justify-between text-xs bg-muted/50 p-2 rounded">
                    <span className="truncate flex-1">{file.name}</span>
                    <button
                      onClick={() => removeFile(index)}
                      disabled={isLoading}
                      className="text-red-500 hover:text-red-700 ml-2 disabled:opacity-50"
                    >
                      <XCircle className="w-3 h-3" />
                    </button>
                  </div>
                ))}
              </div>
            )}

            <Button 
              onClick={handleRetrain}
              disabled={!selectedFiles.length || !selectedClass || isLoading}
              className="w-full"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Retraining...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 mr-2" />
                  Start Retraining
                </>
              )}
            </Button>

            {retrainingProgress && (
              <div className="space-y-3 p-4 bg-muted/50 rounded-lg">
                <div className="flex justify-between text-sm">
                  <span>{retrainingProgress.stage}</span>
                  {retrainingProgress.eta && (
                    <span>ETA: {retrainingProgress.eta}</span>
                  )}
                </div>
                <Progress value={retrainingProgress.progress} />
                {retrainingProgress.error && (
                  <div className="text-xs text-red-600">
                    Error: {retrainingProgress.error}
                  </div>
                )}
                {retrainingProgress.isActive && (
                  <div className="text-xs text-muted-foreground">
                    Processing {selectedFiles.length} images for {selectedClass} class...
                  </div>
                )}
              </div>
            )}
          </div>
        </Card>

        {/* Model Status Section */}
        <Card className="p-6">
          <h3 className="font-medium mb-4">Current Model Status</h3>
          
          <div className="space-y-4">
            {apiStats ? (
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div className="space-y-1">
                  <div className="text-muted-foreground">Model Status</div>
                  <div className="flex items-center gap-2">
                    {apiStats.model.loaded ? (
                      <>
                        <Badge variant="secondary" className="bg-green-100 text-green-800">Loaded</Badge>
                      </>
                    ) : (
                      <Badge variant="destructive">Not Loaded</Badge>
                    )}
                  </div>
                </div>
                <div className="space-y-1">
                  <div className="text-muted-foreground">System Uptime</div>
                  <div className="font-medium">{apiStats.system.uptime}</div>
                </div>
                <div className="space-y-1">
                  <div className="text-muted-foreground">Last Trained</div>
                  <div className="font-medium">{apiStats.system.last_trained}</div>
                </div>
                <div className="space-y-1">
                  <div className="text-muted-foreground">Total Predictions</div>
                  <div className="font-medium">{apiStats.system.prediction_count.toLocaleString()}</div>
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
              </div>
            )}

            <Button 
              onClick={fetchApiStats}
              variant="outline"
              size="sm"
              disabled={isLoading}
              className="w-full"
            >
              Refresh Status
            </Button>

            <div className="text-xs text-muted-foreground border-t pt-4">
              <div><strong>Model Path:</strong> {apiStats?.model.path || 'Loading...'}</div>
              <div><strong>Supported Classes:</strong> {flowerClasses.join(', ')}</div>
              <div><strong>Supported Formats:</strong> PNG, JPG, JPEG, GIF</div>
            </div>
          </div>
        </Card>
      </div>

      {/* Instructions */}
      <Card className="p-6">
        <h3 className="font-medium mb-4">Retraining Instructions</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="flex items-start gap-3">
            <div className="w-6 h-6 bg-blue-100 rounded-full flex items-center justify-center text-xs font-bold text-blue-600 mt-0.5">
              1
            </div>
            <div>
              <div className="font-medium text-sm">Select Class & Images</div>
              <div className="text-xs text-muted-foreground mt-1">
                Choose the flower type and select multiple high-quality images of that flower
              </div>
            </div>
          </div>
          
          <div className="flex items-start gap-3">
            <div className="w-6 h-6 bg-blue-100 rounded-full flex items-center justify-center text-xs font-bold text-blue-600 mt-0.5">
              2
            </div>
            <div>
              <div className="font-medium text-sm">Start Retraining</div>
              <div className="text-xs text-muted-foreground mt-1">
                Click "Start Retraining" to upload images and retrain the model (takes ~30-60 seconds)
              </div>
            </div>
          </div>
          
          <div className="flex items-start gap-3">
            <div className="w-6 h-6 bg-green-100 rounded-full flex items-center justify-center text-xs font-bold text-green-600 mt-0.5">
              3
            </div>
            <div>
              <div className="font-medium text-sm">Model Updated</div>
              <div className="text-xs text-muted-foreground mt-1">
                The model is automatically updated and ready for improved predictions
              </div>
            </div>
          </div>
        </div>
      </Card>

      {/* Technical Details */}
      <Card className="p-6">
        <h3 className="font-medium mb-4">Technical Details</h3>
        <div className="text-xs text-muted-foreground space-y-1">
          <div><strong>API Endpoint:</strong> POST /api/train/retrain</div>
          <div><strong>Retraining Method:</strong> Transfer learning with 5 epochs</div>
          <div><strong>Image Requirements:</strong> Any size (auto-resized to 150x150)</div>
          <div><strong>Batch Processing:</strong> All images processed simultaneously</div>
          <div><strong>Model Update:</strong> Automatic reload after successful training</div>
        </div>
      </Card>
    </div>
  );
}