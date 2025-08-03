import React, { useState, useRef } from 'react';
import { Card } from './ui/card';
import { Button } from './ui/button';
import { Progress } from './ui/progress';
import { Badge } from './ui/badge';
import { Alert, AlertDescription } from './ui/alert';
import { Upload, Image as ImageIcon, Clock, CheckCircle, AlertTriangle, RefreshCw } from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';

interface PredictionResult {
  predicted_class: string;
  confidence: number;
  probabilities: Array<{ name: string; probability: number; color: string }>;
  response_time_ms: number;
  timestamp: string;
}

interface SinglePredictionProps {
  apiBaseUrl: string;
  apiStatus: 'online' | 'offline' | 'checking';
}

export function SinglePrediction({ apiBaseUrl, apiStatus }: SinglePredictionProps) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file: File) => {
    if (file.type.startsWith('image/')) {
      setSelectedFile(file);
      setPrediction(null);
      setError(null);
    } else {
      setError('Please select a valid image file (JPG, PNG, GIF)');
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) return;
    
    if (apiStatus !== 'online') {
      setError('Backend connection required for predictions');
      return;
    }
    
    setIsLoading(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('image', selectedFile);
      
      const response = await fetch(`${apiBaseUrl}/predict/single`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
          const errorData = await response.json();
          throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
        } else {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
      }
      
      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('application/json')) {
        throw new Error('Invalid response format - expected JSON');
      }
      
      const result = await response.json();
      
      // Format the result to match our component interface
      const formattedResult: PredictionResult = {
        predicted_class: result.predicted_class,
        confidence: result.confidence,
        probabilities: result.probabilities || [
          { name: result.predicted_class, probability: result.confidence / 100, color: 'hsl(var(--chart-1))' }
        ],
        response_time_ms: result.response_time_ms || 0,
        timestamp: result.timestamp || new Date().toISOString()
      };
      
      setPrediction(formattedResult);
      
    } catch (error) {
      console.error('Prediction error:', error);
      setError(error instanceof Error ? error.message : 'Failed to process image');
    } finally {
      setIsLoading(false);
    }
  };

  const resetPrediction = () => {
    setSelectedFile(null);
    setPrediction(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  if (apiStatus === 'offline') {
    return (
      <div className="max-w-4xl mx-auto space-y-6">
        <div className="text-center">
          <h2 className="text-2xl font-medium mb-2">Single Image Prediction</h2>
          <p className="text-muted-foreground">Upload a flower image to get instant classification results</p>
        </div>

        <Card className="p-12 text-center">
          <AlertTriangle className="w-16 h-16 text-red-500 mx-auto mb-4" />
          <h3 className="text-xl font-medium mb-2">Backend Connection Required</h3>
          <p className="text-muted-foreground mb-6 max-w-md mx-auto">
            Image prediction requires the Flask ML backend to be running.
          </p>
          <div className="space-y-2 text-sm text-muted-foreground max-w-md mx-auto">
            <p>1. Navigate to the <code className="bg-muted px-1 rounded">backend/</code> directory</p>
            <p>2. Run <code className="bg-muted px-1 rounded">pip install -r requirements.txt</code></p>
            <p>3. Start the server with <code className="bg-muted px-1 rounded">python app.py</code></p>
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-medium mb-2">Single Image Prediction</h2>
        <p className="text-muted-foreground">Upload a flower image to get instant ML classification results</p>
        <Badge variant="outline" className="text-green-600 border-green-600 mt-2">
          <CheckCircle className="w-3 h-3 mr-1" />
          Live ML Model
        </Badge>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Upload Section */}
        <Card className="p-6">
          <h3 className="font-medium mb-4">Upload Image</h3>
          
          {error && (
            <Alert className="mb-4 border-red-200 bg-red-50 dark:border-red-900 dark:bg-red-900/20">
              <AlertTriangle className="h-4 w-4 text-red-600" />
              <AlertDescription className="text-red-700 dark:text-red-300">
                {error}
              </AlertDescription>
            </Alert>
          )}
          
          {!selectedFile ? (
            <div
              className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                dragActive 
                  ? 'border-primary bg-primary/5' 
                  : 'border-muted-foreground/25 hover:border-muted-foreground/50'
              }`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleChange}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              <Upload className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
              <p className="text-lg font-medium mb-2">Drop your flower image here</p>
              <p className="text-muted-foreground mb-4">or click to browse files</p>
              <Button variant="outline">Choose File</Button>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="relative">
                <img
                  src={URL.createObjectURL(selectedFile)}
                  alt="Preview"
                  className="w-full h-48 object-cover rounded-lg"
                />
                <Badge className="absolute top-2 right-2">{selectedFile.name}</Badge>
              </div>
              
              <div className="flex gap-2">
                <Button 
                  onClick={handlePredict} 
                  disabled={isLoading || apiStatus !== 'online'}
                  className="flex-1"
                >
                  {isLoading ? (
                    <>
                      <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                      Processing...
                    </>
                  ) : (
                    'Predict Flower Type'
                  )}
                </Button>
                <Button variant="outline" onClick={resetPrediction}>
                  Reset
                </Button>
              </div>
              
              {isLoading && (
                <div className="space-y-2">
                  <Progress value={66} />
                  <p className="text-sm text-muted-foreground text-center">
                    Processing image with CNN model...
                  </p>
                </div>
              )}
            </div>
          )}
        </Card>

        {/* Results Section */}
        <Card className="p-6">
          <h3 className="font-medium mb-4">Prediction Results</h3>
          
          {!prediction ? (
            <div className="flex flex-col items-center justify-center h-64 text-muted-foreground">
              <ImageIcon className="w-12 h-12 mb-4" />
              <p>Upload an image to see prediction results</p>
              <p className="text-xs mt-2">
                {apiStatus === 'online' ? 'Python ML model ready' : 'Waiting for backend connection'}
              </p>
            </div>
          ) : (
            <div className="space-y-6">
              {/* Main Prediction */}
              <div className="text-center p-4 bg-primary/5 rounded-lg">
                <div className="flex items-center justify-center gap-2 mb-2">
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <span className="font-medium">Predicted Class</span>
                </div>
                <div className="text-2xl font-medium mb-1 capitalize">{prediction.predicted_class}</div>
                <div className="text-lg text-muted-foreground">
                  {prediction.confidence.toFixed(1)}% confidence
                </div>
              </div>

              {/* Response Time */}
              <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground">
                <Clock className="w-4 h-4" />
                <span>Response time: {prediction.response_time_ms}ms</span>
              </div>

              {/* Probability Distribution */}
              {prediction.probabilities.length > 1 && (
                <div>
                  <h4 className="font-medium mb-3">Class Probabilities</h4>
                  <ResponsiveContainer width="100%" height={200}>
                    <PieChart>
                      <Pie
                        data={prediction.probabilities}
                        cx="50%"
                        cy="50%"
                        outerRadius={70}
                        fill="#8884d8"
                        dataKey="probability"
                      >
                        {prediction.probabilities.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip 
                        formatter={(value: number) => `${(value * 100).toFixed(1)}%`}
                        contentStyle={{
                          backgroundColor: 'var(--card)',
                          border: '1px solid var(--border)',
                          borderRadius: '8px',
                          fontSize: '12px'
                        }}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              )}

              {/* Detailed Probabilities */}
              {prediction.probabilities.length > 1 && (
                <div className="space-y-2">
                  {prediction.probabilities.map((prob, index) => (
                    <div key={index} className="flex items-center justify-between">
                      <span className="text-sm capitalize">{prob.name}</span>
                      <div className="flex items-center gap-2">
                        <div className="w-24">
                          <Progress value={prob.probability * 100} />
                        </div>
                        <span className="text-sm font-medium w-12">
                          {(prob.probability * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Timestamp */}
              <div className="text-xs text-muted-foreground text-center">
                Predicted at: {new Date(prediction.timestamp).toLocaleString()}
              </div>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}