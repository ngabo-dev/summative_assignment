// Production API Types and Interfaces

export interface DashboardMetrics {
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

export interface PredictionResult {
  predicted_class: string;
  confidence: number;
  probabilities: Array<{ name: string; probability: number; color: string }>;
  response_time_ms: number;
  timestamp: string;
}

export interface ModelVersion {
  id: string;
  version: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  training_time: string;
  dataset_size: number;
  model_size: string;
  epoch_count: number;
  status: string;
  created_at: string;
  deployed_at?: string;
}

export interface TrainingStatus {
  stage: string;
  progress: number;
  eta: string;
  status: 'running' | 'completed' | 'failed' | 'cancelled';
  start_time: string;
  end_time?: string;
  error?: string;
}

export interface SystemLog {
  id: string;
  timestamp: string;
  level: 'success' | 'info' | 'warning' | 'error';
  category: string;
  message: string;
  details: string;
}

export interface DataInsights {
  class_distribution: Array<{ name: string; count: number; percentage: number }>;
  confidence_distribution: Array<{ range: string; count: number; percentage: number }>;
  upload_trends: Array<{ date: string; roses: number; tulips: number; sunflowers: number }>;
  performance_metrics: {
    total_predictions: number;
    avg_confidence: number;
    active_users: number;
    daily_uploads: number;
  };
}