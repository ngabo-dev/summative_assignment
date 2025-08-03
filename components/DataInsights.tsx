import React from 'react';
import { Card } from './ui/card';
import { Badge } from './ui/badge';
import { TrendingUp, TrendingDown, Users, Calendar } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line } from 'recharts';

export function DataInsights() {
  const classDistribution = [
    { name: 'Roses', count: 1247, percentage: 44.2, color: 'hsl(var(--chart-1))' },
    { name: 'Tulips', count: 956, percentage: 33.9, color: 'hsl(var(--chart-2))' },
    { name: 'Sunflowers', count: 644, percentage: 22.8, color: 'hsl(var(--chart-3))' }
  ];

  const confidenceData = [
    { range: '90-100%', count: 1456, percentage: 51.6 },
    { range: '80-90%', count: 892, percentage: 31.6 },
    { range: '70-80%', count: 334, percentage: 11.8 },
    { range: '60-70%', count: 142, percentage: 5.0 }
  ];

  const uploadTrends = [
    { date: '2024-01-01', roses: 45, tulips: 32, sunflowers: 28 },
    { date: '2024-01-02', roses: 52, tulips: 38, sunflowers: 31 },
    { date: '2024-01-03', roses: 48, tulips: 41, sunflowers: 25 },
    { date: '2024-01-04', roses: 61, tulips: 35, sunflowers: 33 },
    { date: '2024-01-05', roses: 55, tulips: 42, sunflowers: 29 },
    { date: '2024-01-06', roses: 58, tulips: 45, sunflowers: 35 },
    { date: '2024-01-07', roses: 63, tulips: 39, sunflowers: 31 }
  ];

  const topPredictions = [
    { 
      image: 'rose_001.jpg', 
      class: 'Rose', 
      confidence: 98.7, 
      timestamp: '2 mins ago',
      user: 'Admin'
    },
    { 
      image: 'tulip_045.jpg', 
      class: 'Tulip', 
      confidence: 97.3, 
      timestamp: '5 mins ago',
      user: 'DataTeam'
    },
    { 
      image: 'sunflower_023.jpg', 
      class: 'Sunflower', 
      confidence: 96.8, 
      timestamp: '8 mins ago',
      user: 'MLTeam'
    },
    { 
      image: 'rose_078.jpg', 
      class: 'Rose', 
      confidence: 95.9, 
      timestamp: '12 mins ago',
      user: 'Admin'
    }
  ];

  const leastConfident = [
    { 
      image: 'mixed_001.jpg', 
      class: 'Rose', 
      confidence: 62.4, 
      timestamp: '1 hour ago',
      user: 'DataTeam'
    },
    { 
      image: 'unclear_012.jpg', 
      class: 'Tulip', 
      confidence: 64.7, 
      timestamp: '2 hours ago',
      user: 'MLTeam'
    },
    { 
      image: 'partial_034.jpg', 
      class: 'Sunflower', 
      confidence: 67.2, 
      timestamp: '3 hours ago',
      user: 'Admin'
    }
  ];

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-medium mb-2">Data Insights & Analytics</h2>
        <p className="text-muted-foreground">Comprehensive analysis of model performance and data patterns</p>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Total Predictions</p>
              <p className="text-2xl font-medium">2,847</p>
            </div>
            <div className="flex items-center text-green-600">
              <TrendingUp className="w-4 h-4 mr-1" />
              <span className="text-sm">+12%</span>
            </div>
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Avg Confidence</p>
              <p className="text-2xl font-medium">87.3%</p>
            </div>
            <div className="flex items-center text-green-600">
              <TrendingUp className="w-4 h-4 mr-1" />
              <span className="text-sm">+2.1%</span>
            </div>
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Active Users</p>
              <p className="text-2xl font-medium">23</p>
            </div>
            <div className="flex items-center text-red-600">
              <TrendingDown className="w-4 h-4 mr-1" />
              <span className="text-sm">-3</span>
            </div>
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Daily Uploads</p>
              <p className="text-2xl font-medium">156</p>
            </div>
            <div className="flex items-center text-green-600">
              <TrendingUp className="w-4 h-4 mr-1" />
              <span className="text-sm">+8%</span>
            </div>
          </div>
        </Card>
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Class Distribution */}
        <Card className="p-6">
          <h3 className="font-medium mb-4">Class Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={classDistribution}
                cx="50%"
                cy="50%"
                outerRadius={80}
                fill="#8884d8"
                dataKey="count"
                label={({ name, percentage }) => `${name} ${percentage}%`}
              >
                {classDistribution.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip 
                contentStyle={{
                  backgroundColor: 'var(--card)',
                  border: '1px solid var(--border)',
                  borderRadius: '8px',
                  fontSize: '12px'
                }}
              />
            </PieChart>
          </ResponsiveContainer>
          <div className="grid grid-cols-3 gap-4 mt-4">
            {classDistribution.map((item, index) => (
              <div key={index} className="text-center">
                <div className="font-medium">{item.count}</div>
                <div className="text-sm text-muted-foreground">{item.name}</div>
              </div>
            ))}
          </div>
        </Card>

        {/* Confidence Distribution */}
        <Card className="p-6">
          <h3 className="font-medium mb-4">Prediction Confidence Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={confidenceData}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis 
                dataKey="range" 
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
              <Bar dataKey="count" fill="hsl(var(--chart-1))" />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* Upload Trends */}
      <Card className="p-6">
        <h3 className="font-medium mb-4">Upload Trends (Last 7 Days)</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={uploadTrends}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
            <XAxis 
              dataKey="date" 
              className="text-muted-foreground"
              fontSize={12}
              tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
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
              labelFormatter={(value) => new Date(value).toLocaleDateString()}
            />
            <Line 
              type="monotone" 
              dataKey="roses" 
              stroke="hsl(var(--chart-1))" 
              strokeWidth={2}
              name="Roses"
            />
            <Line 
              type="monotone" 
              dataKey="tulips" 
              stroke="hsl(var(--chart-2))" 
              strokeWidth={2}
              name="Tulips"
            />
            <Line 
              type="monotone" 
              dataKey="sunflowers" 
              stroke="hsl(var(--chart-3))" 
              strokeWidth={2}
              name="Sunflowers"
            />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      {/* Prediction Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Most Confident Predictions */}
        <Card className="p-6">
          <h3 className="font-medium mb-4">Most Confident Predictions</h3>
          <div className="space-y-3">
            {topPredictions.map((pred, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 bg-primary/10 rounded-full flex items-center justify-center">
                    <span className="text-xs font-medium">{index + 1}</span>
                  </div>
                  <div>
                    <div className="font-medium text-sm">{pred.image}</div>
                    <div className="text-xs text-muted-foreground">
                      {pred.class} • {pred.user}
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <Badge variant="secondary" className="text-xs">
                    {pred.confidence}%
                  </Badge>
                  <div className="text-xs text-muted-foreground mt-1">
                    {pred.timestamp}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Card>

        {/* Least Confident Predictions */}
        <Card className="p-6">
          <h3 className="font-medium mb-4">Least Confident Predictions</h3>
          <div className="space-y-3">
            {leastConfident.map((pred, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 bg-yellow-100 dark:bg-yellow-900/20 rounded-full flex items-center justify-center">
                    <span className="text-xs font-medium text-yellow-600">!</span>
                  </div>
                  <div>
                    <div className="font-medium text-sm">{pred.image}</div>
                    <div className="text-xs text-muted-foreground">
                      {pred.class} • {pred.user}
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <Badge variant="outline" className="text-xs">
                    {pred.confidence}%
                  </Badge>
                  <div className="text-xs text-muted-foreground mt-1">
                    {pred.timestamp}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* Feature Importance */}
      <Card className="p-6">
        <h3 className="font-medium mb-4">Model Feature Analysis</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center p-4 border rounded-lg">
            <div className="text-2xl font-medium text-blue-600">Petals</div>
            <div className="text-sm text-muted-foreground">Primary Feature</div>
            <div className="text-xs mt-2">Shape, color, and arrangement patterns are the most influential factors in classification</div>
          </div>
          <div className="text-center p-4 border rounded-lg">
            <div className="text-2xl font-medium text-green-600">Leaves</div>
            <div className="text-sm text-muted-foreground">Secondary Feature</div>
            <div className="text-xs mt-2">Leaf structure and venation patterns provide important contextual information</div>
          </div>
          <div className="text-center p-4 border rounded-lg">
            <div className="text-2xl font-medium text-orange-600">Stem</div>
            <div className="text-sm text-muted-foreground">Supporting Feature</div>
            <div className="text-xs mt-2">Stem thickness and color contribute to overall classification accuracy</div>
          </div>
        </div>
      </Card>
    </div>
  );
}