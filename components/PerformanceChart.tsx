import React from 'react';
import { Card } from './ui/card';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const data = [
  { time: '00:00', accuracy: 92.1, precision: 90.5, recall: 93.2 },
  { time: '04:00', accuracy: 93.5, precision: 91.8, recall: 94.1 },
  { time: '08:00', accuracy: 94.2, precision: 92.8, recall: 95.1 },
  { time: '12:00', accuracy: 93.8, precision: 92.1, recall: 94.8 },
  { time: '16:00', accuracy: 94.6, precision: 93.2, recall: 95.3 },
  { time: '20:00', accuracy: 94.2, precision: 92.8, recall: 95.1 },
];

export function PerformanceChart() {
  return (
    <Card className="p-6">
      <h3 className="font-medium mb-4">Performance Over Time (24h)</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
          <XAxis 
            dataKey="time" 
            className="text-muted-foreground"
            fontSize={12}
          />
          <YAxis 
            domain={[85, 100]}
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
            dataKey="accuracy" 
            stroke="hsl(var(--chart-1))" 
            strokeWidth={2}
            name="Accuracy"
          />
          <Line 
            type="monotone" 
            dataKey="precision" 
            stroke="hsl(var(--chart-2))" 
            strokeWidth={2}
            name="Precision"
          />
          <Line 
            type="monotone" 
            dataKey="recall" 
            stroke="hsl(var(--chart-3))" 
            strokeWidth={2}
            name="Recall"
          />
        </LineChart>
      </ResponsiveContainer>
    </Card>
  );
}