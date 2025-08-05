import React from 'react';
import { Card } from './ui/card';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';

const data = [
  { name: 'Roses', value: 435, color: 'hsl(var(--chart-1))' },
  { name: 'Tulips', value: 312, color: 'hsl(var(--chart-2))' },
  { name: 'Sunflowers', value: 198, color: 'hsl(var(--chart-3))' },
];

export function PredictionChart() {
  return (
    <Card className="p-6">
      <h3 className="font-medium mb-4">Predictions by Class (24h)</h3>
      <ResponsiveContainer width="100%" height={300}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            outerRadius={80}
            fill="#8884d8"
            dataKey="value"
            label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
          >
            {data.map((entry, index) => (
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
      <div className="mt-4 grid grid-cols-3 gap-4 text-center">
        {data.map((item, index) => (
          <div key={index} className="text-sm">
            <div className="font-medium">{item.value}</div>
            <div className="text-muted-foreground">{item.name}</div>
          </div>
        ))}
      </div>
    </Card>
  );
}