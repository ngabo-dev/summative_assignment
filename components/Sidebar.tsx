import React from 'react';
import { 
  BarChart3, 
  Image, 
  Upload, 
  PieChart, 
  Settings, 
  Activity,
  Flower2
} from 'lucide-react';
import { cn } from './ui/utils';

type Page = 'dashboard' | 'single-prediction' | 'bulk-upload' | 'data-insights' | 'model-management' | 'system-monitoring';

interface SidebarProps {
  currentPage: Page;
  onPageChange: (page: Page) => void;
}

const navigationItems = [
  {
    id: 'dashboard' as Page,
    label: 'Dashboard',
    icon: BarChart3,
    description: 'Overview & Metrics'
  },
  {
    id: 'single-prediction' as Page,
    label: 'Single Prediction',
    icon: Image,
    description: 'Upload & Classify'
  },
  {
    id: 'bulk-upload' as Page,
    label: 'Bulk Upload',
    icon: Upload,
    description: 'Training & Retraining'
  },
  {
    id: 'data-insights' as Page,
    label: 'Data Insights',
    icon: PieChart,
    description: 'Analytics & Trends'
  },
  {
    id: 'model-management' as Page,
    label: 'Model Management',
    icon: Settings,
    description: 'Versions & Deployment'
  },
  {
    id: 'system-monitoring' as Page,
    label: 'System Monitoring',
    icon: Activity,
    description: 'Logs & Performance'
  }
];

export function Sidebar({ currentPage, onPageChange }: SidebarProps) {
  return (
    <div className="fixed left-0 top-0 h-screen w-64 bg-sidebar border-r border-sidebar-border">
      <div className="p-6 border-b border-sidebar-border">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-primary rounded-lg flex items-center justify-center">
            <Flower2 className="w-6 h-6 text-primary-foreground" />
          </div>
          <div>
            <h2 className="font-medium text-sidebar-foreground">FlowerAI</h2>
            <p className="text-sm text-sidebar-foreground/60">ML Platform</p>
          </div>
        </div>
      </div>
      
      <nav className="p-4 space-y-2">
        {navigationItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentPage === item.id;
          
          return (
            <button
              key={item.id}
              onClick={() => onPageChange(item.id)}
              className={cn(
                "w-full flex items-center gap-3 px-3 py-3 rounded-lg text-left transition-colors",
                isActive
                  ? "bg-sidebar-accent text-sidebar-accent-foreground"
                  : "text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
              )}
            >
              <Icon className="w-5 h-5 flex-shrink-0" />
              <div className="min-w-0 flex-1">
                <div className="font-medium truncate">{item.label}</div>
                <div className="text-xs text-muted-foreground truncate">
                  {item.description}
                </div>
              </div>
            </button>
          );
        })}
      </nav>
      
      <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-sidebar-border">
        <div className="text-xs text-sidebar-foreground/60">
          Model Status: <span className="text-green-500">●</span> Online
        </div>
        <div className="text-xs text-sidebar-foreground/60">
          Version: v2.1.3
        </div>
      </div>
    </div>
  );
}