import { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Activity, Database } from 'lucide-react';

export default function LanggraphStateDisplay() {
  // State to hold the current character and metadata values
  const [stateValues, setStateValues] = useState({
    character: null,
    metadata: null,
    lastUpdated: null
  });

  // Props are globally injected by Chainlit - access them directly
  // Note: props are available as global variables, not through window.chainlit
  useEffect(() => {
    // Update state values when props change
    if (typeof props !== 'undefined' && props.langgraphState) {
      setStateValues({
        character: props.langgraphState.character !== undefined ? props.langgraphState.character : null,
        metadata: props.langgraphState.metadata !== undefined ? props.langgraphState.metadata : null,
        lastUpdated: new Date().toLocaleTimeString()
      });
    }
  }, [props?.langgraphState]);

  // Function to handle manual refresh (mostly for debugging)
  const handleRefresh = () => {
    setStateValues(prev => ({
      ...prev,
      lastUpdated: new Date().toLocaleTimeString()
    }));
  };

  return (
    <Card className="w-full mb-4 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 border-blue-200">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Database className="w-5 h-5 text-blue-600" />
          <h3 className="text-lg font-semibold text-blue-900">
            Langgraph State Monitor
          </h3>
        </div>
        <div className="flex items-center gap-2">
          {stateValues.lastUpdated && (
            <span className="text-xs text-gray-500">
              Updated: {stateValues.lastUpdated}
            </span>
          )}
          <button 
            onClick={handleRefresh}
            className="p-1 hover:bg-blue-100 rounded transition-colors"
            title="Refresh state"
          >
            <Activity className="w-4 h-4 text-blue-600" />
          </button>
        </div>
      </div>
      
      <Separator className="mb-3" />
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* character Value Display */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-gray-700">Current Character:</span>
            <Badge 
              variant={stateValues.character && stateValues.character !== null ? 'default' : 'secondary'}
              className="text-xs"
            >
              {stateValues.character ? 'Active' : 'Inactive'}
            </Badge>
          </div>
          <div className="p-3 bg-white rounded-lg border border-gray-200 min-h-[40px] flex items-center">
            <code className="text-sm font-mono text-gray-800 break-all">
              {stateValues.character ? JSON.stringify(stateValues.character) : 'null'}
            </code>
          </div>
        </div>

        {/* metadata Value Display */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-gray-700">Character Metadata:</span>
            <Badge 
              variant={stateValues.metadata && stateValues.metadata !== null ? 'default' : 'secondary'}
              className="text-xs"
            >
              {stateValues.metadata ? 'Active' : 'Inactive'}
            </Badge>
          </div>
          <div className="p-3 bg-white rounded-lg border border-gray-200 min-h-[40px] flex items-center">
            <code className="text-sm font-mono text-gray-800 break-all">
              {stateValues.metadata ? JSON.stringify(stateValues.metadata) : 'null'}
            </code>
          </div>
        </div>
      </div>

      {/* Connection Status Indicator */}
      <div className="mt-4 pt-3 border-t border-blue-200">
        <div className="flex items-center gap-2 text-xs text-gray-600">
          <div className={`w-2 h-2 rounded-full ${
            (typeof props !== 'undefined' && props?.langgraphState) ? 'bg-green-500' : 'bg-red-500'
          }`} />
          <span>
            {(typeof props !== 'undefined' && props?.langgraphState) ? 'Connected to Langgraph' : 'Disconnected'}
          </span>
        </div>
      </div>
    </Card>
  );
}