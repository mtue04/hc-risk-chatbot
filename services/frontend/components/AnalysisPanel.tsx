import React, { useEffect, useState } from 'react';
import { FileText, CheckCircle, Loader, AlertCircle, PlayCircle } from 'lucide-react';
import AnalysisStepResult from './AnalysisStepResult';

interface AnalysisPlan {
  steps: Array<{
    step_number: number;
    description: string;
    chart_type: string;
  }>;
}

interface StepResult {
  step_number: number;
  sql_query: string;
  data_summary: string;
  chart_type: string;
  chart_image_path: string;
  chart_image_base64?: string;
  insights: string;
}

interface AnalysisStatus {
  thread_id: string;
  status: string;
  plan?: AnalysisPlan;
  step_results?: StepResult[];
  final_summary?: string;
  current_step?: number;
}

interface AnalysisPanelProps {
  threadId: string | null;
  onClose?: () => void;
}

const AnalysisPanel: React.FC<AnalysisPanelProps> = ({ threadId, onClose }) => {
  const [status, setStatus] = useState<AnalysisStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Poll for status updates
  useEffect(() => {
    if (!threadId) return;

    const fetchStatus = async () => {
      try {
        setLoading(true);
        const response = await fetch(`http://localhost:8500/analysis/${threadId}/status`);

        if (!response.ok) {
          throw new Error(`Failed to fetch status: ${response.statusText}`);
        }

        const data: AnalysisStatus = await response.json();
        setStatus(data);
        setError(null);

        // Stop polling if workflow is completed or failed
        if (data.status === 'completed' || data.status === 'failed') {
          return;
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch analysis status');
      } finally {
        setLoading(false);
      }
    };

    // Initial fetch
    fetchStatus();

    // Poll every 2 seconds while workflow is running
    const intervalId = setInterval(() => {
      if (status?.status !== 'completed' && status?.status !== 'failed') {
        fetchStatus();
      }
    }, 2000);

    return () => clearInterval(intervalId);
  }, [threadId, status?.status]);

  if (!threadId) {
    return null;
  }

  if (error) {
    return (
      <div className="border border-red-500/50 rounded-lg p-4 bg-red-900/20">
        <div className="flex items-center gap-2 text-red-400">
          <AlertCircle className="w-5 h-5" />
          <span>{error}</span>
        </div>
      </div>
    );
  }

  if (!status) {
    return (
      <div className="flex items-center justify-center p-8">
        <Loader className="w-6 h-6 animate-spin text-blue-500" />
        <span className="ml-2 text-gray-400">Loading analysis...</span>
      </div>
    );
  }

  const getStatusIcon = () => {
    switch (status.status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'failed':
        return <AlertCircle className="w-5 h-5 text-red-500" />;
      case 'running':
      case 'executing':
        return <Loader className="w-5 h-5 text-blue-500 animate-spin" />;
      default:
        return <PlayCircle className="w-5 h-5 text-gray-500" />;
    }
  };

  const getStatusColor = () => {
    switch (status.status) {
      case 'completed':
        return 'text-green-400';
      case 'failed':
        return 'text-red-400';
      case 'running':
      case 'executing':
        return 'text-blue-400';
      default:
        return 'text-gray-400';
    }
  };

  return (
    <div className="border border-gray-700 rounded-lg p-6 bg-gray-800/30 mb-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <FileText className="w-5 h-5 text-blue-400" />
          <h2 className="text-lg font-semibold text-white">Analysis Workflow</h2>
        </div>
        <div className="flex items-center gap-2">
          {getStatusIcon()}
          <span className={`text-sm font-medium ${getStatusColor()}`}>
            {status.status.charAt(0).toUpperCase() + status.status.slice(1)}
          </span>
        </div>
      </div>

      {/* Analysis Plan */}
      {status.plan && (
        <div className="mb-6">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">Analysis Plan:</h3>
          <div className="space-y-2">
            {status.plan.steps.map((step) => (
              <div
                key={step.step_number}
                className={`flex items-start gap-3 p-3 rounded-lg ${
                  status.current_step && step.step_number <= status.current_step
                    ? 'bg-blue-900/20 border border-blue-500/30'
                    : 'bg-gray-900/50'
                }`}
              >
                <div className="flex-shrink-0 w-6 h-6 rounded-full bg-gray-700 flex items-center justify-center text-xs text-white">
                  {step.step_number}
                </div>
                <div className="flex-1">
                  <p className="text-sm text-gray-300">{step.description}</p>
                  <p className="text-xs text-gray-500 mt-1">Chart: {step.chart_type}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Step Results */}
      {status.step_results && status.step_results.length > 0 && (
        <div className="mb-6">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">Results:</h3>
          {status.step_results.map((step) => (
            <AnalysisStepResult
              key={step.step_number}
              step={step}
              isCompleted={true}
            />
          ))}
        </div>
      )}

      {/* Final Summary */}
      {status.final_summary && (
        <div className="border-t border-gray-700 pt-4">
          <h3 className="text-sm font-semibold text-gray-300 mb-2">Executive Summary:</h3>
          <div className="bg-blue-900/10 border border-blue-500/30 rounded-lg p-4">
            <p className="text-sm text-gray-300 whitespace-pre-wrap">{status.final_summary}</p>
          </div>
        </div>
      )}

      {/* Close Button */}
      {onClose && status.status === 'completed' && (
        <div className="mt-4 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg text-sm transition-colors"
          >
            Close Analysis
          </button>
        </div>
      )}
    </div>
  );
};

export default AnalysisPanel;
