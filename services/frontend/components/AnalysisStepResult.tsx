import React, { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { CheckCircle, Loader, ChartBar, X } from 'lucide-react';

interface StepResult {
  step_number: number;
  description?: string;
  sql_query?: string;
  data_summary?: string;
  chart_type: string;
  chart_image_path?: string;
  chart_image_base64?: string;
  insights: string;
}

interface AnalysisStepResultProps {
  step: StepResult;
  isCompleted: boolean;
}

const AnalysisStepResult: React.FC<AnalysisStepResultProps> = ({ step, isCompleted }) => {
  const [isZoomed, setIsZoomed] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    return () => setMounted(false);
  }, []);

  // Prevent body scroll when zoomed
  useEffect(() => {
    if (isZoomed) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'unset';
    }
    return () => {
      document.body.style.overflow = 'unset';
    };
  }, [isZoomed]);

  return (
    <>
      <div className="border border-gray-700 rounded-lg p-4 mb-4 bg-gray-800/50">
      <div className="flex items-center gap-2 mb-3">
        {isCompleted ? (
          <CheckCircle className="w-5 h-5 text-green-500" />
        ) : (
          <Loader className="w-5 h-5 text-blue-500 animate-spin" />
        )}
        <div className="flex-1">
          <h3 className="font-semibold text-white">
            Step {step.step_number}: {step.description || 'Analysis'}
          </h3>
        </div>
        {step.chart_type && (
          <div className="flex items-center gap-1 text-sm text-gray-400">
            <ChartBar className="w-4 h-4" />
            <span>{step.chart_type}</span>
          </div>
        )}
      </div>

      {/* Chart Image */}
      {step.chart_image_base64 && typeof step.chart_image_base64 === 'string' && (
        <div className="mb-3">
          <p className="text-xs text-gray-400 mb-2">Visualization:</p>
          <div
            className="bg-white rounded-lg p-4 w-full cursor-zoom-in"
            style={{ maxWidth: '100%', overflow: 'hidden' }}
            onClick={() => setIsZoomed(true)}
          >
            <img
              src={`data:image/png;base64,${step.chart_image_base64}`}
              alt={`Step ${step.step_number} chart`}
              style={{
                width: '100%',
                height: 'auto',
                maxHeight: '400px',
                objectFit: 'contain',
                display: 'block'
              }}
            />
          </div>
        </div>
      )}

      {/* Insights */}
      {step.insights && (
        <div>
          <p className="text-xs text-gray-400 mb-1">Insights:</p>
          <p className="text-sm text-gray-300 whitespace-pre-wrap">
            {typeof step.insights === 'string' ? step.insights : JSON.stringify(step.insights)}
          </p>
        </div>
      )}
    </div>

    {/* Zoom Modal - rendered at document body level using Portal */}
    {mounted && isZoomed && step.chart_image_base64 && createPortal(
      <div
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          width: '100vw',
          height: '100vh',
          backgroundColor: 'rgba(0, 0, 0, 0.92)',
          backdropFilter: 'blur(8px)',
          WebkitBackdropFilter: 'blur(8px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: 0,
          margin: 0,
          zIndex: 2147483647, // Maximum z-index value
          cursor: 'zoom-out'
        }}
        onClick={() => setIsZoomed(false)}
      >
        <button
          style={{
            position: 'absolute',
            top: '20px',
            right: '20px',
            background: 'rgba(0, 0, 0, 0.5)',
            border: 'none',
            borderRadius: '50%',
            width: '40px',
            height: '40px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
            color: 'white',
            zIndex: 2147483647
          }}
          onClick={() => setIsZoomed(false)}
          onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(0, 0, 0, 0.8)'}
          onMouseLeave={(e) => e.currentTarget.style.background = 'rgba(0, 0, 0, 0.5)'}
        >
          <X size={24} />
        </button>
        <img
          src={`data:image/png;base64,${step.chart_image_base64}`}
          alt={`Step ${step.step_number} chart (zoomed)`}
          style={{
            maxWidth: '80vw',
            maxHeight: '80vh',
            width: 'auto',
            height: 'auto',
            objectFit: 'contain',
            cursor: 'zoom-out',
            borderRadius: '8px',
            boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)'
          }}
          onClick={(e) => e.stopPropagation()}
        />
      </div>,
      document.body
    )}
  </>
  );
};

export default AnalysisStepResult;
