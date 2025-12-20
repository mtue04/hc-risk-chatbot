import React, { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { CheckCircle, Loader, ChartBar, X, AlertCircle, ZoomIn } from 'lucide-react';

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

  // Check if insights contain error
  const hasError = step.insights?.toLowerCase().includes('error') ||
    step.insights?.toLowerCase().includes('could not') ||
    step.insights?.toLowerCase().includes('failed');

  // Check if chart is available
  const hasChart = step.chart_image_base64 && typeof step.chart_image_base64 === 'string';

  return (
    <>
      <div
        style={{
          border: hasError ? '1px solid rgba(239, 68, 68, 0.4)' : '1px solid rgba(55, 65, 81, 0.8)',
          borderRadius: '12px',
          padding: '16px',
          marginBottom: '16px',
          background: hasError
            ? 'linear-gradient(135deg, rgba(239, 68, 68, 0.08) 0%, rgba(31, 41, 55, 0.5) 100%)'
            : 'linear-gradient(135deg, rgba(31, 41, 55, 0.6) 0%, rgba(31, 41, 55, 0.3) 100%)',
          backdropFilter: 'blur(8px)',
        }}
      >
        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
          {hasError ? (
            <div style={{
              width: '28px',
              height: '28px',
              borderRadius: '50%',
              background: 'rgba(239, 68, 68, 0.2)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}>
              <AlertCircle style={{ width: '16px', height: '16px', color: '#ef4444' }} />
            </div>
          ) : isCompleted ? (
            <div style={{
              width: '28px',
              height: '28px',
              borderRadius: '50%',
              background: 'rgba(34, 197, 94, 0.2)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}>
              <CheckCircle style={{ width: '16px', height: '16px', color: '#22c55e' }} />
            </div>
          ) : (
            <div style={{
              width: '28px',
              height: '28px',
              borderRadius: '50%',
              background: 'rgba(59, 130, 246, 0.2)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}>
              <Loader style={{
                width: '16px',
                height: '16px',
                color: '#3b82f6',
                animation: 'spin 1s linear infinite'
              }} />
            </div>
          )}

          <div style={{ flex: 1 }}>
            <h3 style={{
              fontWeight: 600,
              color: hasError ? '#fca5a5' : 'white',
              fontSize: '15px',
              margin: 0,
              lineHeight: 1.4
            }}>
              Step {step.step_number}: {step.description || 'Analysis'}
            </h3>
          </div>

          {step.chart_type && (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              fontSize: '12px',
              color: '#9ca3af',
              background: 'rgba(55, 65, 81, 0.5)',
              padding: '4px 10px',
              borderRadius: '12px',
            }}>
              <ChartBar style={{ width: '14px', height: '14px' }} />
              <span>{step.chart_type}</span>
            </div>
          )}
        </div>

        {/* Chart Image */}
        {hasChart && (
          <div style={{ marginBottom: '12px' }}>
            <p style={{
              fontSize: '11px',
              textTransform: 'uppercase',
              letterSpacing: '0.5px',
              color: '#6b7280',
              marginBottom: '8px',
              fontWeight: 500
            }}>
              Visualization:
            </p>
            <div
              style={{
                background: 'white',
                borderRadius: '10px',
                padding: '16px',
                width: '100%',
                cursor: 'zoom-in',
                position: 'relative',
                overflow: 'hidden',
                boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
              }}
              onClick={() => setIsZoomed(true)}
            >
              <div style={{
                position: 'absolute',
                top: '8px',
                right: '8px',
                background: 'rgba(0, 0, 0, 0.5)',
                borderRadius: '6px',
                padding: '4px 8px',
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                fontSize: '11px',
                color: 'white',
                opacity: 0.7,
                transition: 'opacity 0.2s',
              }}>
                <ZoomIn size={12} />
                Click to zoom
              </div>
              <img
                src={`data:image/png;base64,${step.chart_image_base64}`}
                alt={`Step ${step.step_number} chart`}
                style={{
                  width: '100%',
                  height: 'auto',
                  objectFit: 'contain',
                  display: 'block'
                }}
              />
            </div>
          </div>
        )}

        {/* Insights */}
        {step.insights && (
          <div style={{
            background: hasError
              ? 'rgba(239, 68, 68, 0.1)'
              : 'rgba(16, 185, 129, 0.08)',
            border: hasError
              ? '1px solid rgba(239, 68, 68, 0.2)'
              : '1px solid rgba(16, 185, 129, 0.15)',
            borderRadius: '8px',
            padding: '12px',
          }}>
            <p style={{
              fontSize: '11px',
              textTransform: 'uppercase',
              letterSpacing: '0.5px',
              color: hasError ? '#f87171' : '#10b981',
              marginBottom: '6px',
              fontWeight: 600
            }}>
              {hasError ? 'Error:' : 'Insights:'}
            </p>
            <p style={{
              fontSize: '13px',
              color: hasError ? '#fca5a5' : '#d1d5db',
              whiteSpace: 'pre-wrap',
              lineHeight: 1.6,
              margin: 0
            }}>
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
            backgroundColor: 'rgba(0, 0, 0, 0.95)',
            backdropFilter: 'blur(12px)',
            WebkitBackdropFilter: 'blur(12px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: 0,
            margin: 0,
            zIndex: 2147483647,
            cursor: 'zoom-out'
          }}
          onClick={() => setIsZoomed(false)}
        >
          <button
            style={{
              position: 'absolute',
              top: '20px',
              right: '20px',
              background: 'rgba(255, 255, 255, 0.1)',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '50%',
              width: '44px',
              height: '44px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
              color: 'white',
              zIndex: 2147483647,
              transition: 'all 0.2s ease'
            }}
            onClick={() => setIsZoomed(false)}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = 'rgba(255, 255, 255, 0.2)';
              e.currentTarget.style.transform = 'scale(1.1)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
              e.currentTarget.style.transform = 'scale(1)';
            }}
          >
            <X size={24} />
          </button>
          <img
            src={`data:image/png;base64,${step.chart_image_base64}`}
            alt={`Step ${step.step_number} chart (zoomed)`}
            style={{
              maxWidth: '90vw',
              maxHeight: '90vh',
              width: 'auto',
              height: 'auto',
              objectFit: 'contain',
              cursor: 'zoom-out',
              borderRadius: '12px',
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
