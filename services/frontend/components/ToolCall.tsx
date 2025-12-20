'use client';

import React, { useState } from 'react';
import { Wrench, ChevronDown, ChevronRight, CheckCircle, AlertCircle, BarChart3, Database, FileText, TrendingUp } from 'lucide-react';
import styles from './ToolCall.module.css';

interface ToolCallProps {
    toolOutput: any;
    index: number;
}

// Tool name to icon mapping
const getToolIcon = (toolName: string) => {
    const name = toolName.toLowerCase();
    if (name.includes('analyze') || name.includes('visualize')) return BarChart3;
    if (name.includes('data') || name.includes('report')) return FileText;
    if (name.includes('predict') || name.includes('risk')) return TrendingUp;
    if (name.includes('query') || name.includes('sql')) return Database;
    return Wrench;
};

// Format value for display
const formatValue = (value: any): string => {
    if (value === null || value === undefined) return 'N/A';
    if (typeof value === 'number') {
        if (Math.abs(value) < 1) return value.toFixed(4);
        if (Math.abs(value) >= 1000000) return `${(value / 1000000).toFixed(2)}M`;
        if (Math.abs(value) >= 1000) return `${(value / 1000).toFixed(2)}K`;
        return value.toFixed(2);
    }
    if (typeof value === 'boolean') return value ? 'Yes' : 'No';
    return String(value);
};

export default function ToolCall({ toolOutput, index }: ToolCallProps) {
    const [isExpanded, setIsExpanded] = useState(false);

    // Extract tool name and arguments
    const toolName = toolOutput.tool_name || 'Unknown Tool';
    const toolArgs = toolOutput.tool_args || toolOutput.args || {};

    // Check if this is just metadata (tool call) or has actual output
    const hasOutput = Object.keys(toolOutput).some(
        key => !['tool_name', 'tool_args', 'args'].includes(key)
    );

    // Check for errors
    const hasError = toolOutput.error || toolOutput.status === 'error';

    // Format the output (exclude tool metadata)
    const outputData = { ...toolOutput };
    delete outputData.tool_name;
    delete outputData.tool_args;
    delete outputData.args;

    // Get a display-friendly tool name
    const displayToolName = toolName
        .replace(/_/g, ' ')
        .split(' ')
        .map((word: string) => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');

    // Get appropriate icon
    const IconComponent = getToolIcon(toolName);

    // Render statistics in a nice format
    const renderStatistics = (stats: any) => {
        if (!stats || typeof stats !== 'object') return null;

        return (
            <div className={styles.statisticsGrid}>
                {Object.entries(stats).map(([key, value]) => {
                    // Skip complex nested objects for now
                    if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
                        return (
                            <div key={key} className={styles.statGroup}>
                                <strong className={styles.statGroupTitle}>{key.replace(/_/g, ' ')}:</strong>
                                <div className={styles.statNested}>
                                    {Object.entries(value).map(([k, v]) => (
                                        <span key={k} className={styles.statItem}>
                                            {k}: <span className={styles.statValue}>{formatValue(v)}</span>
                                        </span>
                                    ))}
                                </div>
                            </div>
                        );
                    }
                    return (
                        <div key={key} className={styles.statItem}>
                            <span className={styles.statLabel}>{key.replace(/_/g, ' ')}:</span>
                            <span className={styles.statValue}>{formatValue(value)}</span>
                        </div>
                    );
                })}
            </div>
        );
    };

    return (
        <div className={`${styles.toolCallContainer} ${hasError ? styles.hasError : ''}`}>
            <button
                className={styles.toolCallHeader}
                onClick={() => setIsExpanded(!isExpanded)}
            >
                <div className={styles.headerLeft}>
                    <IconComponent className={styles.toolIcon} size={16} />
                    <span className={styles.toolName}>{displayToolName}</span>
                    {hasError ? (
                        <AlertCircle className={styles.errorIcon} size={14} />
                    ) : hasOutput ? (
                        <CheckCircle className={styles.successIcon} size={14} />
                    ) : null}
                </div>
                <div className={styles.headerRight}>
                    {isExpanded ? (
                        <ChevronDown size={16} />
                    ) : (
                        <ChevronRight size={16} />
                    )}
                </div>
            </button>

            {isExpanded && (
                <div className={styles.toolCallContent}>
                    {/* Tool Arguments */}
                    {Object.keys(toolArgs).length > 0 && (
                        <div className={styles.section}>
                            <p className={styles.sectionTitle}>Arguments:</p>
                            <div className={styles.argsGrid}>
                                {Object.entries(toolArgs).map(([key, value]) => (
                                    <div key={key} className={styles.argItem}>
                                        <span className={styles.argKey}>{key}:</span>
                                        <span className={styles.argValue}>
                                            {Array.isArray(value)
                                                ? value.join(', ')
                                                : typeof value === 'object'
                                                    ? JSON.stringify(value)
                                                    : String(value)}
                                        </span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Error Message */}
                    {outputData.error && (
                        <div className={styles.section}>
                            <p className={styles.sectionTitle}>Error:</p>
                            <div className={styles.errorMessage}>
                                {outputData.error}
                            </div>
                        </div>
                    )}

                    {/* Tool Output */}
                    {hasOutput && Object.keys(outputData).length > 0 && !outputData.error && (
                        <div className={styles.section}>
                            <p className={styles.sectionTitle}>Result:</p>
                            <div className={styles.outputContent}>
                                {/* Analysis Type */}
                                {outputData.analysis_type && (
                                    <p className={styles.outputField}>
                                        <strong>Analysis Type:</strong>
                                        <span className={styles.badge}>{outputData.analysis_type}</span>
                                    </p>
                                )}

                                {/* Chart Type */}
                                {outputData.chart_type && (
                                    <p className={styles.outputField}>
                                        <strong>Chart Type:</strong>
                                        <span className={styles.badge}>{outputData.chart_type}</span>
                                    </p>
                                )}

                                {/* Features */}
                                {outputData.features && Array.isArray(outputData.features) && (
                                    <p className={styles.outputField}>
                                        <strong>Features:</strong> {outputData.features.join(', ')}
                                    </p>
                                )}

                                {/* Applicant ID */}
                                {outputData.applicant_id && (
                                    <p className={styles.outputField}>
                                        <strong>Applicant ID:</strong> {outputData.applicant_id}
                                    </p>
                                )}

                                {/* Risk Probability */}
                                {outputData.probability !== undefined && (
                                    <p className={styles.outputField}>
                                        <strong>Risk Probability:</strong>
                                        <span className={outputData.probability > 0.5 ? styles.highRisk : styles.lowRisk}>
                                            {(outputData.probability * 100).toFixed(2)}%
                                        </span>
                                    </p>
                                )}

                                {/* Insights */}
                                {outputData.insights && (
                                    <div className={styles.insightsBox}>
                                        <strong>Insights:</strong>
                                        <div className={styles.insightsText}>
                                            {outputData.insights}
                                        </div>
                                    </div>
                                )}

                                {/* Statistics */}
                                {outputData.statistics && (
                                    <div className={styles.outputField}>
                                        <strong>Statistics:</strong>
                                        {renderStatistics(outputData.statistics)}
                                    </div>
                                )}

                                {/* Top Factors */}
                                {outputData.top_factors && (
                                    <div className={styles.outputField}>
                                        <strong>Top Risk Factors:</strong>
                                        <ul className={styles.factorList}>
                                            {outputData.top_factors.slice(0, 5).map((factor: any, idx: number) => (
                                                <li key={idx}>
                                                    {factor.feature || factor.name}: {formatValue(factor.impact || factor.impact_percentage_points)}
                                                </li>
                                            ))}
                                        </ul>
                                    </div>
                                )}

                                {/* Fallback: Show raw JSON for unknown structures */}
                                {!outputData.analysis_type &&
                                    !outputData.applicant_id &&
                                    !outputData.top_factors &&
                                    !outputData.statistics &&
                                    !outputData.insights && (
                                        <pre className={styles.codeBlock}>
                                            {JSON.stringify(outputData, null, 2)}
                                        </pre>
                                    )}
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
