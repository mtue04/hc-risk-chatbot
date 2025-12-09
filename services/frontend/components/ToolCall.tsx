'use client';

import React, { useState } from 'react';
import { Wrench, ChevronDown, ChevronRight, CheckCircle } from 'lucide-react';
import styles from './ToolCall.module.css';

interface ToolCallProps {
    toolOutput: any;
    index: number;
}

export default function ToolCall({ toolOutput, index }: ToolCallProps) {
    const [isExpanded, setIsExpanded] = useState(false);

    // Extract tool name and arguments
    const toolName = toolOutput.tool_name || 'Unknown Tool';
    const toolArgs = toolOutput.tool_args || toolOutput.args || {};

    // Check if this is just metadata (tool call) or has actual output
    const hasOutput = Object.keys(toolOutput).some(
        key => !['tool_name', 'tool_args', 'args'].includes(key)
    );

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

    return (
        <div className={styles.toolCallContainer}>
            <button
                className={styles.toolCallHeader}
                onClick={() => setIsExpanded(!isExpanded)}
            >
                <div className={styles.headerLeft}>
                    <Wrench className={styles.toolIcon} size={16} />
                    <span className={styles.toolName}>{displayToolName}</span>
                    {hasOutput && <CheckCircle className={styles.successIcon} size={14} />}
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
                            <pre className={styles.codeBlock}>
                                {JSON.stringify(toolArgs, null, 2)}
                            </pre>
                        </div>
                    )}

                    {/* Tool Output */}
                    {hasOutput && Object.keys(outputData).length > 0 && (
                        <div className={styles.section}>
                            <p className={styles.sectionTitle}>Result:</p>
                            <div className={styles.outputContent}>
                                {/* Special handling for common fields */}
                                {outputData.applicant_id && (
                                    <p className={styles.outputField}>
                                        <strong>Applicant ID:</strong> {outputData.applicant_id}
                                    </p>
                                )}
                                {outputData.probability !== undefined && (
                                    <p className={styles.outputField}>
                                        <strong>Risk Probability:</strong> {(outputData.probability * 100).toFixed(2)}%
                                    </p>
                                )}
                                {outputData.top_factors && (
                                    <div className={styles.outputField}>
                                        <strong>Top Risk Factors:</strong>
                                        <ul className={styles.factorList}>
                                            {outputData.top_factors.slice(0, 5).map((factor: any, idx: number) => (
                                                <li key={idx}>
                                                    {factor.feature || factor.name}: {factor.impact?.toFixed(2) || factor.impact_percentage_points?.toFixed(2)}
                                                </li>
                                            ))}
                                        </ul>
                                    </div>
                                )}

                                {/* Show raw JSON for other data */}
                                {!outputData.applicant_id && !outputData.top_factors && (
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
