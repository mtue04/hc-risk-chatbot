'use client';

import React from 'react';
import ReactMarkdown from 'react-markdown';
import { User, Bot } from 'lucide-react';
import styles from './ChatMessage.module.css';
import ChartRenderer from './charts/ChartRenderer';
import ToolCall from './ToolCall';
import AnalysisStepResult from './AnalysisStepResult';

export interface Message {
    role: 'user' | 'assistant';
    content: string;
    timestamp?: string;
    charts?: any[]; // Chart data from backend
    tool_outputs?: any[]; // Raw tool outputs from backend
    thinking?: string; // Thinking/reasoning process
    analysis_steps?: Array<{
        step_number: number;
        description?: string;
        data_summary?: string;
        chart_type: string;
        chart_image_base64?: string;
        insights: string;
    }>;
}

interface ChatMessageProps {
    message: Message;
    index: number;
}

export default function ChatMessage({ message, index }: ChatMessageProps) {
    const isUser = message.role === 'user';

    return (
        <div className={`${styles.messageContainer} ${isUser ? styles.user : styles.assistant}`}>
            {/* Avatar */}
            <div className={`${styles.avatar} ${isUser ? styles.avatarUser : styles.avatarBot}`}>
                {isUser ? <User size={18} /> : <Bot size={18} />}
            </div>

            {/* Message Content */}
            <div className={`${styles.bubble} ${isUser ? styles.bubbleUser : styles.bubbleBot}`}>
                {isUser ? (
                    <p className={styles.text}>{message.content}</p>
                ) : (
                    <div className={styles.markdown}>
                        <ReactMarkdown
                            components={{
                                p: ({ children }) => <p className={styles.paragraph}>{children}</p>,
                                strong: ({ children }) => <strong className={styles.bold}>{children}</strong>,
                                em: ({ children }) => <em className={styles.italic}>{children}</em>,
                                code: ({ children }) => <code className={styles.code}>{children}</code>,
                                pre: ({ children }) => <pre className={styles.pre}>{children}</pre>,
                                ul: ({ children }) => <ul className={styles.list}>{children}</ul>,
                                ol: ({ children }) => <ol className={styles.orderedList}>{children}</ol>,
                                li: ({ children }) => <li className={styles.listItem}>{children}</li>,
                                h1: ({ children }) => <h1 className={styles.heading1}>{children}</h1>,
                                h2: ({ children }) => <h2 className={styles.heading2}>{children}</h2>,
                                h3: ({ children }) => <h3 className={styles.heading3}>{children}</h3>,
                            }}
                        >
                            {message.content}
                        </ReactMarkdown>
                    </div>
                )}

                {/* Tool Calls - only for assistant messages */}
                {!isUser && message.tool_outputs && message.tool_outputs.length > 0 && (
                    <div className={styles.toolCallsContainer}>
                        {message.tool_outputs.map((toolOutput, toolIndex) => (
                            <ToolCall key={`${index}-tool-${toolIndex}`} toolOutput={toolOutput} index={toolIndex} />
                        ))}
                    </div>
                )}

                {/* Charts - only for assistant messages */}
                {!isUser && message.charts && message.charts.length > 0 && (
                    <div className={styles.chartsContainer}>
                        {message.charts.map((chart, chartIndex) => (
                            <ChartRenderer key={`${index}-chart-${chartIndex}`} chartData={chart} />
                        ))}
                    </div>
                )}

                {/* Analysis Steps - only for assistant messages */}
                {!isUser && message.analysis_steps && message.analysis_steps.length > 0 && (
                    <div className={styles.analysisStepsContainer} style={{ marginTop: '1rem' }}>
                        {message.analysis_steps.map((step) => (
                            <AnalysisStepResult
                                key={`${index}-step-${step.step_number}`}
                                step={{
                                    step_number: step.step_number,
                                    description: step.description,
                                    chart_type: step.chart_type,
                                    chart_image_base64: step.chart_image_base64,
                                    insights: step.insights,
                                }}
                                isCompleted={true}
                            />
                        ))}
                    </div>
                )}

                {/* Timestamp */}
                {message.timestamp && (
                    <span className={styles.timestamp}>{message.timestamp}</span>
                )}
            </div>
        </div>
    );
}
