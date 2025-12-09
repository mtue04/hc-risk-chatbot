'use client';

import React, { useState, useRef, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { Sparkles, TrendingUp, Users, HelpCircle, RotateCcw } from 'lucide-react';
import Sidebar, { ChatHistory } from '@/components/Sidebar';
import ChatMessage, { Message } from '@/components/ChatMessage';
import ChatInput from '@/components/ChatInput';
import AnalysisPanel from '@/components/AnalysisPanel';
import { sendMessage, sendMultimodal, sendMessageStream, StreamEvent } from '@/lib/api';
import styles from './page.module.css';
import { ChartData } from '@/components/charts/ChartRenderer';

// Parse tool outputs from backend and convert to frontend chart format
function parseChartsFromToolOutputs(toolOutputs: any[] | undefined): ChartData[] {
    if (!toolOutputs || !Array.isArray(toolOutputs)) return [];

    const charts: ChartData[] = [];

    for (const output of toolOutputs) {
        // Skip tool call metadata (only has tool_name, tool_args)
        if (output.tool_name && !output.probability && !output.statistics && !output.top_factors) {
            continue;
        }

        // Handle explicit chart_data
        if (output.chart_data) {
            charts.push(output.chart_data);
            continue;
        }

        // Handle SHAP explanations (from get_risk_prediction or explain_shap_values)
        if (output.top_factors && Array.isArray(output.top_factors)) {
            // Convert SHAP factors to feature importance chart
            const features: Record<string, number> = {};
            output.top_factors.forEach((factor: any) => {
                const name = factor.feature || factor.name;
                const impact = factor.impact || factor.impact_percentage_points || 0;
                if (name) features[name] = impact;
            });

            if (Object.keys(features).length > 0) {
                charts.push({
                    type: 'feature_importance',
                    title: `🔍 Risk Factor Analysis (Applicant ${output.applicant_id || ''})`,
                    data: { features },
                });
            }

            // Also add a risk gauge if probability is available
            if (output.probability !== undefined) {
                charts.push({
                    type: 'risk_gauge',
                    title: 'Risk Score',
                    data: { score: output.probability },
                });
            }
        }

        // Handle analyze_and_visualize output
        if (output.chart_type && output.statistics) {
            if (output.chart_type === 'feature_importance' && output.statistics.shap_values) {
                charts.push({
                    type: 'feature_importance',
                    title: output.insights || 'Feature Importance',
                    data: { features: output.statistics.shap_values },
                });
            } else if (output.chart_type === 'histogram' && output.statistics) {
                // Use pre-computed histogram bins from backend
                const feature = output.features?.[0] || 'Value';
                const stats = output.statistics[feature];
                if (stats && stats.histogram_bins) {
                    // Pass histogram_bins directly to Histogram component
                    charts.push({
                        type: 'histogram',
                        title: `📊 ${feature} Distribution`,
                        data: { histogram_bins: stats.histogram_bins },
                    });
                } else if (stats) {
                    // Fallback: show summary statistics as bar chart
                    charts.push({
                        type: 'bar',
                        title: `📊 ${feature} Statistics`,
                        data: {
                            categories: ['Q25', 'Median', 'Mean', 'Q75'],
                            values: [stats.q25 || 0, stats.median || 0, stats.mean || 0, stats.q75 || 0],
                        },
                    });
                }
            } else if (output.chart_type === 'grouped_bar' && output.statistics) {
                // Transform backend grouped data to frontend format
                // Backend: { "Group1": { "Feature1": { mean, std }, ... }, "Group2": {...} }
                // Frontend needs: { label: value } for simple bar

                const groups = Object.keys(output.statistics);
                if (groups.length >= 2) {
                    // Get all features from first group
                    const firstGroupData = output.statistics[groups[0]];
                    const features = Object.keys(firstGroupData || {});

                    // Create a simple bar chart showing mean values per group per feature
                    features.forEach(feature => {
                        const barData: Record<string, number> = {};
                        groups.forEach(group => {
                            const groupStats = output.statistics[group]?.[feature];
                            if (groupStats?.mean !== undefined) {
                                // Use shorter labels
                                const shortLabel = group.includes('Non-Defaulter') ? 'Non-Defaulter' :
                                    group.includes('Defaulter') ? 'Defaulter' : group;
                                barData[shortLabel] = groupStats.mean;
                            }
                        });

                        if (Object.keys(barData).length > 0) {
                            charts.push({
                                type: 'bar',
                                title: `📊 ${feature} by Risk Group`,
                                data: barData,
                            });
                        }
                    });
                }
            } else if (output.chart_type === 'scatter' && output.chart_data) {
                // Scatter plot with actual x/y data
                charts.push({
                    type: 'scatter',
                    title: `📈 Correlation: ${output.statistics?.feature_x || 'X'} vs ${output.statistics?.feature_y || 'Y'} (r=${(output.statistics?.correlation || 0).toFixed(3)})`,
                    data: output.chart_data,
                });
            } else if (output.chart_type === 'heatmap' && output.chart_data) {
                // Heatmap for correlation matrix
                charts.push({
                    type: 'heatmap',
                    title: `🗺️ Correlation Matrix`,
                    data: output.chart_data,
                });
            } else if (output.chart_type === 'radar' && output.chart_data) {
                // Radar chart for comparison
                charts.push({
                    type: 'comparison_radar',
                    title: output.title || '📊 Comparison',
                    data: output.chart_data,
                });
            } else if (output.chart_type === 'pie' && output.chart_data) {
                // Pie chart for composition
                charts.push({
                    type: 'pie',
                    title: output.title || '📊 Distribution',
                    data: output.chart_data,
                });
            } else if (output.chart_type === 'waterfall' && output.chart_data) {
                // Waterfall chart for breakdown
                charts.push({
                    type: 'waterfall',
                    title: output.title || '📊 Risk Breakdown',
                    data: output.chart_data,
                });
            } else if (output.chart_type === 'funnel' && output.chart_data) {
                // Funnel chart
                charts.push({
                    type: 'funnel',
                    title: output.title || '📊 Funnel Analysis',
                    data: output.chart_data,
                });
            }
        }

        // Handle generate_feature_plot output
        if (output.features && output.plot_type === 'feature_comparison') {
            const features: Record<string, number> = {};
            Object.entries(output.features).forEach(([name, stats]: [string, any]) => {
                if (stats.applicant_value !== undefined) {
                    features[name] = stats.applicant_value;
                }
            });

            if (Object.keys(features).length > 0) {
                charts.push({
                    type: 'bar',
                    title: `📊 Applicant ${output.applicant_id} Feature Values`,
                    data: features,
                });
            }
        }

        // Handle query_applicant_data output
        if (output.applicant_id && output.AMT_INCOME_TOTAL !== undefined) {
            // This is applicant data query result - create bar chart of key metrics
            const metrics: Record<string, number> = {};

            if (output.AMT_INCOME_TOTAL) metrics['Income'] = output.AMT_INCOME_TOTAL;
            if (output.AMT_CREDIT) metrics['Credit'] = output.AMT_CREDIT;
            if (output.AMT_ANNUITY) metrics['Annuity'] = output.AMT_ANNUITY;

            if (Object.keys(metrics).length > 0) {
                charts.push({
                    type: 'bar',
                    title: `📊 Applicant ${output.applicant_id} Financial Profile`,
                    data: metrics,
                });
            }
        }

        // Handle generate_data_report output
        if (output.report_type && output.profile) {
            // Risk gauge from probability
            if (output.risk_analysis?.probability !== undefined) {
                charts.push({
                    type: 'risk_gauge',
                    title: 'Risk Score',
                    data: { score: output.risk_analysis.probability },
                });
            }

            // Profile data as bar chart
            const profileData: Record<string, number> = {};
            if (output.profile.income) profileData['Income'] = output.profile.income;
            if (output.profile.credit_amount) profileData['Credit'] = output.profile.credit_amount;
            if (output.profile.annuity) profileData['Annuity'] = output.profile.annuity;

            if (Object.keys(profileData).length > 0) {
                charts.push({
                    type: 'bar',
                    title: `📊 Applicant ${output.applicant_id} Profile`,
                    data: profileData,
                });
            }

            // Top risk factors if available
            if (output.risk_analysis?.top_factors && Array.isArray(output.risk_analysis.top_factors)) {
                const factors: Record<string, number> = {};
                output.risk_analysis.top_factors.forEach(([name, value]: [string, number]) => {
                    factors[name] = value;
                });

                if (Object.keys(factors).length > 0) {
                    charts.push({
                        type: 'feature_importance',
                        title: '🔍 Top Risk Factors',
                        data: { features: factors },
                    });
                }
            }
        }
    }

    return charts;
}


// Suggestion cards for empty state
const SUGGESTIONS = [
    {
        icon: <TrendingUp size={20} />,
        title: 'Check Risk Score',
        description: '"What is the risk for applicant 100002?"',
        query: 'What is the risk score for applicant 100002?',
    },
    {
        icon: <Sparkles size={20} />,
        title: 'Risk Factors',
        description: '"What factors affect applicant 100002?"',
        query: "What factors affect applicant 100002's risk?",
    },
    {
        icon: <Users size={20} />,
        title: 'Compare Applicants',
        description: '"Compare applicant 100002 and 100003"',
        query: 'Compare income between applicant 100002 and 100003',
    },
    {
        icon: <HelpCircle size={20} />,
        title: 'Hypothetical Analysis',
        description: '"Risk for income 300k, credit 1M, age 35?"',
        query: 'What would be the risk for someone with income 300000, credit amount 1000000, age 35?',
    },
];

export default function Home() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [sessionId, setSessionId] = useState<string>(() => uuidv4());
    const [currentChatId, setCurrentChatId] = useState<string | null>(null);
    const [chatHistory, setChatHistory] = useState<ChatHistory[]>([]);
    const [analysisThreadId, setAnalysisThreadId] = useState<string | null>(null);
    const [showAnalysisPanel, setShowAnalysisPanel] = useState(false);

    const messagesEndRef = useRef<HTMLDivElement>(null);
    const chatContainerRef = useRef<HTMLDivElement>(null);

    // Auto-scroll to bottom when new messages arrive
    useEffect(() => {
        if (messagesEndRef.current) {
            messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [messages]);

    // Load chat history from localStorage
    useEffect(() => {
        const saved = localStorage.getItem('hc-chat-history');
        if (saved) {
            try {
                setChatHistory(JSON.parse(saved));
            } catch (e) {
                console.error('Failed to load chat history:', e);
            }
        }
    }, []);

    // Save chat history to localStorage
    useEffect(() => {
        if (chatHistory.length > 0) {
            localStorage.setItem('hc-chat-history', JSON.stringify(chatHistory.slice(0, 20)));
        }
    }, [chatHistory]);

    // Handle sending a message
    const handleSendMessage = async (content: string, file?: File, audio?: Blob) => {
        if (!content && !file && !audio) return;

        // Create user message
        const userMessage: Message = {
            role: 'user',
            content: file
                ? `[Uploaded: ${file.name}] ${content}`.trim()
                : audio
                    ? '[Voice message]'
                    : content,
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        };

        setMessages(prev => [...prev, userMessage]);
        setIsLoading(true);

        // Set current chat ID if not set
        if (!currentChatId) {
            setCurrentChatId(uuidv4());
        }

        try {
            if (file || audio) {
                // Multimodal request - use non-streaming API
                const question = audio
                    ? 'Transcribe and respond to this voice message'
                    : `Analyze this document: ${file?.name}. ${content}`;

                const response = await sendMultimodal(question, sessionId, audio, file);

                // Check if response includes analysis thread ID
                const threadId = (response as any).analysis_thread_id;
                if (threadId) {
                    setAnalysisThreadId(threadId);
                    setShowAnalysisPanel(true);
                }

                // Create assistant message
                const assistantMessage: Message = {
                    role: 'assistant',
                    content: response.answer,
                    timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                    charts: parseChartsFromToolOutputs(response.tool_outputs),
                    tool_outputs: response.tool_outputs,
                };

                setMessages(prev => [...prev, assistantMessage]);
            } else {
                // Text-only request - use STREAMING
                let currentMessage: Message | null = null;
                let allToolOutputs: any[] = [];
                let messageIndex = -1;

                await sendMessageStream(content, sessionId, undefined, (event: StreamEvent) => {
                    if (event.type === 'ai_message') {
                        if (event.has_tool_calls) {
                            // Message with tool calls - create new message and store tools
                            const newMessage: Message = {
                                role: 'assistant',
                                content: event.content || '',
                                timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                                tool_outputs: event.tool_calls,
                            };
                            currentMessage = newMessage;
                            setMessages(prev => {
                                messageIndex = prev.length;
                                return [...prev, newMessage];
                            });
                            allToolOutputs.push(...(event.tool_calls || []));
                        } else {
                            // Regular AI message (no tool calls)
                            if (currentMessage === null) {
                                // First message - create it
                                currentMessage = {
                                    role: 'assistant',
                                    content: event.content || '',
                                    timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                                };
                                setMessages(prev => {
                                    messageIndex = prev.length;
                                    return [...prev, currentMessage!];
                                });
                            } else {
                                // Continuation of previous message - append content
                                const newContent = event.content || '';
                                if (newContent.trim()) {
                                    setMessages(prev => {
                                        const updated = [...prev];
                                        if (messageIndex >= 0 && messageIndex < updated.length) {
                                            const existingContent = updated[messageIndex].content;
                                            // Smart joining: use space if content flows together, newline if it's a new paragraph
                                            const separator = newContent.startsWith('\n') || existingContent.endsWith('\n') ? '' : ' ';
                                            updated[messageIndex] = {
                                                ...updated[messageIndex],
                                                content: existingContent + separator + newContent,
                                            };
                                        }
                                        return updated;
                                    });
                                }
                            }
                        }
                    } else if (event.type === 'tool_result') {
                        // Tool result arrived
                        allToolOutputs.push(event.result);

                        // Update the last message to include this tool output
                        setMessages(prev => {
                            const updated = [...prev];
                            if (updated.length > 0 && updated[updated.length - 1].role === 'assistant') {
                                const lastMsg = updated[updated.length - 1];
                                updated[updated.length - 1] = {
                                    ...lastMsg,
                                    tool_outputs: [...(lastMsg.tool_outputs || []), event.result],
                                };
                            }
                            return updated;
                        });
                    } else if (event.type === 'analysis_step') {
                        // Analysis step completed - add step result to message
                        try {
                            if (event.step_result) {
                                // If no message exists yet, create one for the analysis results
                                if (currentMessage === null || messageIndex < 0) {
                                    currentMessage = {
                                        role: 'assistant',
                                        content: '',
                                        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                                        analysis_steps: [],
                                    };
                                    setMessages(prev => {
                                        messageIndex = prev.length;
                                        return [...prev, currentMessage!];
                                    });
                                }

                                const stepResult = event.step_result;
                                setMessages(prev => {
                                    const updated = [...prev];
                                    if (messageIndex < updated.length) {
                                        const existingSteps = updated[messageIndex].analysis_steps || [];
                                        updated[messageIndex] = {
                                            ...updated[messageIndex],
                                            analysis_steps: [...existingSteps, stepResult],
                                        };
                                    }
                                    return updated;
                                });
                            }
                        } catch (err) {
                            console.error('Error processing analysis step:', err, event.step_result);
                        }
                    } else if (event.type === 'analysis_started') {
                        // Analysis workflow started - show the analysis panel
                        if (event.thread_id) {
                            setAnalysisThreadId(event.thread_id);
                            setShowAnalysisPanel(true);
                        }
                    } else if (event.type === 'done') {
                        // Stream complete - generate charts from collected tool outputs
                        if (allToolOutputs.length > 0) {
                            const charts = parseChartsFromToolOutputs(allToolOutputs);
                            if (charts.length > 0) {
                                setMessages(prev => {
                                    const updated = [...prev];
                                    if (updated.length > 0 && updated[updated.length - 1].role === 'assistant') {
                                        updated[updated.length - 1] = {
                                            ...updated[updated.length - 1],
                                            charts,
                                        };
                                    }
                                    return updated;
                                });
                            }
                        }
                    } else if (event.type === 'error') {
                        console.error('Stream error:', event.error);
                        throw new Error(event.error || 'Stream error');
                    }
                });
            }

        } catch (error) {
            console.error('Chat error:', error);

            const errorMessage: Message = {
                role: 'assistant',
                content: `Sorry, I encountered an error: ${error instanceof Error ? error.message : 'Unknown error'}. Please try again.`,
                timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            };

            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setIsLoading(false);
        }
    };

    // Handle suggestion click
    const handleSuggestionClick = (query: string) => {
        handleSendMessage(query);
    };

    // Handle new chat
    const handleNewChat = () => {
        // Save current chat to history
        if (messages.length > 0) {
            const title = messages[0].content.slice(0, 40) + (messages[0].content.length > 40 ? '...' : '');
            const newHistory: ChatHistory = {
                id: currentChatId || uuidv4(),
                title,
                timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                messages: messages,
            };
            setChatHistory(prev => [newHistory, ...prev.filter(h => h.id !== newHistory.id)]);
        }

        // Reset state
        setMessages([]);
        setSessionId(uuidv4());
        setCurrentChatId(null);
    };

    // Handle select chat from history
    const handleSelectChat = (chat: ChatHistory) => {
        // Save current chat first
        if (messages.length > 0 && currentChatId) {
            const title = messages[0].content.slice(0, 40) + '...';
            setChatHistory(prev =>
                prev.map(h => h.id === currentChatId ? { ...h, messages, title } : h)
            );
        }

        // Load selected chat
        setMessages(chat.messages);
        setCurrentChatId(chat.id);
        setSessionId(uuidv4()); // New session for restored chat
    };

    return (
        <div className={styles.container}>
            {/* Sidebar */}
            <Sidebar
                chatHistory={chatHistory}
                currentChatId={currentChatId}
                onNewChat={handleNewChat}
                onSelectChat={handleSelectChat}
            />

            {/* Main Content */}
            <main className={styles.main}>
                {/* Chat Area */}
                <div className={styles.chatArea} ref={chatContainerRef}>
                    {messages.length === 0 ? (
                        /* Welcome Screen */
                        <div className={styles.welcome}>
                            <div className={styles.welcomeContent}>
                                <h1 className={styles.welcomeTitle}>
                                    How can I help you today?
                                </h1>
                                <p className={styles.welcomeDesc}>
                                    Ask me about credit risk assessment, applicant profiles, or financial analysis
                                </p>

                                {/* Suggestion Cards */}
                                <div className={styles.suggestions}>
                                    {SUGGESTIONS.map((suggestion, index) => (
                                        <button
                                            key={index}
                                            className={styles.suggestionCard}
                                            onClick={() => handleSuggestionClick(suggestion.query)}
                                        >
                                            <div className={styles.suggestionIcon}>{suggestion.icon}</div>
                                            <div className={styles.suggestionText}>
                                                <span className={styles.suggestionTitle}>{suggestion.title}</span>
                                                <span className={styles.suggestionDesc}>{suggestion.description}</span>
                                            </div>
                                        </button>
                                    ))}
                                </div>
                            </div>
                        </div>
                    ) : (
                        /* Messages */
                        <div className={styles.messages}>
                            {messages.map((message, index) => (
                                <ChatMessage key={index} message={message} index={index} />
                            ))}

                            {/* Loading indicator */}
                            {isLoading && (
                                <div className={styles.loadingContainer}>
                                    <div className={styles.loadingDots}>
                                        <span />
                                        <span />
                                        <span />
                                    </div>
                                    <span className={styles.loadingText}>Analyzing...</span>
                                </div>
                            )}

                            {/* Scroll anchor */}
                            <div ref={messagesEndRef} />
                        </div>
                    )}
                </div>

                {/* Analysis Workflow Panel */}
                {showAnalysisPanel && analysisThreadId && (
                    <div className={styles.analysisContainer}>
                        <AnalysisPanel
                            threadId={analysisThreadId}
                            onClose={() => {
                                setShowAnalysisPanel(false);
                                setAnalysisThreadId(null);
                            }}
                        />
                    </div>
                )}

                {/* Input Area */}
                <div className={styles.inputArea}>
                    {messages.length > 0 && (
                        <button className={styles.newChatBtn} onClick={handleNewChat}>
                            <RotateCcw size={16} />
                            <span>Start New Conversation</span>
                        </button>
                    )}
                    <ChatInput
                        onSendMessage={handleSendMessage}
                        isLoading={isLoading}
                    />
                </div>
            </main>
        </div>
    );
}
