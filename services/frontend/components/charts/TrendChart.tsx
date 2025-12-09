'use client';

import React from 'react';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Legend,
    Area,
} from 'recharts';
import styles from './Charts.module.css';

interface TrendChartProps {
    data: number[] | Record<string, number[]>; // Single series or multiple series
    title?: string;
    config?: {
        labels?: string[];
        showArea?: boolean;
    };
}

const COLORS = ['#E31E24', '#10B981', '#F59E0B', '#3B82F6', '#8B5CF6'];

export default function TrendChart({ data, title, config }: TrendChartProps) {
    const isMultiSeries = typeof data === 'object' && !Array.isArray(data);

    // Transform data for Recharts
    let chartData: any[];
    let seriesKeys: string[];

    if (isMultiSeries) {
        const multiData = data as Record<string, number[]>;
        seriesKeys = Object.keys(multiData);
        const length = multiData[seriesKeys[0]]?.length || 0;

        chartData = Array.from({ length }, (_, i) => {
            const point: Record<string, any> = {
                index: config?.labels?.[i] ?? i + 1,
            };
            seriesKeys.forEach(key => {
                point[key] = multiData[key][i];
            });
            return point;
        });
    } else {
        const singleData = data as number[];
        seriesKeys = ['value'];
        chartData = singleData.map((value, i) => ({
            index: config?.labels?.[i] ?? i + 1,
            value,
        }));
    }

    const CustomTooltip = ({ active, payload, label }: any) => {
        if (active && payload && payload.length) {
            return (
                <div className={styles.tooltip}>
                    <p className={styles.tooltipLabel}>{label}</p>
                    {payload.map((entry: any, index: number) => (
                        <p key={index} style={{ color: entry.color }}>
                            {entry.name}: {entry.value}
                        </p>
                    ))}
                </div>
            );
        }
        return null;
    };

    return (
        <div className={styles.chartWrapper}>
            <div className={styles.chartTitle}>{title || '📈 Trend Analysis'}</div>

            <ResponsiveContainer width="100%" height={280}>
                <LineChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                    <XAxis
                        dataKey="index"
                        stroke="#707070"
                        tick={{ fill: '#b0b0b0', fontSize: 11 }}
                        axisLine={{ stroke: '#333' }}
                    />
                    <YAxis
                        stroke="#707070"
                        tick={{ fill: '#b0b0b0', fontSize: 11 }}
                        axisLine={{ stroke: '#333' }}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    {isMultiSeries && (
                        <Legend wrapperStyle={{ color: '#b0b0b0', fontSize: 12 }} />
                    )}
                    {seriesKeys.map((key, index) => (
                        <Line
                            key={key}
                            type="monotone"
                            dataKey={key}
                            stroke={COLORS[index % COLORS.length]}
                            strokeWidth={2}
                            dot={{ fill: COLORS[index % COLORS.length], r: 4 }}
                            activeDot={{ r: 6 }}
                        />
                    ))}
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
}
