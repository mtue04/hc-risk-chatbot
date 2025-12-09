'use client';

import React from 'react';
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Cell,
    ReferenceLine,
} from 'recharts';
import styles from './Charts.module.css';

interface WaterfallChartProps {
    data: Record<string, number>; // { label: value } where positive/negative indicates direction
    title?: string;
}

export default function WaterfallChart({ data, title }: WaterfallChartProps) {
    const entries = Object.entries(data);

    // Calculate cumulative values for waterfall effect
    let cumulative = 0;
    const chartData = entries.map(([name, value], index) => {
        const isFirst = index === 0;
        const isLast = index === entries.length - 1;

        let start = cumulative;
        let end = cumulative + value;

        if (isFirst) {
            // First bar starts from 0
            start = 0;
            end = value;
        }

        if (isLast) {
            // Last bar shows the total
            start = 0;
            end = cumulative + value;
        }

        const result = {
            name,
            value: Math.abs(value),
            start: Math.min(start, end),
            end: Math.max(start, end),
            isPositive: value >= 0,
            isFirst,
            isLast,
            cumulative: end,
        };

        cumulative += value;
        return result;
    });

    const getColor = (item: any) => {
        if (item.isFirst || item.isLast) return '#3B82F6'; // Blue for start/end
        return item.isPositive ? '#E31E24' : '#10B981'; // Red for increase, Green for decrease
    };

    const CustomTooltip = ({ active, payload }: any) => {
        if (active && payload && payload.length) {
            const item = payload[0].payload;
            return (
                <div className={styles.tooltip}>
                    <p className={styles.tooltipLabel}>{item.name}</p>
                    <p style={{ color: getColor(item) }}>
                        {item.isPositive ? '+' : '-'}{item.value.toLocaleString()}
                    </p>
                    <p className={styles.tooltipSecondary}>
                        Cumulative: {item.cumulative.toLocaleString()}
                    </p>
                </div>
            );
        }
        return null;
    };

    return (
        <div className={styles.chartWrapper}>
            <div className={styles.chartTitle}>{title || '📊 Waterfall Analysis'}</div>

            <ResponsiveContainer width="100%" height={300}>
                <BarChart data={chartData} margin={{ top: 20, right: 30, left: 10, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                    <XAxis
                        dataKey="name"
                        stroke="#707070"
                        tick={{ fill: '#b0b0b0', fontSize: 11 }}
                        axisLine={{ stroke: '#333' }}
                    />
                    <YAxis
                        stroke="#707070"
                        tick={{ fill: '#b0b0b0', fontSize: 11 }}
                        axisLine={{ stroke: '#333' }}
                    />
                    <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.05)' }} />
                    <ReferenceLine y={0} stroke="#555" />

                    {/* Invisible spacer bar */}
                    <Bar dataKey="start" stackId="stack" fill="transparent" />

                    {/* Visible value bar */}
                    <Bar dataKey="value" stackId="stack" radius={[4, 4, 0, 0]}>
                        {chartData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={getColor(entry)} />
                        ))}
                    </Bar>
                </BarChart>
            </ResponsiveContainer>

            <div className={styles.legend}>
                <div className={styles.legendItem}>
                    <span className={styles.legendDot} style={{ background: '#3B82F6' }} />
                    <span>Base/Total</span>
                </div>
                <div className={styles.legendItem}>
                    <span className={styles.legendDot} style={{ background: '#E31E24' }} />
                    <span>Increase</span>
                </div>
                <div className={styles.legendItem}>
                    <span className={styles.legendDot} style={{ background: '#10B981' }} />
                    <span>Decrease</span>
                </div>
            </div>
        </div>
    );
}
