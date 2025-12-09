'use client';

import React from 'react';
import {
    FunnelChart as RechartsFunnel,
    Funnel,
    Cell,
    Tooltip,
    ResponsiveContainer,
    LabelList,
} from 'recharts';
import styles from './Charts.module.css';

interface FunnelChartProps {
    data: Record<string, number>; // { stage: value }
    title?: string;
}

const COLORS = ['#E31E24', '#F59E0B', '#10B981', '#3B82F6', '#8B5CF6'];

export default function FunnelChart({ data, title }: FunnelChartProps) {
    const entries = Object.entries(data);
    const firstValue = entries[0]?.[1] || 1;

    const chartData = entries.map(([name, value], index) => ({
        name,
        value,
        percentage: ((value / firstValue) * 100).toFixed(1),
        fill: COLORS[index % COLORS.length],
    }));

    const CustomTooltip = ({ active, payload }: any) => {
        if (active && payload && payload.length) {
            const item = payload[0].payload;
            return (
                <div className={styles.tooltip}>
                    <p className={styles.tooltipLabel}>{item.name}</p>
                    <p style={{ color: item.fill }}>
                        {item.value.toLocaleString()} ({item.percentage}%)
                    </p>
                </div>
            );
        }
        return null;
    };

    return (
        <div className={styles.chartWrapper}>
            <div className={styles.chartTitle}>{title || '📊 Funnel Analysis'}</div>

            <ResponsiveContainer width="100%" height={300}>
                <RechartsFunnel margin={{ top: 20, right: 80, left: 80, bottom: 20 }}>
                    <Tooltip content={<CustomTooltip />} />
                    <Funnel
                        data={chartData}
                        dataKey="value"
                        isAnimationActive
                    >
                        {chartData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.fill} />
                        ))}
                        <LabelList
                            position="right"
                            dataKey="name"
                            fill="#b0b0b0"
                            fontSize={12}
                        />
                        <LabelList
                            position="left"
                            dataKey="percentage"
                            fill="#ffffff"
                            fontSize={11}
                            formatter={(val: string) => `${val}%`}
                        />
                    </Funnel>
                </RechartsFunnel>
            </ResponsiveContainer>
        </div>
    );
}
