'use client';

import React from 'react';
import {
    BarChart as RechartsBar,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Cell,
    LabelList,
} from 'recharts';
import styles from './Charts.module.css';

interface FeatureImportanceProps {
    // Accept both formats:
    // New format: [{name: "...", importance: 0.xx}, ...]
    // Legacy format: {feature_name: value, ...}
    features: Array<{ name: string; importance: number }> | Record<string, number>;
    title?: string;
}

const CHART_COLORS = {
    positive: '#E31E24',
    negative: '#10B981',
};

export default function FeatureImportance({ features, title }: FeatureImportanceProps) {
    // Handle both array and object formats
    let data: Array<{ name: string; value: number; fill: string }>;

    if (Array.isArray(features)) {
        // New format: [{name, importance}, ...]
        data = features.map(f => ({
            name: f.name,
            value: f.importance,
            fill: f.importance > 0 ? CHART_COLORS.positive : CHART_COLORS.negative,
        }));
    } else {
        // Legacy format: {name: value, ...}
        data = Object.entries(features).map(([name, value]) => ({
            name,
            value,
            fill: value > 0 ? CHART_COLORS.positive : CHART_COLORS.negative,
        }));
    }

    // Sort by absolute value
    data.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

    const CustomTooltip = ({ active, payload }: any) => {
        if (active && payload && payload.length) {
            const item = payload[0].payload;
            return (
                <div className={styles.tooltip}>
                    <p className={styles.tooltipLabel}>{item.name}</p>
                    <p className={styles.tooltipValue} style={{ color: item.fill }}>
                        {item.value > 0 ? '+' : ''}{item.value.toFixed(3)}
                    </p>
                </div>
            );
        }
        return null;
    };

    // Calculate dynamic height based on number of features
    const dynamicHeight = Math.max(300, data.length * 35);

    return (
        <div className={styles.chartWrapper}>
            <div className={styles.chartTitle}>{title || '📊 Feature Importance'}</div>

            <ResponsiveContainer width="100%" height={dynamicHeight}>
                <RechartsBar
                    data={data}
                    layout="vertical"
                    margin={{ top: 10, right: 60, left: 120, bottom: 10 }}
                >
                    <CartesianGrid strokeDasharray="3 3" stroke="#333" horizontal={false} />
                    <XAxis
                        type="number"
                        stroke="#707070"
                        tick={{ fill: '#b0b0b0', fontSize: 11 }}
                        axisLine={{ stroke: '#333' }}
                    />
                    <YAxis
                        type="category"
                        dataKey="name"
                        stroke="#707070"
                        tick={{ fill: '#b0b0b0', fontSize: 11 }}
                        axisLine={{ stroke: '#333' }}
                        width={110}
                    />
                    <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.05)' }} />
                    <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                        {data.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.fill} />
                        ))}
                        <LabelList
                            dataKey="value"
                            position="right"
                            formatter={(val: number) => (val > 0 ? '+' : '') + val.toFixed(2)}
                            fill="#b0b0b0"
                            fontSize={11}
                        />
                    </Bar>
                </RechartsBar>
            </ResponsiveContainer>

            <div className={styles.legend}>
                <div className={styles.legendItem}>
                    <span className={styles.legendDot} style={{ background: CHART_COLORS.positive }} />
                    <span>Increases Risk</span>
                </div>
                <div className={styles.legendItem}>
                    <span className={styles.legendDot} style={{ background: CHART_COLORS.negative }} />
                    <span>Decreases Risk</span>
                </div>
            </div>
        </div>
    );
}
