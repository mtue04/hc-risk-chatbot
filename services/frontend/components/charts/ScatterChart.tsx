'use client';

import React from 'react';
import {
    ScatterChart as RechartsScatter,
    Scatter,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    ZAxis,
} from 'recharts';
import styles from './Charts.module.css';

interface ScatterChartProps {
    data: {
        x: number[];
        y: number[];
        labels?: string[];
        sizes?: number[];
    };
    title?: string;
}

export default function ScatterChart({ data, title }: ScatterChartProps) {
    // Defensive checks for data
    if (!data || !data.x || !data.y || !Array.isArray(data.x) || !Array.isArray(data.y)) {
        console.error('ScatterChart: Invalid data format', data);
        return (
            <div style={{ padding: '20px', color: '#ff6b6b', textAlign: 'center' }}>
                ⚠️ Cannot render scatter plot: Invalid data format
            </div>
        );
    }

    if (data.x.length === 0) {
        return (
            <div style={{ padding: '20px', color: '#ffa500', textAlign: 'center' }}>
                ⚠️ No data points to display
            </div>
        );
    }

    // Transform data for Recharts
    const chartData = data.x.map((x, i) => ({
        x,
        y: data.y[i],
        label: data.labels?.[i] || `Point ${i + 1}`,
        size: data.sizes?.[i] || 100,
    }));

    const CustomTooltip = ({ active, payload }: any) => {
        if (active && payload && payload.length) {
            const point = payload[0].payload;
            return (
                <div className={styles.tooltip}>
                    <p className={styles.tooltipLabel}>{point.label}</p>
                    <p>X: {point.x.toLocaleString()}</p>
                    <p>Y: {point.y.toLocaleString()}</p>
                </div>
            );
        }
        return null;
    };

    return (
        <div className={styles.chartWrapper} style={{ minHeight: '350px' }}>
            <div className={styles.chartTitle}>{title || '📈 Scatter Plot'}</div>

            <ResponsiveContainer width="100%" height={300}>
                <RechartsScatter margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                    <XAxis
                        type="number"
                        dataKey="x"
                        stroke="#707070"
                        tick={{ fill: '#b0b0b0', fontSize: 11 }}
                        axisLine={{ stroke: '#333' }}
                        name="X"
                    />
                    <YAxis
                        type="number"
                        dataKey="y"
                        stroke="#707070"
                        tick={{ fill: '#b0b0b0', fontSize: 11 }}
                        axisLine={{ stroke: '#333' }}
                        name="Y"
                    />
                    <ZAxis type="number" dataKey="size" range={[50, 400]} />
                    <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: '3 3' }} />
                    <Scatter
                        data={chartData}
                        fill="#E31E24"
                        fillOpacity={0.7}
                    />
                </RechartsScatter>
            </ResponsiveContainer>
        </div>
    );
}
