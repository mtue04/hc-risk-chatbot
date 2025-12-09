'use client';

import React, { useMemo } from 'react';
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
} from 'recharts';
import styles from './Charts.module.css';

interface HistogramProps {
    // Accept multiple formats:
    // New format: {bins: [0, 10, 20, ...], counts: [5, 12, 23, ...], label?: "..."}
    // Raw values format: number[] (will compute bins client-side)
    // Backend binned format: {histogram_bins: [{bin, range_start, range_end, count, label}, ...]}
    data: number[] | {
        bins?: number[];
        counts?: number[];
        histogram_bins?: Array<{ bin: number; range_start: number; range_end: number; count: number; label?: string }>;
        label?: string;
    };
    title?: string;
    bins?: number;
}

export default function Histogram({ data, title, bins = 20 }: HistogramProps) {
    // Calculate histogram bins
    const histogramData = useMemo(() => {
        if (!data) return [];

        // Check if it's pre-computed bins/counts format
        if (typeof data === 'object' && !Array.isArray(data)) {
            if (data.histogram_bins && Array.isArray(data.histogram_bins)) {
                // Backend format: {histogram_bins: [{bin, count, label, ...}, ...]}
                return data.histogram_bins.map(b => ({
                    range: b.label || `${b.range_start.toFixed(0)}`,
                    count: b.count,
                    min: b.range_start,
                    max: b.range_end,
                }));
            }

            if (data.bins && data.counts && Array.isArray(data.bins) && Array.isArray(data.counts)) {
                // New format: {bins: [0, 10, 20, ...], counts: [5, 12, ...]}
                return data.counts.map((count, i) => ({
                    range: `${data.bins![i]}`,
                    count,
                    min: data.bins![i],
                    max: data.bins![i + 1] || data.bins![i] + 10,
                }));
            }

            return [];
        }

        // Raw values format: compute bins client-side
        if (!Array.isArray(data) || data.length === 0) return [];

        const values = data as number[];
        const min = Math.min(...values);
        const max = Math.max(...values);
        const binWidth = (max - min) / bins;

        // Initialize bins
        const histogram = Array.from({ length: bins }, (_, i) => ({
            range: `${(min + i * binWidth).toFixed(1)}`,
            count: 0,
            min: min + i * binWidth,
            max: min + (i + 1) * binWidth,
        }));

        // Count values in each bin
        values.forEach(value => {
            const binIndex = Math.min(Math.floor((value - min) / binWidth), bins - 1);
            if (binIndex >= 0 && binIndex < bins) {
                histogram[binIndex].count++;
            }
        });

        return histogram;
    }, [data, bins]);

    // Get label from data if available
    const dataLabel = typeof data === 'object' && !Array.isArray(data) ? data.label : undefined;
    const totalCount = Array.isArray(data) ? data.length : histogramData.reduce((sum, b) => sum + b.count, 0);

    const CustomTooltip = ({ active, payload }: any) => {
        if (active && payload && payload.length) {
            const item = payload[0].payload;
            return (
                <div className={styles.tooltip}>
                    <p className={styles.tooltipLabel}>
                        Range: {item.min.toFixed(2)} - {item.max.toFixed(2)}
                    </p>
                    <p style={{ color: '#E31E24' }}>Count: {item.count}</p>
                </div>
            );
        }
        return null;
    };

    return (
        <div className={styles.chartWrapper}>
            <div className={styles.chartTitle}>{title || '📊 Distribution'}</div>

            <ResponsiveContainer width="100%" height={280}>
                <BarChart data={histogramData} margin={{ top: 10, right: 30, left: 10, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                    <XAxis
                        dataKey="range"
                        stroke="#707070"
                        tick={{ fill: '#b0b0b0', fontSize: 10 }}
                        axisLine={{ stroke: '#333' }}
                        interval="preserveStartEnd"
                    />
                    <YAxis
                        stroke="#707070"
                        tick={{ fill: '#b0b0b0', fontSize: 11 }}
                        axisLine={{ stroke: '#333' }}
                    />
                    <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.05)' }} />
                    <Bar dataKey="count" fill="#E31E24" radius={[4, 4, 0, 0]} opacity={0.85} />
                </BarChart>
            </ResponsiveContainer>

            <div className={styles.chartInfo}>
                Total: {totalCount} values | Bins: {histogramData.length}
            </div>
        </div>
    );
}
