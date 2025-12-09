'use client';

import React from 'react';
import {
    BarChart as RechartsBarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Legend,
    Cell,
} from 'recharts';
import styles from './Charts.module.css';

interface BarChartProps {
    data: any; // Accept any format, validate at runtime
    title?: string;
    stacked?: boolean;
    grouped?: boolean;
    categories?: string[];
}

const COLORS = ['#E31E24', '#10B981', '#F59E0B', '#3B82F6', '#8B5CF6', '#EC4899'];

// Format large numbers: 1000 -> 1K, 1000000 -> 1M, etc.
const formatNumber = (value: number): string => {
    if (value === null || value === undefined) return '0';
    const absValue = Math.abs(value);
    if (absValue >= 1_000_000_000) return `${(value / 1_000_000_000).toFixed(1)}B`;
    if (absValue >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
    if (absValue >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
    return value.toFixed(value % 1 === 0 ? 0 : 2);
};

// Format for tooltip with full precision
const formatTooltipNumber = (value: number): string => {
    if (value === null || value === undefined) return '0';
    return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
};

export default function BarChart({ data, title, stacked, grouped, categories }: BarChartProps) {
    // Guard against empty or invalid data
    if (!data || typeof data !== 'object' || Object.keys(data).length === 0) {
        return (
            <div className={styles.chartWrapper}>
                <div className={styles.chartTitle}>{title || '📊 Bar Chart'}</div>
                <div className={styles.emptyChart}>No data available</div>
            </div>
        );
    }

    let chartData: Array<Record<string, any>> = [];
    let barKeys: string[] = [];
    let isGrouped = grouped || stacked;

    // Check data format with runtime type checks
    const hasCategories = data.categories && Array.isArray(data.categories);
    const hasValues = data.values && Array.isArray(data.values);
    const hasGroups = data.groups && Array.isArray(data.groups);

    if (hasCategories && hasValues && !hasGroups) {
        // New simple format: {categories: [...], values: [...]}
        barKeys = ['value'];
        chartData = data.categories.map((cat: string, i: number) => ({
            name: cat,
            value: data.values[i] || 0,
        }));
    } else if (hasCategories && hasGroups) {
        // Grouped format: {categories: [...], groups: [{name, values}, ...]}
        isGrouped = true;
        barKeys = data.groups.map((g: any) => g.name);
        chartData = data.categories.map((cat: string, i: number) => {
            const point: Record<string, any> = { name: cat };
            data.groups.forEach((group: any) => {
                point[group.name] = group.values[i] || 0;
            });
            return point;
        });
    } else {
        // Legacy format: {label: value} or {label: [values]}
        const entries = Object.entries(data).filter(([key]) => !['label', 'categories', 'values', 'groups'].includes(key));

        if (entries.length === 0) {
            return (
                <div className={styles.chartWrapper}>
                    <div className={styles.chartTitle}>{title || '📊 Bar Chart'}</div>
                    <div className={styles.emptyChart}>No valid data</div>
                </div>
            );
        }

        const firstValue = entries[0][1];
        isGrouped = isGrouped || Array.isArray(firstValue);

        if (isGrouped && Array.isArray(firstValue)) {
            // Legacy grouped: {group1: [v1, v2], group2: [v3, v4]}
            barKeys = entries.map(([key]) => key);
            const length = (firstValue as number[]).length;
            const cats = categories || Array.from({ length }, (_, i) => `Cat ${i + 1}`);

            chartData = cats.map((cat, i) => {
                const point: Record<string, any> = { name: cat };
                entries.forEach(([key, values]) => {
                    point[key] = Array.isArray(values) ? (values[i] || 0) : 0;
                });
                return point;
            });
        } else {
            // Legacy simple: {label1: value1, label2: value2}
            barKeys = ['value'];
            chartData = entries.map(([name, value]) => ({
                name,
                value: typeof value === 'number' ? value : 0,
            }));
        }
    }

    const CustomTooltip = ({ active, payload, label }: any) => {
        if (active && payload && payload.length) {
            return (
                <div className={styles.tooltip}>
                    <p className={styles.tooltipLabel}>{label}</p>
                    {payload.map((entry: any, index: number) => (
                        <p key={index} style={{ color: entry.fill || entry.color }}>
                            {entry.name}: {formatTooltipNumber(entry.value)}
                        </p>
                    ))}
                </div>
            );
        }
        return null;
    };

    return (
        <div className={styles.chartWrapper}>
            <div className={styles.chartTitle}>{title || '📊 Bar Chart'}</div>

            <ResponsiveContainer width="100%" height={280}>
                <RechartsBarChart data={chartData} margin={{ top: 10, right: 30, left: 20, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                    <XAxis
                        dataKey="name"
                        stroke="#707070"
                        tick={{ fill: '#b0b0b0', fontSize: 11 }}
                        axisLine={{ stroke: '#333' }}
                        interval={0}
                        angle={chartData.length > 5 ? -45 : 0}
                        textAnchor={chartData.length > 5 ? "end" : "middle"}
                        height={chartData.length > 5 ? 60 : 30}
                    />
                    <YAxis
                        stroke="#707070"
                        tick={{ fill: '#b0b0b0', fontSize: 11 }}
                        axisLine={{ stroke: '#333' }}
                        tickFormatter={formatNumber}
                        width={60}
                    />
                    <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.05)' }} />

                    {isGrouped && <Legend wrapperStyle={{ color: '#b0b0b0', fontSize: 12 }} />}

                    {barKeys.map((key, index) => (
                        <Bar
                            key={key}
                            dataKey={key}
                            fill={COLORS[index % COLORS.length]}
                            radius={[4, 4, 0, 0]}
                            stackId={stacked ? 'stack' : undefined}
                        >
                            {!isGrouped && chartData.map((_, i) => (
                                <Cell key={`cell-${i}`} fill={COLORS[i % COLORS.length]} />
                            ))}
                        </Bar>
                    ))}
                </RechartsBarChart>
            </ResponsiveContainer>
        </div>
    );
}

