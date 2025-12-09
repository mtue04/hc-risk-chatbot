'use client';

import React from 'react';
import {
    PieChart as RechartsPieChart,
    Pie,
    Cell,
    Tooltip,
    ResponsiveContainer,
    Legend,
} from 'recharts';
import styles from './Charts.module.css';

interface PieChartProps {
    data: Record<string, number>; // { label: value }
    title?: string;
}

const COLORS = ['#E31E24', '#10B981', '#F59E0B', '#3B82F6', '#8B5CF6', '#EC4899'];

export default function PieChart({ data, title }: PieChartProps) {
    const chartData = Object.entries(data).map(([name, value]) => ({ name, value }));
    const total = chartData.reduce((sum, item) => sum + item.value, 0);

    const CustomTooltip = ({ active, payload }: any) => {
        if (active && payload && payload.length) {
            const item = payload[0];
            const percentage = ((item.value / total) * 100).toFixed(1);
            return (
                <div className={styles.tooltip}>
                    <p className={styles.tooltipLabel}>{item.name}</p>
                    <p style={{ color: item.payload.fill }}>
                        {item.value.toLocaleString()} ({percentage}%)
                    </p>
                </div>
            );
        }
        return null;
    };

    const RADIAN = Math.PI / 180;
    const renderCustomizedLabel = ({
        cx, cy, midAngle, innerRadius, outerRadius, percent, name,
    }: any) => {
        const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
        const x = cx + radius * Math.cos(-midAngle * RADIAN);
        const y = cy + radius * Math.sin(-midAngle * RADIAN);

        return percent > 0.05 ? (
            <text
                x={x}
                y={y}
                fill="white"
                textAnchor="middle"
                dominantBaseline="central"
                fontSize={11}
                fontWeight={500}
            >
                {`${(percent * 100).toFixed(0)}%`}
            </text>
        ) : null;
    };

    return (
        <div className={styles.chartWrapper}>
            <div className={styles.chartTitle}>{title || '📊 Distribution'}</div>

            <ResponsiveContainer width="100%" height={300}>
                <RechartsPieChart>
                    <Pie
                        data={chartData}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={renderCustomizedLabel}
                        innerRadius={50}
                        outerRadius={100}
                        dataKey="value"
                    >
                        {chartData.map((_, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                    </Pie>
                    <Tooltip content={<CustomTooltip />} />
                    <Legend
                        wrapperStyle={{ color: '#b0b0b0', fontSize: 12 }}
                        layout="vertical"
                        align="right"
                        verticalAlign="middle"
                    />
                </RechartsPieChart>
            </ResponsiveContainer>
        </div>
    );
}
