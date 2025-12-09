'use client';

import React from 'react';
import {
    RadarChart,
    PolarGrid,
    PolarAngleAxis,
    PolarRadiusAxis,
    Radar,
    ResponsiveContainer,
    Legend,
    Tooltip,
} from 'recharts';
import styles from './Charts.module.css';

interface ComparisonRadarProps {
    data: Record<string, Record<string, number>>; // { applicantId: { metric: value } }
    title?: string;
}

const COLORS = ['#E31E24', '#10B981', '#F59E0B', '#3B82F6', '#8B5CF6'];

export default function ComparisonRadar({ data, title }: ComparisonRadarProps) {
    const applicants = Object.keys(data);
    const metrics = Object.keys(data[applicants[0]] || {});

    // Transform data for Recharts
    const chartData = metrics.map(metric => {
        const point: Record<string, any> = { metric };
        applicants.forEach(applicant => {
            point[applicant] = data[applicant][metric];
        });
        return point;
    });

    const CustomTooltip = ({ active, payload }: any) => {
        if (active && payload && payload.length) {
            return (
                <div className={styles.tooltip}>
                    <p className={styles.tooltipLabel}>{payload[0]?.payload?.metric}</p>
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
            <div className={styles.chartTitle}>{title || '📈 Applicant Comparison'}</div>

            <ResponsiveContainer width="100%" height={350}>
                <RadarChart data={chartData}>
                    <PolarGrid stroke="#333" />
                    <PolarAngleAxis dataKey="metric" tick={{ fill: '#b0b0b0', fontSize: 11 }} />
                    <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: '#707070', fontSize: 10 }} />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend
                        wrapperStyle={{ color: '#b0b0b0', fontSize: 12 }}
                        formatter={(value) => `Applicant ${value}`}
                    />
                    {applicants.map((applicant, index) => (
                        <Radar
                            key={applicant}
                            name={applicant}
                            dataKey={applicant}
                            stroke={COLORS[index % COLORS.length]}
                            fill={COLORS[index % COLORS.length]}
                            fillOpacity={0.2}
                            strokeWidth={2}
                        />
                    ))}
                </RadarChart>
            </ResponsiveContainer>
        </div>
    );
}
