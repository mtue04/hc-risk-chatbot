'use client';

import React from 'react';
import { PieChart as RechartsPie, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import styles from './Charts.module.css';

interface RiskGaugeProps {
    score: number; // 0-1 or 0-100
    target?: number;
    ranges?: number[];
    isBullet?: boolean;
}

const RISK_COLORS = {
    low: '#10B981',
    medium: '#F59E0B',
    high: '#E31E24',
};

export default function RiskGauge({ score, target, ranges, isBullet }: RiskGaugeProps) {
    // Normalize score to 0-100
    const normalizedScore = score > 1 ? score : score * 100;

    // Determine risk level and color
    const getRiskLevel = (s: number) => {
        if (s < 30) return { label: 'Low Risk', color: RISK_COLORS.low };
        if (s < 60) return { label: 'Medium Risk', color: RISK_COLORS.medium };
        return { label: 'High Risk', color: RISK_COLORS.high };
    };

    const risk = getRiskLevel(normalizedScore);

    // Data for the gauge
    const gaugeData = [
        { name: 'Score', value: normalizedScore },
        { name: 'Remaining', value: 100 - normalizedScore },
    ];

    // Background segments
    const backgroundData = [
        { name: 'Low', value: 30, color: 'rgba(16, 185, 129, 0.2)' },
        { name: 'Medium', value: 30, color: 'rgba(245, 158, 11, 0.2)' },
        { name: 'High', value: 40, color: 'rgba(227, 30, 36, 0.2)' },
    ];

    return (
        <div className={styles.gaugeContainer}>
            <div className={styles.chartTitle}>📊 Risk Score</div>

            <div className={styles.gaugeWrapper}>
                <ResponsiveContainer width="100%" height={180}>
                    <RechartsPie data={gaugeData} startAngle={180} endAngle={0} cy="80%">
                        {/* Background Arc */}
                        <Pie
                            data={backgroundData}
                            dataKey="value"
                            startAngle={180}
                            endAngle={0}
                            innerRadius={60}
                            outerRadius={80}
                            cy="80%"
                            stroke="none"
                        >
                            {backgroundData.map((entry, index) => (
                                <Cell key={`bg-${index}`} fill={entry.color} />
                            ))}
                        </Pie>

                        {/* Score Arc */}
                        <Pie
                            data={gaugeData}
                            dataKey="value"
                            startAngle={180}
                            endAngle={180 - (normalizedScore / 100) * 180}
                            innerRadius={60}
                            outerRadius={80}
                            cy="80%"
                            stroke="none"
                        >
                            <Cell fill={risk.color} />
                            <Cell fill="transparent" />
                        </Pie>
                    </RechartsPie>
                </ResponsiveContainer>

                {/* Center Text */}
                <div className={styles.gaugeCenter}>
                    <span className={styles.gaugeValue} style={{ color: risk.color }}>
                        {normalizedScore.toFixed(1)}%
                    </span>
                    <span className={styles.gaugeLabel} style={{ color: risk.color }}>
                        {risk.label}
                    </span>
                </div>
            </div>

            {target && (
                <div className={styles.targetInfo}>
                    Target: {target}%
                </div>
            )}
        </div>
    );
}
