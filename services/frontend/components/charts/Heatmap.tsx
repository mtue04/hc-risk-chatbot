'use client';

import React from 'react';
import styles from './Charts.module.css';

interface HeatmapProps {
    data: {
        values: number[][];
        xLabels: string[];
        yLabels: string[];
    };
    title?: string;
}

export default function Heatmap({ data, title }: HeatmapProps) {
    const { values, xLabels, yLabels } = data;

    // Find min and max for color scaling
    const allValues = values.flat();
    const minVal = Math.min(...allValues);
    const maxVal = Math.max(...allValues);
    const range = maxVal - minVal || 1;

    // Get color based on value
    const getColor = (value: number) => {
        const normalized = (value - minVal) / range;
        // Gradient from green (low) to yellow (mid) to red (high)
        if (normalized < 0.5) {
            const ratio = normalized * 2;
            return `rgb(${Math.round(16 + ratio * 229)}, ${Math.round(185 - ratio * 27)}, ${Math.round(129 - ratio * 118)})`;
        } else {
            const ratio = (normalized - 0.5) * 2;
            return `rgb(${Math.round(245 - ratio * 18)}, ${Math.round(158 - ratio * 128)}, ${Math.round(11 + ratio * 25)})`;
        }
    };

    return (
        <div className={styles.chartWrapper}>
            <div className={styles.chartTitle}>{title || '🗺️ Correlation Matrix'}</div>

            <div className={styles.heatmapContainer}>
                {/* Y-axis labels */}
                <div className={styles.heatmapYLabels}>
                    <div className={styles.heatmapCorner} />
                    {yLabels.map((label, i) => (
                        <div key={i} className={styles.heatmapYLabel}>
                            {label}
                        </div>
                    ))}
                </div>

                {/* Grid */}
                <div className={styles.heatmapGrid}>
                    {/* X-axis labels */}
                    <div className={styles.heatmapXLabels}>
                        {xLabels.map((label, i) => (
                            <div key={i} className={styles.heatmapXLabel}>
                                {label}
                            </div>
                        ))}
                    </div>

                    {/* Cells */}
                    <div className={styles.heatmapCells}>
                        {values.map((row, y) => (
                            <div key={y} className={styles.heatmapRow}>
                                {row.map((value, x) => (
                                    <div
                                        key={x}
                                        className={styles.heatmapCell}
                                        style={{ backgroundColor: getColor(value) }}
                                        title={`${yLabels[y]} × ${xLabels[x]}: ${value.toFixed(2)}`}
                                    >
                                        <span className={styles.heatmapCellValue}>
                                            {value.toFixed(2)}
                                        </span>
                                    </div>
                                ))}
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Color Legend */}
            <div className={styles.heatmapLegend}>
                <span>{minVal.toFixed(2)}</span>
                <div className={styles.heatmapGradient} />
                <span>{maxVal.toFixed(2)}</span>
            </div>
        </div>
    );
}
