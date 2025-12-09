'use client';

import React from 'react';
import RiskGauge from './RiskGauge';
import FeatureImportance from './FeatureImportance';
import ComparisonRadar from './ComparisonRadar';
import TrendChart from './TrendChart';
import BarChart from './BarChart';
import PieChart from './PieChart';
import Histogram from './Histogram';
import ScatterChart from './ScatterChart';
import Heatmap from './Heatmap';
import WaterfallChart from './WaterfallChart';
import FunnelChart from './FunnelChart';
import styles from './ChartRenderer.module.css';

export interface ChartData {
    type:
    | 'risk_gauge'
    | 'feature_importance'
    | 'comparison_radar'
    | 'trend'
    | 'bar'
    | 'pie'
    | 'histogram'
    | 'scatter'
    | 'heatmap'
    | 'waterfall'
    | 'funnel'
    | 'multi_line'
    | 'grouped_bar'
    | 'stacked_bar'
    | 'bullet';
    title?: string;
    data: any;
    config?: any;
}

interface ChartRendererProps {
    chartData: ChartData;
}

export default function ChartRenderer({ chartData }: ChartRendererProps) {
    const { type, title, data, config } = chartData;

    const renderChart = () => {
        switch (type) {
            case 'risk_gauge':
                return <RiskGauge score={data.score} />;

            case 'feature_importance':
                // No limit on features - LLM can pass as many as needed
                return <FeatureImportance features={data.features} title={title} />;

            case 'comparison_radar':
                return <ComparisonRadar data={data} title={title} />;

            case 'trend':
            case 'multi_line':
                return <TrendChart data={data} title={title} config={config} />;

            case 'bar':
            case 'grouped_bar':
            case 'stacked_bar':
                return (
                    <BarChart
                        data={data}
                        title={title}
                        stacked={type === 'stacked_bar'}
                        grouped={type === 'grouped_bar'}
                    />
                );

            case 'pie':
                return <PieChart data={data} title={title} />;

            case 'histogram':
                return <Histogram data={data} title={title} bins={config?.bins} />;

            case 'scatter':
                return <ScatterChart data={data} title={title} />;

            case 'heatmap':
                return <Heatmap data={data} title={title} />;

            case 'waterfall':
                return <WaterfallChart data={data} title={title} />;

            case 'funnel':
                return <FunnelChart data={data} title={title} />;

            case 'bullet':
                // Bullet chart can reuse gauge with different styling
                return (
                    <RiskGauge
                        score={data.value}
                        target={data.target}
                        ranges={data.ranges}
                        isBullet
                    />
                );

            default:
                console.warn(`Unknown chart type: ${type}`);
                // Return null to hide unsupported types silently
                return null;
        }
    };

    return (
        <div className={styles.chartContainer}>
            {renderChart()}
        </div>
    );
}
