'use client';

import { motion } from 'framer-motion';
import { ModelInfo } from '@/lib/api';

interface ModelInfoPanelProps {
    modelInfo: ModelInfo | null;
}

export default function ModelInfoPanel({ modelInfo }: ModelInfoPanelProps) {
    if (!modelInfo) return null;

    const hasAccuracy = modelInfo.accuracy !== undefined && modelInfo.accuracy !== null;
    const hasF1 = modelInfo.f1_score !== undefined && modelInfo.f1_score !== null;

    const metrics = [
        { label: 'Accuracy', value: hasAccuracy ? `${((modelInfo.accuracy as number) * 100).toFixed(2)}%` : 'N/A', color: '#00ff88' },
        { label: 'F1 Score', value: hasF1 ? `${((modelInfo.f1_score as number) * 100).toFixed(2)}%` : 'N/A', color: '#00BFFF' },
        { label: 'Parameters', value: `${(modelInfo.parameters / 1e6).toFixed(2)}M`, color: '#ff8c00' },
        { label: 'Inference', value: `${modelInfo.inference_speed_ms}ms`, color: '#ff4444' },
    ];

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-2xl p-6 backdrop-blur-xl"
            style={{
                background: 'linear-gradient(135deg, rgba(0,191,255,0.1) 0%, rgba(0,100,150,0.05) 100%)',
                border: '1px solid rgba(0,191,255,0.2)',
                boxShadow: '0 0 40px rgba(0,191,255,0.1)',
            }}
        >
            <h3 className="text-lg font-semibold text-white mb-2 flex items-center gap-2">
                <span className="text-2xl">🧠</span>
                {modelInfo.name}
            </h3>
            <p className="text-xs text-cyan-400 mb-1">v{modelInfo.version}</p>
            <p className="text-[11px] text-gray-400 mb-4">
                {modelInfo.evaluated ? `Evaluated (${modelInfo.evaluation_label ?? 'unknown label'})` : 'Not evaluated'}
                {modelInfo.evaluation_source ? ` • ${modelInfo.evaluation_source}` : ''}
            </p>

            <div className="grid grid-cols-2 gap-3">
                {metrics.map((metric, index) => (
                    <motion.div
                        key={metric.label}
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: index * 0.1 }}
                        className="p-3 rounded-xl bg-black/30 border border-white/10"
                    >
                        <div className="text-xs text-gray-400 mb-1">{metric.label}</div>
                        <div className="text-lg font-bold" style={{ color: metric.color }}>
                            {metric.value}
                        </div>
                    </motion.div>
                ))}
            </div>
        </motion.div>
    );
}
