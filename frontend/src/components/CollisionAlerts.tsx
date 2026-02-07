'use client';

import { motion } from 'framer-motion';
import { CollisionAlert } from '@/lib/api';

interface CollisionAlertsProps {
    alerts: CollisionAlert[];
}

const severityColors: Record<string, { bg: string; text: string; glow: string }> = {
    high: { bg: 'bg-red-500/20', text: 'text-red-400', glow: '#ff4444' },
    medium: { bg: 'bg-orange-500/20', text: 'text-orange-400', glow: '#ff8c00' },
    low: { bg: 'bg-green-500/20', text: 'text-green-400', glow: '#00ff88' },
};

export default function CollisionAlerts({ alerts }: CollisionAlertsProps) {
    return (
        <div className="rounded-2xl p-6 backdrop-blur-xl h-full"
            style={{
                background: 'linear-gradient(135deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.03) 100%)',
                border: '1px solid rgba(255,255,255,0.1)',
            }}
        >
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <span className="text-2xl">⚠️</span>
                Collision Alerts
            </h3>

            <div className="space-y-3 max-h-[400px] overflow-y-auto pr-2">
                {alerts.length === 0 ? (
                    <div className="text-gray-500 text-center py-8">
                        No collision alerts
                    </div>
                ) : (
                    alerts.map((alert, index) => {
                        const colors = severityColors[alert.severity] || severityColors.low;

                        return (
                            <motion.div
                                key={alert.id}
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: index * 0.1 }}
                                className={`p-4 rounded-xl ${colors.bg} border border-white/10`}
                                style={{ boxShadow: `0 0 20px ${colors.glow}15` }}
                            >
                                <div className="flex justify-between items-start mb-2">
                                    <span className={`text-xs font-semibold uppercase px-2 py-1 rounded ${colors.bg} ${colors.text}`}>
                                        {alert.severity}
                                    </span>
                                    <span className="text-xs text-gray-500">{alert.id}</span>
                                </div>

                                <div className="text-sm text-gray-300 mb-2">
                                    <span className="text-cyan-400">{alert.object1_id}</span>
                                    <span className="mx-2">↔</span>
                                    <span className="text-cyan-400">{alert.object2_id}</span>
                                </div>

                                <div className="flex justify-between text-xs text-gray-400">
                                    <span>Distance: <span className="text-white">{alert.distance_km} km</span></span>
                                    <span>Prob: <span className={colors.text}>{(alert.probability * 100).toFixed(2)}%</span></span>
                                </div>
                            </motion.div>
                        );
                    })
                )}
            </div>
        </div>
    );
}
