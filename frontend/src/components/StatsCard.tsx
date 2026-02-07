'use client';

import { motion } from 'framer-motion';
import { ReactNode } from 'react';

interface StatsCardProps {
    title: string;
    value: string | number;
    icon: ReactNode;
    color?: string;
    subtitle?: string;
}

export default function StatsCard({ title, value, icon, color = '#00BFFF', subtitle }: StatsCardProps) {
    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="relative overflow-hidden rounded-2xl p-6 backdrop-blur-xl"
            style={{
                background: 'linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%)',
                border: '1px solid rgba(255,255,255,0.1)',
                boxShadow: `0 0 30px ${color}20`,
            }}
        >
            {/* Glow effect */}
            <div
                className="absolute -top-10 -right-10 w-32 h-32 rounded-full blur-3xl opacity-20"
                style={{ background: color }}
            />

            <div className="relative z-10">
                <div className="flex items-center justify-between mb-3">
                    <span className="text-gray-400 text-sm font-medium">{title}</span>
                    <div className="text-2xl" style={{ color }}>
                        {icon}
                    </div>
                </div>

                <div className="text-3xl font-bold text-white mb-1">
                    {value}
                </div>

                {subtitle && (
                    <span className="text-xs text-gray-500">{subtitle}</span>
                )}
            </div>
        </motion.div>
    );
}
