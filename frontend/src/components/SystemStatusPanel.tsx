'use client';

import { motion } from 'framer-motion';
import { AiCacheStatus, DataStatus, HealthStatus } from '@/lib/api';

interface SystemStatusPanelProps {
    health: HealthStatus | null;
    dataStatus: DataStatus | null;
    cacheStatus: AiCacheStatus | null;
    onStartAiCache: () => void;
    startingAiCache: boolean;
}

function formatPercent(v: number | undefined): string {
    if (v === undefined || Number.isNaN(v)) return 'N/A';
    return `${(v * 100).toFixed(1)}%`;
}

export default function SystemStatusPanel({ health, dataStatus, cacheStatus, onStartAiCache, startingAiCache }: SystemStatusPanelProps) {
    const ok = health?.ok ?? false;
    const dbOk = health?.db_ok ?? false;
    const cimLoaded = health?.cim_loaded ?? false;
    const cached = cacheStatus?.metrics?.cached_objects ?? 0;
    const total = cacheStatus?.metrics?.total_objects ?? 0;
    const coverage = cacheStatus?.metrics?.cache_coverage;

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-2xl p-4 backdrop-blur-xl"
            style={{
                background: 'linear-gradient(135deg, rgba(0,0,0,0.7) 0%, rgba(30,30,60,0.35) 100%)',
                border: '1px solid rgba(255,255,255,0.14)',
            }}
        >
            <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold text-white">System Status</h3>
                <div className={`text-xs font-semibold ${ok ? 'text-green-400' : 'text-red-400'}`}>{ok ? 'ONLINE' : 'OFFLINE'}</div>
            </div>

            <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="p-2 rounded-lg bg-black/30 border border-white/10">
                    <div className="text-gray-400">DB</div>
                    <div className={`font-semibold ${dbOk ? 'text-green-400' : 'text-red-400'}`}>{dbOk ? 'OK' : 'ERROR'}</div>
                </div>
                <div className="p-2 rounded-lg bg-black/30 border border-white/10">
                    <div className="text-gray-400">CIM</div>
                    <div className={`font-semibold ${cimLoaded ? 'text-green-400' : 'text-yellow-400'}`}>{cimLoaded ? 'LOADED' : 'NOT LOADED'}</div>
                </div>
                <div className="p-2 rounded-lg bg-black/30 border border-white/10">
                    <div className="text-gray-400">Data Fresh</div>
                    <div className={`font-semibold ${dataStatus?.is_fresh ? 'text-green-400' : 'text-yellow-400'}`}>{dataStatus ? (dataStatus.is_fresh ? 'YES' : 'NO') : 'N/A'}</div>
                </div>
                <div className="p-2 rounded-lg bg-black/30 border border-white/10">
                    <div className="text-gray-400">Objects</div>
                    <div className="font-semibold text-cyan-300">{dataStatus ? dataStatus.objects_count.toLocaleString() : 'N/A'}</div>
                </div>
            </div>

            <div className="mt-3 p-2 rounded-lg bg-black/30 border border-white/10">
                <div className="flex items-center justify-between">
                    <div className="text-xs text-gray-400">AI Cache</div>
                    <div className="text-xs font-semibold text-purple-300">{formatPercent(coverage)}</div>
                </div>
                <div className="mt-1 text-[11px] text-gray-400">
                    {total ? `${cached.toLocaleString()}/${total.toLocaleString()} cached` : 'No metrics'}
                    {cacheStatus?.running ? ' • running' : ''}
                </div>
                <button
                    onClick={onStartAiCache}
                    disabled={startingAiCache || cacheStatus?.running}
                    className="mt-2 w-full px-3 py-2 rounded-lg bg-purple-500/20 border border-purple-400/30 text-purple-200 text-xs font-semibold hover:bg-purple-500/30 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    {cacheStatus?.running ? 'Caching…' : startingAiCache ? 'Starting…' : 'Cache AI Now'}
                </button>
            </div>

            <div className="mt-2 text-[10px] text-gray-500">
                Last update: {dataStatus?.last_update ?? 'Unknown'}
            </div>
        </motion.div>
    );
}

