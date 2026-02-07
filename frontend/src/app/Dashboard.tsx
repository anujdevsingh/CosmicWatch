'use client';

import { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import { motion, AnimatePresence } from 'framer-motion';
import { fetchDebris, fetchStats, fetchCollisions, fetchModelInfo, refreshData, fetchHealthz, getDataStatus, getAiCacheStatus, startAiCache, DebrisObject, DashboardStats, CollisionAlert, ModelInfo, HealthStatus, DataStatus, AiCacheStatus } from '@/lib/api';
import CollisionAlerts from '@/components/CollisionAlerts';
import ModelInfoPanel from '@/components/ModelInfoPanel';
import PcCalculator from '@/components/PcCalculator';
import SystemStatusPanel from '@/components/SystemStatusPanel';
import CdmModelPanel from '@/components/CdmModelPanel';

// Dynamic import for Earth3D (requires client-side only)
const Earth3D = dynamic(() => import('@/components/Earth3D'), {
    ssr: false,
    loading: () => (
        <div className="fixed inset-0 bg-black flex items-center justify-center">
            <div className="text-cyan-400 animate-pulse text-xl">Loading 3D Globe...</div>
        </div>
    ),
});

export default function Dashboard() {
    const [debris, setDebris] = useState<DebrisObject[]>([]);
    const [stats, setStats] = useState<DashboardStats | null>(null);
    const [collisions, setCollisions] = useState<CollisionAlert[]>([]);
    const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
    const [health, setHealth] = useState<HealthStatus | null>(null);
    const [dataStatus, setDataStatus] = useState<DataStatus | null>(null);
    const [cacheStatus, setCacheStatus] = useState<AiCacheStatus | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [showPanels, setShowPanels] = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const [refreshMessage, setRefreshMessage] = useState<string | null>(null);
    const [startingAiCache, setStartingAiCache] = useState(false);

    // Handle data refresh from Celestrak
    async function handleRefreshData() {
        setRefreshing(true);
        setRefreshMessage('🔄 Fetching fresh data from Celestrak...');

        try {
            const result = await refreshData();

            if (result.success) {
                setRefreshMessage(`✅ ${result.message} (${result.objects_updated} objects)`);
                const [debrisData, statsData, collisionData, modelData, healthData, dataStatusData, cacheStatusData] = await Promise.all([
                    fetchDebris(500),
                    fetchStats(),
                    fetchCollisions(10),
                    fetchModelInfo(),
                    fetchHealthz(),
                    getDataStatus(),
                    getAiCacheStatus(),
                ]);
                setDebris(debrisData);
                setStats(statsData);
                setCollisions(collisionData);
                setModelInfo(modelData);
                setHealth(healthData);
                setDataStatus(dataStatusData);
                setCacheStatus(cacheStatusData);
            } else {
                setRefreshMessage(`⚠️ ${result.message}`);
            }
        } catch {
            setRefreshMessage('❌ Failed to refresh data');
        }

        setRefreshing(false);
        // Clear message after 5 seconds
        setTimeout(() => setRefreshMessage(null), 5000);
    }

    useEffect(() => {
        let active = true;
        let inFlightFast = false;
        let inFlightDebris = false;

        async function loadFast() {
            if (inFlightFast) return;
            inFlightFast = true;
            try {
                const results = await Promise.allSettled([
                    fetchStats(),
                    fetchCollisions(10),
                    fetchModelInfo(),
                    fetchHealthz(),
                    getDataStatus(),
                    getAiCacheStatus(),
                ]);

                if (!active) return;

                const statsResult = results[0];
                const collisionsResult = results[1];
                const modelResult = results[2];
                const healthResult = results[3];
                const dataStatusResult = results[4];
                const cacheStatusResult = results[5];

                if (statsResult.status === 'fulfilled') setStats(statsResult.value);
                if (collisionsResult.status === 'fulfilled') setCollisions(collisionsResult.value);
                if (modelResult.status === 'fulfilled') setModelInfo(modelResult.value);
                if (healthResult.status === 'fulfilled') setHealth(healthResult.value);
                if (dataStatusResult.status === 'fulfilled') setDataStatus(dataStatusResult.value);
                if (cacheStatusResult.status === 'fulfilled') setCacheStatus(cacheStatusResult.value);

                const anyRejected = results.some(r => r.status === 'rejected');
                setError(anyRejected ? 'Some data failed to load. Check that the API server is running.' : null);
            } catch {
                if (!active) return;
                setError('Failed to load data. Make sure the API server is running on port 8000.');
            } finally {
                inFlightFast = false;
            }
        }

        async function loadDebris() {
            if (inFlightDebris) return;
            inFlightDebris = true;
            try {
                const debrisData = await fetchDebris(500);
                if (!active) return;
                setDebris(debrisData);
            } catch {
                if (!active) return;
                setError('Failed to load data. Make sure the API server is running on port 8000.');
            } finally {
                inFlightDebris = false;
            }
        }

        loadFast();
        loadDebris();

        let interval: NodeJS.Timeout | null = null;
        let debrisInterval: NodeJS.Timeout | null = null;

        function startPolling() {
            if (interval) return;
            interval = setInterval(() => {
                void loadFast();
            }, 15000);
            if (debrisInterval) return;
            debrisInterval = setInterval(() => {
                void loadDebris();
            }, 60000);
        }

        function stopPolling() {
            if (interval) {
                clearInterval(interval);
                interval = null;
            }
            if (debrisInterval) {
                clearInterval(debrisInterval);
                debrisInterval = null;
            }
        }

        function handleVisibility() {
            if (document.visibilityState === 'visible') startPolling();
            else stopPolling();
        }

        startPolling();
        document.addEventListener('visibilitychange', handleVisibility);

        return () => {
            active = false;
            stopPolling();
            document.removeEventListener('visibilitychange', handleVisibility);
        };
    }, []);

    async function handleStartAiCache() {
        setStartingAiCache(true);
        try {
            await startAiCache(20000);
            const status = await getAiCacheStatus();
            setCacheStatus(status);
        } catch {
        } finally {
            setStartingAiCache(false);
        }
    }

    return (
        <div className="relative w-screen h-screen overflow-hidden bg-black">
            {/* Full-page 3D Globe Background */}
            <div className="absolute inset-0">
                <Earth3D debris={debris} />
            </div>

            {/* Top right buttons */}
            <div className="fixed top-4 right-4 z-50 flex gap-2">
                {/* Refresh Data button */}
                <button
                    onClick={handleRefreshData}
                    disabled={refreshing}
                    className={`px-4 py-2 rounded-full backdrop-blur-xl border text-white text-sm transition-all ${refreshing
                            ? 'bg-yellow-500/30 border-yellow-400/50 cursor-wait'
                            : 'bg-green-500/20 border-green-400/50 hover:bg-green-500/40'
                        }`}
                >
                    {refreshing ? '⏳ Refreshing...' : '🔄 Refresh Data'}
                </button>

                {/* Toggle panels button */}
                <button
                    onClick={() => setShowPanels(!showPanels)}
                    className="px-4 py-2 rounded-full bg-black/50 backdrop-blur-xl border border-white/20 text-white text-sm hover:bg-white/10 transition-all"
                >
                    {showPanels ? '🔲 Hide Panels' : '📊 Show Panels'}
                </button>
            </div>

            {/* Refresh message toast */}
            {refreshMessage && (
                <div className="fixed top-16 left-1/2 -translate-x-1/2 z-50 px-6 py-3 rounded-lg bg-black/80 backdrop-blur-xl border border-white/20 text-white text-sm shadow-xl">
                    {refreshMessage}
                </div>
            )}

            <AnimatePresence>
                {showPanels && (
                    <>
                        {/* Header - Fixed at top */}
                        <motion.header
                            initial={{ opacity: 0, y: -50 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -50 }}
                            className="fixed top-0 left-0 right-0 z-40 p-4 bg-gradient-to-b from-black/80 to-transparent"
                        >
                            <div className="text-center">
                                <h1 className="text-2xl font-bold">
                                    <span className="text-2xl mr-2">🛰️</span>
                                    <span className="gradient-text">Cosmic Intelligence Dashboard</span>
                                </h1>
                                <div className="mt-1 text-[11px] text-gray-400">
                                    Educational prototype • Not for operational collision avoidance
                                </div>
                                {modelInfo && (
                                    <div className="mt-2 flex justify-center gap-3">
                                        <span className="px-2 py-1 rounded-full bg-green-500/20 text-green-400 text-xs font-semibold">
                                            🎯 {modelInfo.accuracy != null ? `${(modelInfo.accuracy * 100).toFixed(2)}% Accuracy` : 'Not evaluated'}
                                        </span>
                                        <span className="px-2 py-1 rounded-full bg-cyan-500/20 text-cyan-400 text-xs font-semibold">
                                            ⚡ {modelInfo.inference_speed_ms}ms Inference
                                        </span>
                                        <span className="px-2 py-1 rounded-full bg-purple-500/20 text-purple-400 text-xs font-semibold">
                                            🛰️ {debris.length} Objects
                                        </span>
                                    </div>
                                )}
                            </div>
                        </motion.header>

                        {/* Stats Bar - Bottom */}
                        {stats && (
                            <motion.div
                                initial={{ opacity: 0, y: 50 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: 50 }}
                                className="fixed bottom-0 left-0 right-0 z-40 p-4 bg-gradient-to-t from-black/80 to-transparent"
                            >
                                <div className="flex justify-center gap-3 flex-wrap">
                                    <MiniStat icon="🛰️" label="Total" value={stats.total_objects.toLocaleString()} color="#00BFFF" />
                                    <MiniStat icon="🧠" label="AI Enhanced" value={stats.ai_enhanced.toLocaleString()} color="#00ff88" />
                                    <MiniStat icon="🔴" label="Critical" value={stats.critical_count} color="#ff4444" />
                                    <MiniStat icon="🟠" label="High" value={stats.high_count} color="#ff8c00" />
                                    <MiniStat icon="🟡" label="Medium" value={stats.medium_count} color="#ffdd00" />
                                    <MiniStat icon="🟢" label="Low" value={stats.low_count} color="#00ff88" />
                                </div>
                                <p className="text-center text-gray-500 text-xs mt-2">
                                    Drag to rotate • Scroll to zoom • {debris.length} objects visible
                                </p>
                            </motion.div>
                        )}

                        {/* Right Sidebar - Floating */}
                        <motion.div
                            initial={{ opacity: 0, x: 100 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: 100 }}
                            className="fixed top-20 right-4 z-40 w-72 space-y-4 max-h-[calc(100vh-180px)] overflow-y-auto"
                        >
                            <SystemStatusPanel
                                health={health}
                                dataStatus={dataStatus}
                                cacheStatus={cacheStatus}
                                onStartAiCache={handleStartAiCache}
                                startingAiCache={startingAiCache}
                            />
                            <CdmModelPanel />
                            <ModelInfoPanel modelInfo={modelInfo} />
                            <PcCalculator />
                            <CollisionAlerts alerts={collisions} />
                        </motion.div>
                    </>
                )}
            </AnimatePresence>

            {/* Error message */}
            {error && (
                <div className="fixed top-20 left-1/2 transform -translate-x-1/2 z-50 p-4 rounded-xl bg-red-500/20 border border-red-500/30 text-red-400 text-center backdrop-blur-xl">
                    {error}
                </div>
            )}
        </div>
    );
}

// Mini stat component for bottom bar
function MiniStat({ icon, label, value, color }: { icon: string; label: string; value: string | number; color: string }) {
    return (
        <div
            className="px-4 py-2 rounded-xl backdrop-blur-xl flex items-center gap-2"
            style={{
                background: 'rgba(0,0,0,0.5)',
                border: `1px solid ${color}30`,
                boxShadow: `0 0 20px ${color}20`,
            }}
        >
            <span>{icon}</span>
            <div>
                <div className="text-xs text-gray-400">{label}</div>
                <div className="text-sm font-bold" style={{ color }}>{value}</div>
            </div>
        </div>
    );
}
