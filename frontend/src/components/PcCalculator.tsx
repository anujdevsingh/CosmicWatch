'use client';

import { useMemo, useState } from 'react';
import { computePcFromCdm, ConjunctionPcResponse } from '@/lib/api';

export default function PcCalculator() {
    const [cdmText, setCdmText] = useState('');
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<ConjunctionPcResponse | null>(null);
    const [error, setError] = useState<string | null>(null);

    const hasInput = useMemo(() => cdmText.trim().length > 0, [cdmText]);

    async function handleCompute() {
        setLoading(true);
        setError(null);
        setResult(null);
        try {
            const res = await computePcFromCdm(cdmText);
            setResult(res);
        } catch (e) {
            const message = e instanceof Error ? e.message : 'Failed to compute Pc';
            setError(message);
        } finally {
            setLoading(false);
        }
    }

    return (
        <div className="bg-black/40 border border-white/10 rounded-2xl p-4 backdrop-blur-xl">
            <h3 className="text-white font-semibold text-sm mb-2">Pc Calculator (CDM)</h3>
            <p className="text-gray-400 text-xs mb-3">
                Paste a CDM-like KVN text with OBJECT1/OBJECT2 state + RTN covariance + HARD_BODY_RADIUS_KM.
            </p>

            <textarea
                value={cdmText}
                onChange={(e) => setCdmText(e.target.value)}
                placeholder="OBJECT1_X = ...&#10;OBJECT1_Y = ...&#10;...&#10;HARD_BODY_RADIUS_KM = ..."
                className="w-full h-28 resize-none rounded-xl bg-black/50 border border-white/10 text-gray-100 text-xs p-3 outline-none focus:border-cyan-400/40"
            />

            <button
                onClick={handleCompute}
                disabled={!hasInput || loading}
                className={`mt-3 w-full px-3 py-2 rounded-xl text-sm transition-all border ${loading || !hasInput
                    ? 'bg-white/5 border-white/10 text-gray-500 cursor-not-allowed'
                    : 'bg-cyan-500/15 border-cyan-400/30 text-cyan-200 hover:bg-cyan-500/25'
                    }`}
            >
                {loading ? 'Computing…' : 'Compute Pc'}
            </button>

            {error && (
                <div className="mt-3 rounded-xl bg-red-500/10 border border-red-500/20 p-3 text-xs text-red-300">
                    {error}
                </div>
            )}

            {result && (
                <div className="mt-3 rounded-xl bg-white/5 border border-white/10 p-3 text-xs text-gray-200 space-y-1">
                    <div className="flex justify-between"><span className="text-gray-400">Pc</span><span>{result.pc.toExponential(3)}</span></div>
                    <div className="flex justify-between"><span className="text-gray-400">Miss Distance (km)</span><span>{result.miss_distance_km.toFixed(6)}</span></div>
                    <div className="flex justify-between"><span className="text-gray-400">HBR (km)</span><span>{result.hard_body_radius_km.toFixed(6)}</span></div>
                    <div className="flex justify-between"><span className="text-gray-400">Rel Speed (km/s)</span><span>{result.rel_speed_km_s.toFixed(6)}</span></div>
                    <div className="flex justify-between"><span className="text-gray-400">σ Major/Minor (km)</span><span>{result.sigma_major_km.toFixed(6)} / {result.sigma_minor_km.toFixed(6)}</span></div>
                    <div className="text-[11px] text-gray-500 mt-2">{result.notes}</div>
                </div>
            )}
        </div>
    );
}

