'use client';

import { useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { CdmModelInfo, CdmPredictRequest, CdmPredictResponse, getCdmModelInfo, predictCdm } from '@/lib/api';

function fmtPercent(v: number | null | undefined): string {
    if (v == null || Number.isNaN(v)) return 'N/A';
    return `${(v * 100).toFixed(1)}%`;
}

const riskLabels: Record<string, string> = {
    '0': 'LOW',
    '1': 'MEDIUM',
    '2': 'HIGH',
    '3': 'CRITICAL',
};

export default function CdmModelPanel() {
    const [info, setInfo] = useState<CdmModelInfo | null>(null);
    const [pred, setPred] = useState<CdmPredictResponse | null>(null);
    const [loadingInfo, setLoadingInfo] = useState(false);
    const [loadingPred, setLoadingPred] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const [form, setForm] = useState<CdmPredictRequest>({
        MIN_RNG: 500,
        hours_to_tca: 24,
        SAT1_OBJECT_TYPE: 'DEBRIS',
        SAT2_OBJECT_TYPE: 'DEBRIS',
        SAT1_RCS: 'SMALL',
        SAT2_RCS: 'SMALL',
        SAT_1_EXCL_VOL: 1,
        SAT_2_EXCL_VOL: 1,
    });

    useEffect(() => {
        let active = true;
        setLoadingInfo(true);
        void getCdmModelInfo()
            .then((d) => {
                if (!active) return;
                setInfo(d);
            })
            .catch((e) => {
                if (!active) return;
                setError(e instanceof Error ? e.message : 'Failed to load CDM model info');
            })
            .finally(() => {
                if (!active) return;
                setLoadingInfo(false);
            });
        return () => {
            active = false;
        };
    }, []);

    const sortedProbs = useMemo(() => {
        if (!pred) return [];
        return Object.entries(pred.probabilities).sort((a, b) => b[1] - a[1]);
    }, [pred]);

    async function handlePredict() {
        setLoadingPred(true);
        setError(null);
        try {
            const out = await predictCdm(form);
            setPred(out);
        } catch (e) {
            setError(e instanceof Error ? e.message : 'Prediction failed');
        } finally {
            setLoadingPred(false);
        }
    }

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-2xl p-4 backdrop-blur-xl"
            style={{
                background: 'linear-gradient(135deg, rgba(0,0,0,0.7) 0%, rgba(30,60,60,0.35) 100%)',
                border: '1px solid rgba(255,255,255,0.14)',
            }}
        >
            <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold text-white">CDM Model</h3>
                <div className="text-[11px] text-gray-400">{loadingInfo ? 'Loading…' : info ? info.kind.toUpperCase() : 'N/A'}</div>
            </div>

            {info ? (
                <div className="grid grid-cols-2 gap-2 text-xs mb-3">
                    <div className="p-2 rounded-lg bg-black/30 border border-white/10">
                        <div className="text-gray-400">Target</div>
                        <div className="font-semibold text-cyan-300">{info.target}</div>
                    </div>
                    <div className="p-2 rounded-lg bg-black/30 border border-white/10">
                        <div className="text-gray-400">Test Macro-F1</div>
                        <div className="font-semibold text-green-300">{info.test_macro_f1 != null ? info.test_macro_f1.toFixed(3) : 'N/A'}</div>
                    </div>
                    <div className="p-2 rounded-lg bg-black/30 border border-white/10">
                        <div className="text-gray-400">Val Macro-F1</div>
                        <div className="font-semibold text-green-300">{info.val_macro_f1 != null ? info.val_macro_f1.toFixed(3) : 'N/A'}</div>
                    </div>
                    <div className="p-2 rounded-lg bg-black/30 border border-white/10">
                        <div className="text-gray-400">Test Acc</div>
                        <div className="font-semibold text-purple-200">{fmtPercent(info.test_accuracy)}</div>
                    </div>
                </div>
            ) : (
                <div className="text-xs text-gray-400 mb-3">CDM model info unavailable.</div>
            )}

            <div className="grid grid-cols-2 gap-2 text-xs">
                <label className="block">
                    <div className="text-gray-400 mb-1">MIN_RNG</div>
                    <input
                        value={form.MIN_RNG ?? ''}
                        onChange={(e) => setForm((s) => ({ ...s, MIN_RNG: e.target.value === '' ? null : Number(e.target.value) }))}
                        type="number"
                        className="w-full px-2 py-2 rounded-lg bg-black/30 border border-white/10 text-white text-xs"
                    />
                </label>
                <label className="block">
                    <div className="text-gray-400 mb-1">Hours to TCA</div>
                    <input
                        value={form.hours_to_tca ?? ''}
                        onChange={(e) => setForm((s) => ({ ...s, hours_to_tca: e.target.value === '' ? null : Number(e.target.value) }))}
                        type="number"
                        className="w-full px-2 py-2 rounded-lg bg-black/30 border border-white/10 text-white text-xs"
                    />
                </label>
                <label className="block">
                    <div className="text-gray-400 mb-1">SAT1 Type</div>
                    <input
                        value={form.SAT1_OBJECT_TYPE ?? ''}
                        onChange={(e) => setForm((s) => ({ ...s, SAT1_OBJECT_TYPE: e.target.value }))}
                        className="w-full px-2 py-2 rounded-lg bg-black/30 border border-white/10 text-white text-xs"
                    />
                </label>
                <label className="block">
                    <div className="text-gray-400 mb-1">SAT2 Type</div>
                    <input
                        value={form.SAT2_OBJECT_TYPE ?? ''}
                        onChange={(e) => setForm((s) => ({ ...s, SAT2_OBJECT_TYPE: e.target.value }))}
                        className="w-full px-2 py-2 rounded-lg bg-black/30 border border-white/10 text-white text-xs"
                    />
                </label>
                <label className="block">
                    <div className="text-gray-400 mb-1">SAT1 RCS</div>
                    <input
                        value={form.SAT1_RCS ?? ''}
                        onChange={(e) => setForm((s) => ({ ...s, SAT1_RCS: e.target.value }))}
                        className="w-full px-2 py-2 rounded-lg bg-black/30 border border-white/10 text-white text-xs"
                    />
                </label>
                <label className="block">
                    <div className="text-gray-400 mb-1">SAT2 RCS</div>
                    <input
                        value={form.SAT2_RCS ?? ''}
                        onChange={(e) => setForm((s) => ({ ...s, SAT2_RCS: e.target.value }))}
                        className="w-full px-2 py-2 rounded-lg bg-black/30 border border-white/10 text-white text-xs"
                    />
                </label>
                <label className="block">
                    <div className="text-gray-400 mb-1">SAT1 Excl Vol</div>
                    <input
                        value={form.SAT_1_EXCL_VOL ?? ''}
                        onChange={(e) => setForm((s) => ({ ...s, SAT_1_EXCL_VOL: e.target.value === '' ? null : Number(e.target.value) }))}
                        type="number"
                        className="w-full px-2 py-2 rounded-lg bg-black/30 border border-white/10 text-white text-xs"
                    />
                </label>
                <label className="block">
                    <div className="text-gray-400 mb-1">SAT2 Excl Vol</div>
                    <input
                        value={form.SAT_2_EXCL_VOL ?? ''}
                        onChange={(e) => setForm((s) => ({ ...s, SAT_2_EXCL_VOL: e.target.value === '' ? null : Number(e.target.value) }))}
                        type="number"
                        className="w-full px-2 py-2 rounded-lg bg-black/30 border border-white/10 text-white text-xs"
                    />
                </label>
            </div>

            <button
                onClick={handlePredict}
                disabled={loadingPred}
                className="mt-3 w-full px-3 py-2 rounded-lg bg-cyan-500/20 border border-cyan-400/30 text-cyan-200 text-xs font-semibold hover:bg-cyan-500/30 disabled:opacity-50 disabled:cursor-not-allowed"
            >
                {loadingPred ? 'Scoring…' : 'Score CDM'}
            </button>

            {error ? <div className="mt-2 text-[11px] text-red-300">{error}</div> : null}

            {pred ? (
                <div className="mt-3 p-2 rounded-lg bg-black/30 border border-white/10">
                    <div className="flex items-center justify-between">
                        <div className="text-xs text-gray-400">Prediction</div>
                        <div className="text-xs font-semibold text-white">
                            {pred.predicted_class} • {riskLabels[pred.predicted_class] ?? 'CLASS'}
                        </div>
                    </div>
                    <div className="mt-2 space-y-1">
                        {sortedProbs.map(([k, v]) => (
                            <div key={k} className="flex items-center justify-between text-[11px] text-gray-300">
                                <div>
                                    {k} {riskLabels[k] ? `(${riskLabels[k]})` : ''}
                                </div>
                                <div className="font-semibold text-gray-100">{fmtPercent(v)}</div>
                            </div>
                        ))}
                    </div>
                </div>
            ) : null}
        </motion.div>
    );
}

