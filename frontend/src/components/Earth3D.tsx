'use client';

import { useRef, useMemo, useState, useCallback, Suspense } from 'react';
import { Canvas, useFrame, useLoader } from '@react-three/fiber';
import { OrbitControls, Stars, Html } from '@react-three/drei';
import * as THREE from 'three';
import { TextureLoader } from 'three';
import { DebrisObject } from '@/lib/api';

// Risk level colors
const RISK_COLORS: Record<string, string> = {
    CRITICAL: '#ff4444',
    HIGH: '#ff8c00',
    MEDIUM: '#ffdd00',
    LOW: '#00ff88',
};

// Earth component with realistic textures
function EarthWithTexture() {
    const earthRef = useRef<THREE.Mesh>(null);

    // Use reliable Three.js example texture (NASA Blue Marble)
    const earthTexture = useLoader(
        TextureLoader,
        'https://threejs.org/examples/textures/planets/earth_atmos_2048.jpg'
    );

    useFrame(() => {
        if (earthRef.current) {
            earthRef.current.rotation.y += 0.0003;
        }
    });

    return (
        <group>
            {/* Main Earth sphere with realistic texture */}
            <mesh ref={earthRef}>
                <sphereGeometry args={[1, 64, 64]} />
                <meshPhongMaterial
                    map={earthTexture}
                    specular={new THREE.Color('#333333')}
                    shininess={10}
                />
            </mesh>

            {/* Atmosphere glow - inner layer (soft blue haze) */}
            <mesh scale={1.02}>
                <sphereGeometry args={[1, 32, 32]} />
                <meshBasicMaterial
                    color="#66ccff"
                    transparent
                    opacity={0.2}
                    side={THREE.BackSide}
                />
            </mesh>

            {/* Atmosphere glow - middle layer */}
            <mesh scale={1.05}>
                <sphereGeometry args={[1, 32, 32]} />
                <meshBasicMaterial
                    color="#4499ff"
                    transparent
                    opacity={0.12}
                    side={THREE.BackSide}
                />
            </mesh>

            {/* Atmospheric rim effect - outer glow */}
            <mesh scale={1.1}>
                <sphereGeometry args={[1, 32, 32]} />
                <meshBasicMaterial
                    color="#2277ff"
                    transparent
                    opacity={0.06}
                    side={THREE.BackSide}
                />
            </mesh>
        </group>
    );
}


// ... DebrisMarker code remains the same ...

// Individual clickable debris marker
function DebrisMarker({ debris, onSelect }: { debris: DebrisObject; onSelect: (d: DebrisObject) => void }) {
    const meshRef = useRef<THREE.Mesh>(null);
    const [hovered, setHovered] = useState(false);

    // Convert lat/lon to 3D position
    const position = useMemo(() => {
        const radius = 1.15 + (debris.altitude / 6000);
        const lat = (debris.latitude * Math.PI) / 180;
        const lon = (debris.longitude * Math.PI) / 180;

        return new THREE.Vector3(
            radius * Math.cos(lat) * Math.cos(lon),
            radius * Math.sin(lat),
            radius * Math.cos(lat) * Math.sin(lon)
        );
    }, [debris]);

    const isSatellite = debris.object_type === 'SATELLITE';
    const color = isSatellite ? '#00ffff' : (RISK_COLORS[debris.risk_level] || RISK_COLORS.LOW);
    const size = 0.02 + (debris.risk_score * 0.025);

    return (
        <mesh
            ref={meshRef}
            position={position}
            onClick={(e) => {
                e.stopPropagation();
                onSelect(debris);
            }}
            onPointerOver={(e) => {
                e.stopPropagation();
                setHovered(true);
                document.body.style.cursor = 'pointer';
            }}
            onPointerOut={() => {
                setHovered(false);
                document.body.style.cursor = 'auto';
            }}
        >
            {isSatellite ? (
                <boxGeometry args={[hovered ? size * 1.5 : size, hovered ? size * 1.5 : size, hovered ? size * 1.5 : size]} />
            ) : (
                <sphereGeometry args={[hovered ? size * 1.5 : size, 12, 12]} />
            )}

            <meshBasicMaterial
                color={color}
                transparent
                opacity={hovered ? 1 : (isSatellite ? 0.9 : 0.85)}
            />
            {hovered && (
                <Html distanceFactor={8}>
                    <div className="px-3 py-2 bg-black/90 text-white text-xs rounded-lg whitespace-nowrap backdrop-blur-sm border border-white/20 shadow-lg">
                        <div className="flex items-center gap-2 mb-1">
                            <span className={`w-2 h-2 rounded-full ${isSatellite ? 'bg-cyan-400' : 'bg-red-400'}`}></span>
                            <span className="font-bold text-gray-200">{isSatellite ? 'SATELLITE' : 'DEBRIS'}</span>
                        </div>
                        <div className="font-bold text-lg mb-1" style={{ color }}>{debris.id}</div>
                        <div className="text-gray-400 text-[10px]">{debris.altitude.toFixed(0)} km</div>
                        {!isSatellite && (
                            <div className="mt-1 font-semibold" style={{ color: RISK_COLORS[debris.risk_level] }}>{debris.risk_level} RISK</div>
                        )}
                    </div>
                </Html>
            )}
        </mesh>
    );
}

// Debris markers component with hover callback
function DebrisMarkers({ debris, onSelectDebris, onHoverChange }: { debris: DebrisObject[]; onSelectDebris: (d: DebrisObject) => void; onHoverChange: (hovering: boolean) => void }) {
    // Show more debris markers for better visual impact
    const visibleDebris = useMemo(() => debris.slice(0, 500), [debris]);

    return (
        <group
            onPointerOver={() => onHoverChange(true)}
            onPointerOut={() => onHoverChange(false)}
        >
            {visibleDebris.map((d) => (
                <DebrisMarker key={d.id} debris={d} onSelect={onSelectDebris} />
            ))}
        </group>
    );
}

// Main 3D scene
function Scene({ debris, onSelectDebris }: { debris: DebrisObject[]; onSelectDebris: (d: DebrisObject) => void }) {
    const [autoRotate, setAutoRotate] = useState(false); // Default to false as requested

    return (
        <>
            {/* Enhanced Lighting - brighter Earth that stands out */}
            <ambientLight intensity={0.2} />
            <directionalLight position={[5, 3, 5]} intensity={3.0} color="#ffffff" />
            <directionalLight position={[-3, 1, 3]} intensity={1.0} color="#99bbff" />
            <pointLight position={[-10, -10, -10]} intensity={0.3} color="#4499ff" />

            {/* Subtle stars - reduced for cleaner background */}
            <Stars
                radius={400}
                depth={60}
                count={2500}
                factor={2}
                saturation={0}
                fade
                speed={0.1}
            />

            <Suspense fallback={null}>
                <EarthWithTexture />
            </Suspense>

            {debris.length > 0 && (
                <DebrisMarkers
                    debris={debris}
                    onSelectDebris={onSelectDebris}
                    onHoverChange={() => { }} // No longer needed for rotation
                />
            )}

            <OrbitControls
                enablePan={false}
                minDistance={2.1}
                maxDistance={10}
                autoRotate={autoRotate}
                autoRotateSpeed={0.5}
                enableDamping
                dampingFactor={0.05}
                rotateSpeed={0.5}
                zoomSpeed={0.8}
            />

            <Html position={[0, -2, 0]} center>
                <div className="flex gap-2 pointer-events-auto" style={{ userSelect: 'none' }}>
                    <button
                        onClick={() => setAutoRotate(!autoRotate)}
                        className={`px-4 py-2 rounded-full text-xs font-bold transition-all border border-white/20 backdrop-blur-sm ${autoRotate
                            ? 'bg-cyan-500/80 text-white shadow-[0_0_20px_rgba(6,182,212,0.4)]'
                            : 'bg-black/40 text-gray-300 hover:bg-white/10'
                            }`}
                    >
                        {autoRotate ? '⏸ PAUSE ROTATION' : '▶ START ROTATION'}
                    </button>
                </div>
            </Html>
        </>
    );
}

// Debris detail popup
function DebrisDetailPopup({ debris, onClose }: { debris: DebrisObject; onClose: () => void }) {
    const riskColor = RISK_COLORS[debris.risk_level] || '#00ff88';

    return (
        <div className="fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 z-50">
            <div
                className="p-6 rounded-2xl backdrop-blur-xl min-w-[320px]"
                style={{
                    background: 'linear-gradient(135deg, rgba(0,0,0,0.95) 0%, rgba(20,20,40,0.95) 100%)',
                    border: `2px solid ${riskColor}50`,
                    boxShadow: `0 0 40px ${riskColor}30`,
                }}
            >
                <div className="flex justify-between items-start mb-4">
                    <div>
                        <div className="flex items-center gap-2 mb-1">
                            <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${debris.object_type === 'SATELLITE' ? 'bg-cyan-500/20 text-cyan-400' : 'bg-red-500/20 text-red-400'}`}>
                                {debris.object_type || 'DEBRIS'}
                            </span>
                            <span
                                className="text-xs px-2 py-0.5 rounded-full font-semibold"
                                style={{ background: `${riskColor}30`, color: riskColor }}
                            >
                                {debris.risk_level} RISK
                            </span>
                        </div>
                        <h3 className="text-xl font-bold text-white">{debris.id}</h3>
                    </div>
                    <button
                        onClick={onClose}
                        className="text-gray-400 hover:text-white text-2xl leading-none w-8 h-8 flex items-center justify-center rounded-full hover:bg-white/10"
                    >
                        ×
                    </button>
                </div>

                <div className="space-y-3 text-sm">
                    <div className="grid grid-cols-2 gap-4">
                        <InfoRow label="Altitude" value={`${debris.altitude.toFixed(1)} km`} />
                        <InfoRow label="Velocity" value={`${debris.velocity.toFixed(2)} km/s`} />
                        <InfoRow label="Inclination" value={`${debris.inclination.toFixed(1)}°`} />
                        <InfoRow label="Size" value={`${debris.size.toFixed(2)} m`} />
                        <InfoRow label="Latitude" value={`${debris.latitude.toFixed(2)}°`} />
                        <InfoRow label="Longitude" value={`${debris.longitude.toFixed(2)}°`} />
                    </div>

                    <div className="pt-3 border-t border-white/10">
                        <div className="flex justify-between items-center">
                            <span className="text-gray-400">AI Confidence</span>
                            <div className="flex items-center gap-2">
                                <div className="w-24 h-2 bg-gray-700 rounded-full overflow-hidden">
                                    <div
                                        className="h-full rounded-full"
                                        style={{
                                            width: `${debris.confidence * 100}%`,
                                            background: `linear-gradient(90deg, ${riskColor}, ${riskColor}88)`
                                        }}
                                    />
                                </div>
                                <span className="text-white font-semibold">{(debris.confidence * 100).toFixed(1)}%</span>
                            </div>
                        </div>
                    </div>

                    <div className="pt-3 border-t border-white/10">
                        <div className="flex justify-between items-center">
                            <span className="text-gray-400">Risk Score</span>
                            <span className="text-2xl font-bold" style={{ color: riskColor }}>
                                {(debris.risk_score * 100).toFixed(0)}%
                            </span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

function InfoRow({ label, value }: { label: string; value: string }) {
    return (
        <div>
            <div className="text-gray-500 text-xs">{label}</div>
            <div className="text-white font-medium">{value}</div>
        </div>
    );
}

// Exported component - FULL SCREEN
interface Earth3DProps {
    debris: DebrisObject[];
}

export default function Earth3D({ debris }: Earth3DProps) {
    const [selectedDebris, setSelectedDebris] = useState<DebrisObject | null>(null);

    const handleSelectDebris = useCallback((d: DebrisObject) => {
        setSelectedDebris(d);
    }, []);

    return (
        <div className="w-full h-full relative" style={{ background: '#000' }}>
            <Canvas
                camera={{ position: [0, 0, 4.5], fov: 45, near: 0.1, far: 1000 }}
                gl={{ antialias: true, alpha: false }}
                onCreated={({ gl }) => {
                    gl.setClearColor('#000000');
                }}
            >
                <Scene debris={debris} onSelectDebris={handleSelectDebris} />
            </Canvas>

            {/* Debris detail popup */}
            {selectedDebris && (
                <>
                    <div
                        className="fixed inset-0 bg-black/50 z-40"
                        onClick={() => setSelectedDebris(null)}
                    />
                    <DebrisDetailPopup
                        debris={selectedDebris}
                        onClose={() => setSelectedDebris(null)}
                    />
                </>
            )}
        </div>
    );
}
