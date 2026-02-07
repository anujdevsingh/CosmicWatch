import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  reactCompiler: true,
  // Allow cross-origin requests from any IP on the local network
  allowedDevOrigins: [
    'http://192.168.1.4:3000',
    'http://192.168.1.4:3001',
    'http://localhost:3000',
    'http://localhost:3001',
    'http://127.0.0.1:3000',
    'http://127.0.0.1:3001',
  ],
};

export default nextConfig;
