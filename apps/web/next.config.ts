import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  transpilePackages: ["react-plotly.js"],
};

export default nextConfig;
