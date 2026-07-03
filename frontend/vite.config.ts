import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

// Vite build config for the oncology-arbiter tumor-board SPA.
// Output is emitted to ../src/oncology_arbiter/api/static/dist/ so the
// FastAPI app can serve it via the /ui static mount when
// ONCOLOGY_ARBITER_SERVE_FRONTEND=1.
export default defineConfig({
  plugins: [react()],
  base: "/ui/",
  build: {
    outDir: path.resolve(__dirname, "..", "src", "oncology_arbiter", "api", "static", "dist"),
    emptyOutDir: true,
    sourcemap: false,
  },
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: {
      "/v1": "http://localhost:8000",
      "/health": "http://localhost:8000",
    },
  },
});
