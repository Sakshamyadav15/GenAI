import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

// `npm run dev` talks to the general backend (:8000); `npm run dev:legal` (mode "legal", see .env.legal)
// talks to the Indian-Acts backend (:8001) on its own port, so both can run side by side.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '.', '');
  return {
    plugins: [react(), tailwindcss()],
    server: {
      port: Number(env.VITE_DEV_PORT) || 5173,
      proxy: { '/api': { target: env.VITE_API_TARGET || 'http://localhost:8000', changeOrigin: true } },
    },
  };
});
