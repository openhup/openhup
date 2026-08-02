import { sveltekit } from '@sveltejs/kit/vite';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [tailwindcss(), sveltekit()],
  server: {
    // Proxy the API in development so the frontend is same-origin here as it is in production.
    // Avoids CORS configuration existing at all, which is one less thing to get wrong.
    proxy: {
      '/api': {
        target: process.env.OPENHUP_API ?? 'http://127.0.0.1:8080',
        changeOrigin: true,
        ws: true
      }
    }
  }
});
