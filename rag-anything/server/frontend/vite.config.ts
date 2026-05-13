/// <reference types="vitest" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
  build: {
    outDir: '../static/dist',
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      '/ingest':     { target: 'http://localhost:9621', changeOrigin: true },
      '/query':      { target: 'http://localhost:9621', changeOrigin: true },
      '/jobs':       { target: 'http://localhost:9621', changeOrigin: true },
      '/graph':      { target: 'http://localhost:9621', changeOrigin: true },
      '/workspace':  { target: 'http://localhost:9621', changeOrigin: true },
      '/workspaces': { target: 'http://localhost:9621', changeOrigin: true },
      '/files':      { target: 'http://localhost:9621', changeOrigin: true },
      '/content':    { target: 'http://localhost:9621', changeOrigin: true },
      '/uploads':    { target: 'http://localhost:9621', changeOrigin: true },
      '/output':     { target: 'http://localhost:9621', changeOrigin: true },
      '/config':     { target: 'http://localhost:9621', changeOrigin: true },
      '/retry':      { target: 'http://localhost:9621', changeOrigin: true },
      '/evaluate':   { target: 'http://localhost:9621', changeOrigin: true },
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: [],
  },
})
