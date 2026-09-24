import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "./",
  server: {
    port: 1420,
    strictPort: true,
    // Tauri watches native sources itself. Watching its target directory here
    // races Windows executable locks during rebuilds and can crash Vite.
    watch: { ignored: ["**/src-tauri/**", "**/.toolchain/**"] },
  },
  clearScreen: false,
});
