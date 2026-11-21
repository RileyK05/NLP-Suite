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
  build: {
    target: "es2022",
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes("node_modules")) return;
          if (
            id.includes("/react/") ||
            id.includes("\\react\\") ||
            id.includes("/react-dom/") ||
            id.includes("\\react-dom\\") ||
            id.includes("/scheduler/") ||
            id.includes("\\scheduler\\")
          ) {
            return "react-vendor";
          }
          if (id.includes("@codemirror") || id.includes("@lezer")) {
            return "codemirror";
          }
          if (id.includes("@tauri-apps")) {
            return "tauri";
          }
          if (id.includes("lucide-react")) {
            return "icons";
          }
        },
      },
    },
  },
});
