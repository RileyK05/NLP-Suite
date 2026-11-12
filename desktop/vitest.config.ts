import { defineConfig, mergeConfig } from "vitest/config";
import viteConfig from "./vite.config";

export default mergeConfig(
  viteConfig,
  defineConfig({
    test: {
      // Most component files create a jsdom environment. Starting one per
      // CPU at once spends more time and memory loading runtimes than testing.
      maxWorkers: 4,
    },
  }),
);
