const IMAGES = new Set([".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"]);

export function viewableArtifact(path: string): boolean {
  const dot = path.lastIndexOf(".");
  const suffix = dot >= 0 ? path.slice(dot).toLowerCase() : "";
  return IMAGES.has(suffix) || /\.(html?|pdf|svg|txt|kml|md)$/.test(suffix);
}
