// Shared API base URL. Set VITE_API_BASE_URL at build time to point at a
// deployed backend (e.g. its Railway URL); falls back to the local Flask
// dev server otherwise.
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:5000";
