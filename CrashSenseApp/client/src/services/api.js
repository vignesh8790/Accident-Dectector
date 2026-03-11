import axios from 'axios';

export const backendUrl = import.meta.env.VITE_API_URL || '';
const api = axios.create({ baseURL: backendUrl ? `${backendUrl}/api` : '/api' });

export const getMediaUrl = (path) => {
  if (!path) return '';
  if (path.startsWith('http')) return path;
  // If no backendUrl set (local dev), and path starts with /, return as is
  if (!backendUrl && path.startsWith('/')) return path;
  // Prevent double slash if backendUrl is present but ends without slash and path starts with slash
  const cleanPath = path.startsWith('/') ? path.substring(1) : path;
  return backendUrl ? `${backendUrl}/${cleanPath}` : `/${cleanPath}`;
};

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('crashsense_token');
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('crashsense_token');
      if (window.location.pathname !== '/login') {
        window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  }
);

export default api;
