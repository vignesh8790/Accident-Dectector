import axios from 'axios';

const backendUrl = import.meta.env.VITE_API_URL || '';
const api = axios.create({ baseURL: backendUrl ? `${backendUrl}/api` : '/api' });

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
