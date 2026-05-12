import axios from 'axios'

const client = axios.create({ baseURL: '/' })

client.interceptors.response.use(
  (r) => r,
  (err) => {
    const detail = err.response?.data?.detail ?? err.message
    return Promise.reject(new Error(detail))
  }
)

export default client
