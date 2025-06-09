/**
 * NeuroCluster Streamer JavaScript SDK
 * Official JavaScript/TypeScript client library for the NCS API
 * 
 * @version 1.0.0
 * @author NCS API Development Team
 * @license MIT
 */

(function (global, factory) {
  // UMD pattern for universal module support
  typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports) :
  typeof define === 'function' && define.amd ? define(['exports'], factory) :
  (global = typeof globalThis !== 'undefined' ? globalThis : global || self, factory(global.NCS = {}));
}(this, (function (exports) { 'use strict';

  // Determine environment
  const isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
  const isBrowser = typeof window !== 'undefined';

  // HTTP client abstraction
  class HttpClient {
    constructor(config = {}) {
      this.baseUrl = config.baseUrl || '';
      this.timeout = config.timeout || 30000;
      this.headers = config.headers || {};
      this.interceptors = {
        request: [],
        response: []
      };
    }

    use(interceptor) {
      if (interceptor.request) this.interceptors.request.push(interceptor.request);
      if (interceptor.response) this.interceptors.response.push(interceptor.response);
    }

    async request(config) {
      // Apply request interceptors
      for (const interceptor of this.interceptors.request) {
        config = interceptor(config) || config;
      }

      const url = `${this.baseUrl}${config.url}`;
      const options = {
        method: config.method || 'GET',
        headers: { ...this.headers, ...config.headers },
        ...config
      };

      if (config.data) {
        options.body = JSON.stringify(config.data);
        options.headers['Content-Type'] = 'application/json';
      }

      try {
        let response;
        
        if (isNode) {
          // Node.js environment
          const https = require('https');
          const http = require('http');
          const urlParse = require('url').parse;
          
          response = await this.nodeRequest(url, options);
        } else {
          // Browser environment
          const controller = new AbortController();
          const timeoutId = setTimeout(() => controller.abort(), this.timeout);
          
          options.signal = controller.signal;
          
          try {
            response = await fetch(url, options);
            clearTimeout(timeoutId);
          } catch (error) {
            clearTimeout(timeoutId);
            throw error;
          }
        }

        // Apply response interceptors
        for (const interceptor of this.interceptors.response) {
          response = interceptor(response) || response;
        }

        return response;
      } catch (error) {
        // Apply error interceptors
        for (const interceptor of this.interceptors.response) {
          if (interceptor.length > 1) {
            try {
              return interceptor(null, error);
            } catch (e) {
              error = e;
            }
          }
        }
        throw error;
      }
    }

    async nodeRequest(url, options) {
      const https = require('https');
      const http = require('http');
      const urlParse = require('url').parse;
      
      return new Promise((resolve, reject) => {
        const parsedUrl = urlParse(url);
        const isHttps = parsedUrl.protocol === 'https:';
        const client = isHttps ? https : http;
        
        const reqOptions = {
          hostname: parsedUrl.hostname,
          port: parsedUrl.port || (isHttps ? 443 : 80),
          path: parsedUrl.path,
          method: options.method,
          headers: options.headers,
          timeout: this.timeout
        };

        const req = client.request(reqOptions, (res) => {
          let data = '';
          res.on('data', chunk => data += chunk);
          res.on('end', () => {
            const response = {
              status: res.statusCode,
              statusText: res.statusMessage,
              headers: res.headers,
              data: data,
              json: async () => JSON.parse(data),
              text: async () => data,
              ok: res.statusCode >= 200 && res.statusCode < 300
            };
            resolve(response);
          });
        });

        req.on('error', reject);
        req.on('timeout', () => reject(new Error('Request timeout')));

        if (options.body) {
          req.write(options.body);
        }
        
        req.end();
      });
    }
  }

  // Custom error classes
  class NCSError extends Error {
    constructor(message, errorCode = null, requestId = null) {
      super(message);
      this.name = 'NCSError';
      this.errorCode = errorCode;
      this.requestId = requestId;
    }
  }

  class AuthenticationError extends NCSError {
    constructor(message = 'Authentication failed') {
      super(message);
      this.name = 'AuthenticationError';
    }
  }

  class RateLimitError extends NCSError {
    constructor(message = 'Rate limit exceeded', retryAfter = null) {
      super(message);
      this.name = 'RateLimitError';
      this.retryAfter = retryAfter;
    }
  }

  class ValidationError extends NCSError {
    constructor(message = 'Validation failed', details = null) {
      super(message);
      this.name = 'ValidationError';
      this.details = details;
    }
  }

  class ProcessingError extends NCSError {
    constructor(message = 'Processing failed') {
      super(message);
      this.name = 'ProcessingError';
    }
  }

  class ConnectionError extends NCSError {
    constructor(message = 'Connection failed') {
      super(message);
      this.name = 'ConnectionError';
    }
  }

  // Rate limiter utility
  class RateLimiter {
    constructor(requestsPerMinute = 100) {
      this.requestsPerMinute = requestsPerMinute;
      this.requests = [];
    }

    async checkLimit() {
      const now = Date.now();
      const oneMinuteAgo = now - 60000;
      
      // Remove old requests
      this.requests = this.requests.filter(time => time > oneMinuteAgo);
      
      if (this.requests.length >= this.requestsPerMinute) {
        const oldestRequest = Math.min(...this.requests);
        const waitTime = 60000 - (now - oldestRequest);
        await this.sleep(waitTime);
        return this.checkLimit();
      }
      
      this.requests.push(now);
    }

    sleep(ms) {
      return new Promise(resolve => setTimeout(resolve, ms));
    }
  }

  // WebSocket streaming connection
  class StreamingConnection {
    constructor(url, options = {}) {
      this.url = url;
      this.options = options;
      this.ws = null;
      this.connected = false;
      this.reconnectAttempts = 0;
      this.maxReconnectAttempts = options.reconnectAttempts || 5;
      
      this.onMessage = options.onMessage || (() => {});
      this.onError = options.onError || (() => {});
      this.onClose = options.onClose || (() => {});
      this.onReconnect = options.onReconnect || (() => {});
    }

    async connect() {
      return new Promise((resolve, reject) => {
        try {
          // Choose WebSocket implementation
          let WebSocketClass;
          if (isBrowser) {
            WebSocketClass = WebSocket;
          } else {
            WebSocketClass = require('ws');
          }

          this.ws = new WebSocketClass(this.url);
          
          this.ws.onopen = () => {
            this.connected = true;
            this.reconnectAttempts = 0;
            resolve();
          };

          this.ws.onmessage = (event) => {
            try {
              const data = JSON.parse(event.data);
              this.onMessage(data);
            } catch (error) {
              this.onError(new Error('Failed to parse message: ' + error.message));
            }
          };

          this.ws.onclose = () => {
            this.connected = false;
            this.onClose();
            
            if (this.reconnectAttempts < this.maxReconnectAttempts) {
              this.handleReconnect();
            }
          };

          this.ws.onerror = (error) => {
            this.onError(error);
            if (!this.connected) reject(error);
          };
        } catch (error) {
          reject(error);
        }
      });
    }

    async sendPoint(point) {
      if (!this.connected) throw new Error('WebSocket not connected');
      
      const message = JSON.stringify({
        type: 'point',
        data: point
      });
      
      this.ws.send(message);
    }

    async sendBatch(points) {
      if (!this.connected) throw new Error('WebSocket not connected');
      
      const message = JSON.stringify({
        type: 'batch',
        data: points
      });
      
      this.ws.send(message);
    }

    handleReconnect() {
      if (this.reconnectAttempts < this.maxReconnectAttempts) {
        this.reconnectAttempts++;
        const delay = Math.pow(2, this.reconnectAttempts) * 1000;
        
        this.onReconnect(this.reconnectAttempts);
        
        setTimeout(() => {
          console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
          this.connect().catch(console.error);
        }, delay);
      }
    }

    async stop() {
      if (this.ws) {
        this.ws.close();
        this.ws = null;
        this.connected = false;
      }
    }
  }

  // Main NCS Client class
  class NCSClient {
    constructor(config = {}) {
      this.config = {
        timeout: 30000,
        maxRetries: 3,
        retryDelay: 1000,
        rateLimit: 100,
        ...config
      };

      this.httpClient = new HttpClient({
        baseUrl: this.config.baseUrl,
        timeout: this.config.timeout,
        headers: this.buildHeaders()
      });

      this.rateLimiter = new RateLimiter(this.config.rateLimit);
      this.streamingConnections = new Map();

      // Expose interceptors
      this.interceptors = {
        request: {
          use: (fn) => this.httpClient.interceptors.request.push(fn)
        },
        response: {
          use: (successFn, errorFn) => {
            if (errorFn) {
              this.httpClient.interceptors.response.push((res, err) => {
                if (err) return errorFn(err);
                return successFn(res);
              });
            } else {
              this.httpClient.interceptors.response.push(successFn);
            }
          }
        }
      };
    }

    buildHeaders() {
      const headers = { ...this.config.headers };
      
      if (this.config.apiKey) {
        headers['X-API-Key'] = this.config.apiKey;
      } else if (this.config.jwtToken) {
        headers['Authorization'] = `Bearer ${this.config.jwtToken}`;
      }
      
      return headers;
    }

    async makeRequest(method, url, data = null, options = {}) {
      await this.rateLimiter.checkLimit();
      
      const config = {
        method,
        url,
        data,
        headers: this.buildHeaders(),
        ...options
      };

      let lastError;
      
      for (let attempt = 0; attempt <= this.config.maxRetries; attempt++) {
        try {
          const response = await this.httpClient.request(config);
          
          if (!response.ok) {
            await this.handleHttpError(response);
          }
          
          const result = await response.json();
          return result;
        } catch (error) {
          lastError = error;
          
          if (attempt < this.config.maxRetries && this.shouldRetry(error)) {
            await this.sleep(this.config.retryDelay * Math.pow(2, attempt));
            continue;
          }
          
          break;
        }
      }
      
      throw this.transformError(lastError);
    }

    async handleHttpError(response) {
      const errorData = await response.json().catch(() => ({}));
      
      switch (response.status) {
        case 401:
          throw new AuthenticationError(errorData.detail || 'Authentication failed');
        case 422:
          throw new ValidationError(errorData.detail || 'Validation failed', errorData);
        case 429:
          const retryAfter = response.headers['retry-after'];
          throw new RateLimitError(errorData.detail || 'Rate limit exceeded', retryAfter);
        case 500:
          throw new ProcessingError(errorData.detail || 'Server error');
        default:
          throw new NCSError(
            errorData.detail || `HTTP ${response.status}: ${response.statusText}`,
            response.status
          );
      }
    }

    shouldRetry(error) {
      return error instanceof ConnectionError || 
             (error instanceof NCSError && error.errorCode >= 500);
    }

    transformError(error) {
      if (error instanceof NCSError) {
        return error;
      }
      
      if (error.name === 'AbortError' || error.message.includes('timeout')) {
        return new ConnectionError('Request timeout');
      }
      
      if (error.message.includes('network') || error.message.includes('fetch')) {
        return new ConnectionError(error.message);
      }
      
      return new NCSError(error.message);
    }

    sleep(ms) {
      return new Promise(resolve => setTimeout(resolve, ms));
    }

    // Authentication
    async authenticate(username, password) {
      const formData = new URLSearchParams();
      formData.append('username', username);
      formData.append('password', password);

      try {
        const response = await this.httpClient.request({
          method: 'POST',
          url: '/auth/login',
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
          },
          body: formData.toString()
        });

        if (!response.ok) {
          throw new AuthenticationError('Invalid credentials');
        }

        const result = await response.json();
        this.config.jwtToken = result.access_token;
        
        // Update headers
        this.httpClient.headers = this.buildHeaders();
        
        return result.access_token;
      } catch (error) {
        throw this.transformError(error);
      }
    }

    // Core API methods
    async processPoints(points, options = {}) {
      const data = {
        points,
        batch_mode: options.batchMode || false,
        timeout: options.timeout || this.config.timeout
      };

      return this.makeRequest('POST', '/api/v1/process_points', data);
    }

    async healthCheck() {
      return this.makeRequest('GET', '/api/v1/health');
    }

    async getClustersSummary(filters = {}) {
      const params = new URLSearchParams();
      
      if (filters.minSize) params.append('min_size', filters.minSize);
      if (filters.qualityThreshold) params.append('quality_threshold', filters.qualityThreshold);
      if (filters.clusterId) params.append('cluster_id', filters.clusterId);
      
      const url = `/api/v1/clusters_summary${params.toString() ? `?${params}` : ''}`;
      return this.makeRequest('GET', url);
    }

    async getAlgorithmStatus() {
      return this.makeRequest('GET', '/api/v1/algorithm_status');
    }

    // Batch processing utilities
    async processBatchesConcurrent(batches, options = {}) {
      const maxConcurrent = options.maxConcurrent || 3;
      const results = [];
      
      for (let i = 0; i < batches.length; i += maxConcurrent) {
        const chunk = batches.slice(i, i + maxConcurrent);
        const promises = chunk.map(batch => this.processPoints(batch));
        const chunkResults = await Promise.allSettled(promises);
        
        chunkResults.forEach((result, index) => {
          if (result.status === 'fulfilled') {
            results.push(result.value);
          } else {
            console.error(`Batch ${i + index} failed:`, result.reason);
            results.push(null);
          }
        });
      }
      
      return results.filter(result => result !== null);
    }

    // Streaming methods
    async startStreaming(options = {}) {
      const wsUrl = this.config.baseUrl.replace(/^http/, 'ws') + '/ws/stream';
      const url = new URL(wsUrl);
      
      if (this.config.apiKey) {
        url.searchParams.set('api_key', this.config.apiKey);
      } else if (this.config.jwtToken) {
        url.searchParams.set('token', this.config.jwtToken);
      }

      const connection = new StreamingConnection(url.toString(), options);
      await connection.connect();
      
      const connectionId = options.connectionId || 'default';
      this.streamingConnections.set(connectionId, connection);
      
      return connection;
    }

    async stopStreaming(connectionId = 'default') {
      const connection = this.streamingConnections.get(connectionId);
      if (connection) {
        await connection.stop();
        this.streamingConnections.delete(connectionId);
      }
    }

    async stopAllStreaming() {
      const connections = Array.from(this.streamingConnections.values());
      await Promise.all(connections.map(conn => conn.stop()));
      this.streamingConnections.clear();
    }
  }

  // Factory function for creating clients
  function createClient(config) {
    return new NCSClient(config);
  }

  // Environment-specific client creation (Node.js only)
  function createClientFromEnv(overrides = {}) {
    if (!isNode) {
      throw new Error('createClientFromEnv is only available in Node.js environments');
    }

    const config = {
      baseUrl: process.env.NCS_API_URL,
      apiKey: process.env.NCS_API_KEY,
      jwtToken: process.env.NCS_JWT_TOKEN,
      timeout: parseInt(process.env.NCS_TIMEOUT || '30000'),
      maxRetries: parseInt(process.env.NCS_MAX_RETRIES || '3'),
      retryDelay: parseInt(process.env.NCS_RETRY_DELAY || '1000'),
      rateLimit: parseInt(process.env.NCS_RATE_LIMIT || '100'),
      ...overrides
    };

    if (!config.baseUrl) {
      throw new Error('NCS_API_URL environment variable is required');
    }

    if (!config.apiKey && !config.jwtToken) {
      throw new Error('Either NCS_API_KEY or NCS_JWT_TOKEN environment variable is required');
    }

    return new NCSClient(config);
  }

  // Export everything for different module systems
  exports.NCSClient = NCSClient;
  exports.NCSError = NCSError;
  exports.AuthenticationError = AuthenticationError;
  exports.RateLimitError = RateLimitError;
  exports.ValidationError = ValidationError;
  exports.ProcessingError = ProcessingError;
  exports.ConnectionError = ConnectionError;
  exports.StreamingConnection = StreamingConnection;
  exports.createClient = createClient;
  
  if (isNode) {
    exports.createClientFromEnv = createClientFromEnv;
  }

  // Default export for ES modules
  exports.default = NCSClient;

})));