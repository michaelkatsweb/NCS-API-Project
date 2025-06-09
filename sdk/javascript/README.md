# NeuroCluster Streamer JavaScript SDK

Official JavaScript/TypeScript client library for the NeuroCluster Streamer API.

## Features

- **Modern ES6+/TypeScript** support with full type definitions
- **Promise-based API** with async/await support  
- **WebSocket streaming** for real-time data processing
- **Automatic retry logic** with configurable backoff
- **Request/response interceptors** for custom handling
- **Browser and Node.js** compatible
- **Zero dependencies** (except for optional WebSocket support)
- **Comprehensive error handling** with typed exceptions
- **Built-in rate limiting** and request queuing

## Installation

### NPM
```bash
npm install ncs-javascript-sdk
```

### Yarn
```bash
yarn add ncs-javascript-sdk
```

### CDN (Browser)
```html
<script src="https://unpkg.com/ncs-javascript-sdk@latest/dist/ncs-client.min.js"></script>
```

## Quick Start

### ES6 Modules / TypeScript
```typescript
import { NCSClient } from 'ncs-javascript-sdk';

const client = new NCSClient({
  baseUrl: 'https://api.yourdomain.com',
  apiKey: 'your-api-key'
});

// Process data points
const points = [[1, 2, 3], [1.1, 2.1, 3.1], [5, 6, 7]];
const result = await client.processPoints(points);

console.log(`Found ${result.clusters.length} clusters`);
```

### CommonJS (Node.js)
```javascript
const { NCSClient } = require('ncs-javascript-sdk');

const client = new NCSClient({
  baseUrl: 'https://api.yourdomain.com', 
  apiKey: 'your-api-key'
});

client.processPoints([[1, 2, 3]])
  .then(result => console.log(result))
  .catch(error => console.error(error));
```

### Browser Global
```html
<script src="https://unpkg.com/ncs-javascript-sdk@latest/dist/ncs-client.min.js"></script>
<script>
  const client = new NCS.NCSClient({
    baseUrl: 'https://api.yourdomain.com',
    apiKey: 'your-api-key'
  });
  
  client.processPoints([[1, 2, 3]])
    .then(result => console.log(result));
</script>
```

## API Reference

### Client Configuration

```typescript
interface NCSClientConfig {
  baseUrl: string;
  apiKey?: string;
  jwtToken?: string;
  timeout?: number;
  maxRetries?: number;
  retryDelay?: number;
  rateLimit?: number;
  headers?: Record<string, string>;
}

const client = new NCSClient({
  baseUrl: 'https://api.yourdomain.com',
  apiKey: 'your-api-key',
  timeout: 30000,
  maxRetries: 3,
  retryDelay: 1000,
  rateLimit: 100 // requests per minute
});
```

### Authentication

```typescript
// API Key authentication
const client = new NCSClient({
  baseUrl: 'https://api.yourdomain.com',
  apiKey: 'your-api-key'
});

// JWT authentication
const token = await client.authenticate('username', 'password');
console.log('JWT token:', token);

// Or initialize with JWT
const client = new NCSClient({
  baseUrl: 'https://api.yourdomain.com',
  jwtToken: 'your-jwt-token'
});
```

### Processing Data Points

```typescript
const points: Point[] = [
  [1.0, 2.0, 3.0],
  [1.1, 2.1, 3.1], 
  [5.0, 6.0, 7.0]
];

const result = await client.processPoints(points, {
  timeout: 30000,
  batchMode: false
});

console.log('Clusters:', result.clusters);
console.log('Outliers:', result.outliers);
console.log('Quality:', result.algorithmQuality);
```

### Streaming Data

```typescript
// Start streaming connection
const stream = await client.startStreaming({
  onMessage: (data) => {
    console.log('Cluster update:', data);
  },
  onError: (error) => {
    console.error('Stream error:', error);
  },
  onClose: () => {
    console.log('Stream closed');
  }
});

// Send points to stream
await stream.sendPoint([1, 2, 3]);
await stream.sendBatch([[1, 2, 3], [4, 5, 6]]);

// Stop streaming
await stream.stop();
```

### Batch Processing

```typescript
const batches = [
  [[1, 2, 3], [1.1, 2.1, 3.1]], 
  [[5, 6, 7], [5.1, 6.1, 7.1]],
  [[10, 11, 12], [10.1, 11.1, 12.1]]
];

// Process batches concurrently
const results = await client.processBatchesConcurrent(batches, {
  maxConcurrent: 3
});

console.log(`Processed ${results.length} batches`);
```

### Algorithm Status

```typescript
const status = await client.getAlgorithmStatus();

console.log('Ready:', status.isReady);
console.log('Active clusters:', status.activeClusters);
console.log('Quality:', status.clusteringQuality);
console.log('Memory usage:', status.memoryUsageMb);
```

### Health Check

```typescript
const health = await client.healthCheck();

console.log('Status:', health.status);
console.log('Version:', health.version);
console.log('Uptime:', health.uptimeSeconds);
```

## Error Handling

```typescript
import { 
  NCSClient, 
  NCSError, 
  AuthenticationError,
  RateLimitError,
  ValidationError 
} from 'ncs-javascript-sdk';

try {
  const result = await client.processPoints(points);
} catch (error) {
  if (error instanceof AuthenticationError) {
    console.error('Authentication failed:', error.message);
  } else if (error instanceof RateLimitError) {
    console.error('Rate limited, retry after:', error.retryAfter);
  } else if (error instanceof ValidationError) {
    console.error('Validation error:', error.message);
  } else if (error instanceof NCSError) {
    console.error('API error:', error.message, error.statusCode);
  } else {
    console.error('Unexpected error:', error);
  }
}
```

## Configuration Examples

### Environment-based Configuration

```typescript
const client = new NCSClient({
  baseUrl: process.env.NCS_API_URL || 'https://api.yourdomain.com',
  apiKey: process.env.NCS_API_KEY,
  timeout: parseInt(process.env.NCS_TIMEOUT || '30000'),
  maxRetries: parseInt(process.env.NCS_MAX_RETRIES || '3')
});
```

### Custom Headers and Interceptors

```typescript
const client = new NCSClient({
  baseUrl: 'https://api.yourdomain.com',
  apiKey: 'your-api-key',
  headers: {
    'X-Custom-Header': 'value'
  }
});

// Request interceptor
client.interceptors.request.use(config => {
  console.log('Making request:', config.url);
  return config;
});

// Response interceptor  
client.interceptors.response.use(
  response => response,
  error => {
    console.error('Request failed:', error);
    return Promise.reject(error);
  }
);
```

## TypeScript Support

Full TypeScript definitions are included:

```typescript
import { 
  NCSClient,
  Point,
  Cluster,
  ProcessingResult,
  AlgorithmStatus,
  HealthStatus,
  NCSClientConfig 
} from 'ncs-javascript-sdk';

const client: NCSClient = new NCSClient({
  baseUrl: 'https://api.yourdomain.com',
  apiKey: 'your-api-key'
});

const points: Point[] = [[1, 2, 3]];
const result: ProcessingResult = await client.processPoints(points);
```

## Browser Support

- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+
- IE 11+ (with polyfills)

## Node.js Support

- Node.js 14+
- Works with CommonJS and ES modules

## Examples

See the `examples/` directory for complete working examples:

- `basic-usage.js` - Simple point processing
- `streaming.js` - Real-time streaming
- `batch-processing.js` - Large dataset processing
- `error-handling.js` - Comprehensive error handling
- `typescript-example.ts` - TypeScript usage

## Testing

```bash
# Run tests
npm test

# Run tests with coverage
npm run test:coverage

# Run tests in watch mode
npm run test:watch
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: https://docs.ncs-api.com
- **Issues**: https://github.com/your-org/ncs-api/issues
- **Email**: sdk@yourdomain.com

## Changelog

### v1.0.0
- Initial release
- TypeScript support
- Promise-based API
- WebSocket streaming
- Comprehensive error handling
- Browser and Node.js compatibility