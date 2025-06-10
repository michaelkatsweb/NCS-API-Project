// File: docs/.vitepress/config.js
// Description: VitePress configuration for NCS API documentation
// Last updated: 2025-06-10 08:50:27
import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'NCS API Documentation',
  description: 'Official documentation for the NeuroCluster Streamer API',
  
  themeConfig: {
    nav: [
      { text: 'Home', link: '/' },
      { text: 'API Reference', link: '/api-reference' },
      { text: 'SDK Guide', link: '/sdk-guide' }
    ],

    sidebar: [
      {
        text: 'Getting Started',
        items: [
          { text: 'Introduction', link: '/' },
          { text: 'Quick Start', link: '/quickstart' }
        ]
      },
      {
        text: 'API Documentation',
        items: [
          { text: 'Authentication', link: '/api/authentication' },
          { text: 'Endpoints', link: '/api/endpoints' }
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/your-org/ncs-api' }
    ]
  }
})
