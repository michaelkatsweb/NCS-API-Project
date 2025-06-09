#!/usr/bin/env node

/**
 * File: /NeuroCluster_website/deploy.js
 * NeuroCluster Website Deployment Script
 * 
 * Simple build and deployment script for the NeuroCluster Streamer website.
 * Optimizes assets, generates sitemap, and prepares for production deployment.
 * 
 * Usage:
 *   node deploy.js [options]
 * 
 * Options:
 *   --build     Build for production (minify, optimize)
 *   --serve     Start local development server
 *   --sitemap   Generate sitemap.xml
 *   --validate  Validate HTML and check links
 *   --help      Show this help message
 * 
 * @author NeuroCluster Team
 * @version 1.0.0
 */

const fs = require('fs');
const path = require('path');
const http = require('http');
const url = require('url');

// Configuration
const CONFIG = {
    sourceDir: '.',
    buildDir: './dist',
    port: 8000,
    baseUrl: 'https://neurocluster.research.com',
    excludeFromSitemap: [
        '/node_modules/',
        '/dist/',
        '/.git/',
        '/config/',
        '/components/',
        'deploy.js',
        'package.json',
        '.gitignore'
    ]
};

// ANSI color codes for console output
const colors = {
    reset: '\x1b[0m',
    bright: '\x1b[1m',
    red: '\x1b[31m',
    green: '\x1b[32m',
    yellow: '\x1b[33m',
    blue: '\x1b[34m',
    magenta: '\x1b[35m',
    cyan: '\x1b[36m',
    white: '\x1b[37m'
};

// Utility functions
const log = {
    info: (msg) => console.log(`${colors.blue}ℹ${colors.reset} ${msg}`),
    success: (msg) => console.log(`${colors.green}✓${colors.reset} ${msg}`),
    warning: (msg) => console.log(`${colors.yellow}⚠${colors.reset} ${msg}`),
    error: (msg) => console.log(`${colors.red}✗${colors.reset} ${msg}`),
    header: (msg) => console.log(`\n${colors.bright}${colors.cyan}${msg}${colors.reset}\n`)
};

/**
 * Display help information
 */
function showHelp() {
    console.log(`
${colors.bright}NeuroCluster Website Deployment Script${colors.reset}

${colors.cyan}Usage:${colors.reset}
  node deploy.js [options]

${colors.cyan}Options:${colors.reset}
  --build     Build for production (minify, optimize)
  --serve     Start local development server
  --sitemap   Generate sitemap.xml
  --validate  Validate HTML and check links
  --help      Show this help message

${colors.cyan}Examples:${colors.reset}
  node deploy.js --serve          Start development server
  node deploy.js --build          Build for production
  node deploy.js --sitemap        Generate sitemap
  node deploy.js --build --sitemap   Build and generate sitemap

${colors.cyan}Development Server:${colors.reset}
  The development server will start on http://localhost:${CONFIG.port}
  and serve files from the current directory with proper MIME types.

${colors.cyan}Production Build:${colors.reset}
  Creates an optimized version in ./dist/ with:
  - Minified CSS and JavaScript
  - Optimized HTML
  - Generated sitemap.xml
  - Proper cache headers configuration
    `);
}

/**
 * Check if a file exists
 */
function fileExists(filePath) {
    try {
        return fs.statSync(filePath).isFile();
    } catch (e) {
        return false;
    }
}

/**
 * Create directory if it doesn't exist
 */
function ensureDir(dirPath) {
    if (!fs.existsSync(dirPath)) {
        fs.mkdirSync(dirPath, { recursive: true });
        log.info(`Created directory: ${dirPath}`);
    }
}

/**
 * Copy file with logging
 */
function copyFile(src, dest) {
    ensureDir(path.dirname(dest));
    fs.copyFileSync(src, dest);
    log.info(`Copied: ${src} → ${dest}`);
}

/**
 * Read file with error handling
 */
function readFile(filePath) {
    try {
        return fs.readFileSync(filePath, 'utf8');
    } catch (e) {
        log.error(`Failed to read file: ${filePath}`);
        return null;
    }
}

/**
 * Write file with error handling
 */
function writeFile(filePath, content) {
    try {
        ensureDir(path.dirname(filePath));
        fs.writeFileSync(filePath, content, 'utf8');
        log.success(`Written: ${filePath}`);
        return true;
    } catch (e) {
        log.error(`Failed to write file: ${filePath}`);
        return false;
    }
}

/**
 * Get MIME type for file extension
 */
function getMimeType(filePath) {
    const ext = path.extname(filePath).toLowerCase();
    const mimeTypes = {
        '.html': 'text/html',
        '.css': 'text/css',
        '.js': 'application/javascript',
        '.json': 'application/json',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.svg': 'image/svg+xml',
        '.ico': 'image/x-icon',
        '.pdf': 'application/pdf',
        '.woff': 'font/woff',
        '.woff2': 'font/woff2',
        '.ttf': 'font/ttf',
        '.eot': 'application/vnd.ms-fontobject'
    };
    return mimeTypes[ext] || 'text/plain';
}

/**
 * Start development server
 */
function startDevServer() {
    log.header('Starting Development Server');
    
    const server = http.createServer((req, res) => {
        let filePath = path.join(CONFIG.sourceDir, url.parse(req.url).pathname);
        
        // Default to index.html for directory requests
        if (fs.existsSync(filePath) && fs.statSync(filePath).isDirectory()) {
            filePath = path.join(filePath, 'index.html');
        }
        
        // Handle 404 for missing files
        if (!fileExists(filePath)) {
            res.writeHead(404, { 'Content-Type': 'text/html' });
            res.end(`
                <html>
                <head><title>404 - Not Found</title></head>
                <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                    <h1>404 - Page Not Found</h1>
                    <p>The requested page could not be found.</p>
                    <a href="/">Return to Homepage</a>
                </body>
                </html>
            `);
            return;
        }
        
        // Set appropriate headers
        const mimeType = getMimeType(filePath);
        res.writeHead(200, {
            'Content-Type': mimeType,
            'Cache-Control': 'no-cache',
            'Access-Control-Allow-Origin': '*'
        });
        
        // Stream file content
        const fileStream = fs.createReadStream(filePath);
        fileStream.pipe(res);
        
        fileStream.on('error', (err) => {
            res.writeHead(500);
            res.end('Internal Server Error');
            log.error(`Error serving ${filePath}: ${err.message}`);
        });
        
        // Log request
        console.log(`${colors.green}${req.method}${colors.reset} ${req.url} → ${filePath}`);
    });
    
    server.listen(CONFIG.port, () => {
        log.success(`Server running at http://localhost:${CONFIG.port}`);
        log.info('Press Ctrl+C to stop the server');
        log.info('');
        log.info('Available pages:');
        log.info('  http://localhost:' + CONFIG.port + '/');
        log.info('  http://localhost:' + CONFIG.port + '/pages/research.html');
        log.info('  http://localhost:' + CONFIG.port + '/pages/documentation.html');
        log.info('  http://localhost:' + CONFIG.port + '/pages/demo.html');
        log.info('  http://localhost:' + CONFIG.port + '/pages/contact.html');
        log.info('  http://localhost:' + CONFIG.port + '/blog/');
    });
    
    // Graceful shutdown
    process.on('SIGINT', () => {
        log.info('\nShutting down server...');
        server.close(() => {
            log.success('Server stopped');
            process.exit(0);
        });
    });
}

/**
 * Minify CSS content
 */
function minifyCSS(css) {
    return css
        .replace(/\/\*[\s\S]*?\*\//g, '') // Remove comments
        .replace(/\s+/g, ' ') // Collapse whitespace
        .replace(/;\s*}/g, '}') // Remove last semicolon
        .replace(/\s*{\s*/g, '{') // Clean braces
        .replace(/\s*}\s*/g, '}')
        .replace(/\s*;\s*/g, ';') // Clean semicolons
        .replace(/\s*,\s*/g, ',') // Clean commas
        .replace(/\s*:\s*/g, ':') // Clean colons
        .trim();
}

/**
 * Minify JavaScript content
 */
function minifyJS(js) {
    return js
        .replace(/\/\*[\s\S]*?\*\//g, '') // Remove block comments
        .replace(/\/\/.*$/gm, '') // Remove line comments
        .replace(/\s+/g, ' ') // Collapse whitespace
        .replace(/\s*{\s*/g, '{') // Clean braces
        .replace(/\s*}\s*/g, '}')
        .replace(/\s*;\s*/g, ';') // Clean semicolons
        .replace(/\s*,\s*/g, ',') // Clean commas
        .trim();
}

/**
 * Optimize HTML content
 */
function optimizeHTML(html) {
    return html
        .replace(/<!--[\s\S]*?-->/g, '') // Remove comments
        .replace(/\s+/g, ' ') // Collapse whitespace
        .replace(/>\s+</g, '><') // Remove whitespace between tags
        .trim();
}

/**
 * Get all HTML files recursively
 */
function getHTMLFiles(dir, fileList = []) {
    const files = fs.readdirSync(dir);
    
    files.forEach(file => {
        const filePath = path.join(dir, file);
        const stat = fs.statSync(filePath);
        
        if (stat.isDirectory() && !CONFIG.excludeFromSitemap.some(ex => filePath.includes(ex))) {
            getHTMLFiles(filePath, fileList);
        } else if (file.endsWith('.html')) {
            fileList.push(filePath);
        }
    });
    
    return fileList;
}

/**
 * Generate sitemap.xml
 */
function generateSitemap() {
    log.header('Generating Sitemap');
    
    const htmlFiles = getHTMLFiles(CONFIG.sourceDir);
    const urls = [];
    
    htmlFiles.forEach(filePath => {
        // Convert file path to URL
        let url = filePath.replace(CONFIG.sourceDir, '').replace(/\\/g, '/');
        if (url.startsWith('/')) url = url.substring(1);
        if (url === 'index.html') url = '';
        
        // Skip excluded paths
        if (CONFIG.excludeFromSitemap.some(ex => url.includes(ex.substring(1)))) {
            return;
        }
        
        const fullUrl = CONFIG.baseUrl + '/' + url;
        const lastmod = fs.statSync(filePath).mtime.toISOString().split('T')[0];
        
        urls.push({
            url: fullUrl,
            lastmod: lastmod,
            priority: url === '' ? '1.0' : url.includes('pages/') ? '0.8' : '0.6'
        });
    });
    
    // Generate XML
    const sitemap = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${urls.map(item => `  <url>
    <loc>${item.url}</loc>
    <lastmod>${item.lastmod}</lastmod>
    <priority>${item.priority}</priority>
  </url>`).join('\n')}
</urlset>`;
    
    const sitemapPath = path.join(CONFIG.buildDir || CONFIG.sourceDir, 'sitemap.xml');
    if (writeFile(sitemapPath, sitemap)) {
        log.success(`Generated sitemap with ${urls.length} URLs`);
    }
}

/**
 * Build for production
 */
function buildProduction() {
    log.header('Building for Production');
    
    // Clean build directory
    if (fs.existsSync(CONFIG.buildDir)) {
        log.info('Cleaning build directory...');
        fs.rmSync(CONFIG.buildDir, { recursive: true, force: true });
    }
    
    ensureDir(CONFIG.buildDir);
    
    // Copy and optimize files
    const copyFiles = (srcDir, destDir) => {
        const files = fs.readdirSync(srcDir);
        
        files.forEach(file => {
            const srcPath = path.join(srcDir, file);
            const destPath = path.join(destDir, file);
            const stat = fs.statSync(srcPath);
            
            if (stat.isDirectory()) {
                // Skip excluded directories
                if (CONFIG.excludeFromSitemap.some(ex => srcPath.includes(ex))) {
                    return;
                }
                
                ensureDir(destPath);
                copyFiles(srcPath, destPath);
            } else {
                const ext = path.extname(file).toLowerCase();
                const content = readFile(srcPath);
                
                if (!content) return;
                
                let optimizedContent = content;
                
                // Optimize based on file type
                if (ext === '.css') {
                    optimizedContent = minifyCSS(content);
                    log.info(`Minified CSS: ${file}`);
                } else if (ext === '.js' && !file.includes('.min.')) {
                    optimizedContent = minifyJS(content);
                    log.info(`Minified JS: ${file}`);
                } else if (ext === '.html') {
                    optimizedContent = optimizeHTML(content);
                    log.info(`Optimized HTML: ${file}`);
                }
                
                writeFile(destPath, optimizedContent);
            }
        });
    };
    
    copyFiles(CONFIG.sourceDir, CONFIG.buildDir);
    
    // Generate sitemap
    generateSitemap();
    
    // Create deployment configuration
    const deployConfig = {
        timestamp: new Date().toISOString(),
        version: '1.0.0',
        files: fs.readdirSync(CONFIG.buildDir).length,
        cacheControl: {
            html: 'public, max-age=3600',
            css: 'public, max-age=31536000',
            js: 'public, max-age=31536000',
            images: 'public, max-age=31536000',
            fonts: 'public, max-age=31536000'
        }
    };
    
    writeFile(path.join(CONFIG.buildDir, 'deploy-config.json'), JSON.stringify(deployConfig, null, 2));
    
    log.success('Production build completed!');
    log.info(`Build output: ${CONFIG.buildDir}`);
    log.info('Ready for deployment to static hosting providers');
}

/**
 * Validate HTML and check links
 */
function validateProject() {
    log.header('Validating Project');
    
    const htmlFiles = getHTMLFiles(CONFIG.sourceDir);
    let issues = 0;
    
    htmlFiles.forEach(filePath => {
        const content = readFile(filePath);
        if (!content) return;
        
        // Basic HTML validation
        if (!content.includes('<!DOCTYPE html>')) {
            log.warning(`${filePath}: Missing DOCTYPE declaration`);
            issues++;
        }
        
        if (!content.includes('<meta charset=')) {
            log.warning(`${filePath}: Missing charset declaration`);
            issues++;
        }
        
        if (!content.includes('<title>')) {
            log.warning(`${filePath}: Missing title tag`);
            issues++;
        }
        
        // Check for broken internal links
        const linkRegex = /href=["']([^"']+)["']/g;
        let match;
        
        while ((match = linkRegex.exec(content)) !== null) {
            const link = match[1];
            
            // Skip external links
            if (link.startsWith('http') || link.startsWith('mailto:') || link.startsWith('#')) {
                continue;
            }
            
            // Resolve relative path
            const linkPath = path.resolve(path.dirname(filePath), link);
            
            if (!fileExists(linkPath)) {
                log.warning(`${filePath}: Broken link to ${link}`);
                issues++;
            }
        }
    });
    
    if (issues === 0) {
        log.success('Validation completed - no issues found!');
    } else {
        log.warning(`Validation completed - ${issues} issues found`);
    }
}

/**
 * Main execution
 */
function main() {
    const args = process.argv.slice(2);
    
    if (args.length === 0 || args.includes('--help')) {
        showHelp();
        return;
    }
    
    log.header('NeuroCluster Website Deployment');
    
    // Check if we're in the right directory
    if (!fileExists('index.html')) {
        log.error('index.html not found. Please run this script from the website root directory.');
        process.exit(1);
    }
    
    // Execute based on arguments
    if (args.includes('--serve')) {
        startDevServer();
    } else {
        if (args.includes('--validate')) {
            validateProject();
        }
        
        if (args.includes('--build')) {
            buildProduction();
        } else if (args.includes('--sitemap')) {
            generateSitemap();
        }
        
        if (!args.includes('--validate') && !args.includes('--build') && !args.includes('--sitemap')) {
            log.warning('No action specified. Use --help for usage information.');
        }
    }
}

// Handle uncaught exceptions
process.on('uncaughtException', (err) => {
    log.error(`Uncaught exception: ${err.message}`);
    process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
    log.error(`Unhandled rejection at ${promise}: ${reason}`);
    process.exit(1);
});

// Run the script
if (require.main === module) {
    main();
}

module.exports = {
    startDevServer,
    buildProduction,
    generateSitemap,
    validateProject
};