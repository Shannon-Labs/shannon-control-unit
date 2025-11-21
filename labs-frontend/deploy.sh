#!/bin/bash

# Shannon Labs Frontend Deployment Script
# Builds and deploys the frontend to Cloudflare Pages

set -e

echo "🏗️  Building frontend..."
npm run build

echo "🚀 Deploying to Cloudflare Pages..."
wrangler pages deploy dist/public --project-name=shannon-labs --branch=main

echo "✅ Deployment complete!"
echo "🌐 Live at: https://shannonlabs.dev"
echo "🔗 Preview: https://shannon-labs.pages.dev"
