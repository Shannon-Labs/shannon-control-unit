# Shannon Labs Frontend

Modern, full-stack web application for Shannon Labs, built with React, TypeScript, and Vite.

## 🚀 Quick Start

### Development
```bash
npm install
npm run dev
```

The app will be available at `http://localhost:5000`

### Production Build
```bash
npm run build
```

Builds the frontend to `dist/public/`

## 📦 Deployment

### Cloudflare Pages

The site is deployed to Cloudflare Pages:
- **Production**: https://shannonlabs.dev
- **Preview**: https://shannon-labs.pages.dev

#### Deploy Script
```bash
./deploy.sh
```

This will:
1. Build the production bundle
2. Deploy to Cloudflare Pages
3. Update the live site

#### Manual Deployment
```bash
npm run build
wrangler pages deploy dist/public --project-name=shannon-labs --branch=main
```

## 🛠️ Tech Stack

- **Framework**: React 19 + TypeScript
- **Build Tool**: Vite 7
- **Styling**: Tailwind CSS 4
- **UI Components**: Radix UI + shadcn/ui
- **Routing**: Wouter
- **State Management**: TanStack Query
- **Backend**: Express.js (for local development)
- **Database**: Drizzle ORM + Neon (PostgreSQL)

## 📁 Project Structure

```
labs-frontend/
├── client/           # Frontend React application
├── server/           # Express backend (dev only)
├── shared/           # Shared types and utilities
├── attached_assets/  # Static assets
├── dist/            # Build output (gitignored)
└── wrangler.toml    # Cloudflare configuration
```

## 🔧 Configuration

- **Vite Config**: `vite.config.ts`
- **TypeScript**: `tsconfig.json`
- **Tailwind**: `postcss.config.js`
- **Cloudflare**: `wrangler.toml`

## 📝 Available Scripts

- `npm run dev` - Start development server with hot reload
- `npm run dev:client` - Start Vite dev server only (port 5000)
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run check` - TypeScript type checking
- `npm run db:push` - Push database schema changes

## 🌐 Custom Domain Setup

The custom domain `shannonlabs.dev` is configured in Cloudflare Pages:

1. Go to Cloudflare Dashboard → Pages → shannon-labs
2. Navigate to "Custom domains" tab
3. Domain should already be configured and active

DNS is managed automatically by Cloudflare.

## 📄 License

MIT
