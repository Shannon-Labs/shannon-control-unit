# Cloudflare Security Notes

Updated: 2026-05-03

## Fixed in this repo

- `labs-frontend/client/public/_headers` adds Cloudflare Pages response headers for HSTS, content sniffing protection, frame denial, referrer policy, permissions policy, CSP, COOP, CORP, X-Permitted-Cross-Domain-Policies, and insecure-request upgrades.
- `labs-frontend/client/public/.well-known/security.txt` and `labs-frontend/client/public/security.txt` publish the security contact expected by common scanners.
- `labs-frontend/client/public/robots.txt` keeps normal indexing open while disallowing common AI crawlers at the site-policy layer.
- The Shannon Labs homepage now promotes DeepSeek-TUI as a highlighted project.

These changes apply to the Cloudflare Pages site for `shannonlabs.dev` after deployment.

## Verified live on shannonlabs.dev (2026-05-03)

- HSTS (max-age 1y), CSP, X-Frame-Options DENY, X-Content-Type-Options nosniff,
  Referrer-Policy strict-origin-when-cross-origin, Permissions-Policy locked down.
- COOP same-origin, CORP same-site, X-Permitted-Cross-Domain-Policies none.
- TLS healthy on the apex; security.txt resolves at /.well-known/security.txt.

## Open issues observed during external probe (2026-05-03)

- `api.shannonlabs.dev` returns HTTP 525 (origin TLS handshake failure).
- `klod.shannonlabs.dev` returns HTTP 530 (Cloudflare Tunnel error 1033).
- `voice.shannonlabs.dev` returns HTTP 530 (Cloudflare Tunnel error 1033).
- `klod.shannonlabs.io.shannonlabs.dev` should be removed if not intentional.
- `_domainconnect.shannonlabs.dev` from GoDaddy DomainConnect should be removed
  if you are no longer running domain connect flows.

Until the broken subdomains are fixed or removed, do **not** add
`includeSubDomains` or `preload` to the HSTS header on the apex — doing so
will harden the broken state into the user's browser cache.

## Known accepted trade-offs

- CSP keeps `style-src 'unsafe-inline'` because the SPA uses inline `style={...}`
  attributes throughout. Cloudflare Pages cannot inject per-request nonces,
  so nonce-based CSP would require a Worker.
- CSP keeps `img-src https:` so any future external image (OG previews, blog
  embeds) does not silently break. Tighten to `'self' data:` if you want to
  forbid any external image.
- `labs-frontend/server/` Express code is not deployed — Cloudflare Pages
  serves the static `dist/public` build. `npm audit` shows 8 issues (path-to-regexp,
  qs, body-parser via express); these only matter if the dev server is exposed
  publicly. Run `npm audit fix` if you ever start running the server.
- `labs-frontend/shared/schema.ts` defines a `users` table with a plaintext
  `password` column. Currently unused. If you bring up the server, switch to a
  hash column (argon2id) before storing real credentials.

## Cloudflare settings still requiring account permissions

The available Cloudflare connector could list zones and DNS records, but zone settings and Security Center endpoints returned authentication or authorization errors. These settings must be changed in the dashboard or with a token that can edit zone settings:

- Enable **Always Use HTTPS** for `shannonlabs.dev`.
- Enable zone HSTS after confirming every public subdomain serves HTTPS correctly. Do not enable HSTS `includeSubDomains` or preload while Cloudflare still reports missing TLS for tunnel-backed names.
- Verify SSL/TLS mode is not `Off`; use Full or Full (strict) where origins have valid certificates.
- Enable Bot Fight Mode, AI crawler controls, or AI Labyrinth where appropriate.
- Re-run Security Center after deployment and setting changes.

## Shannonlabs DNS records seen during triage

The Cloudflare account currently has proxied records for:

- `shannonlabs.dev` -> `shannon-labs.pages.dev`
- `www.shannonlabs.dev` -> `shannonlabs.dev`
- `api.shannonlabs.dev` -> IPv6 origin
- `klod.shannonlabs.dev` -> Cloudflare Tunnel
- `voice.shannonlabs.dev` -> Cloudflare Tunnel
- `klod.shannonlabs.io.shannonlabs.dev` -> Cloudflare Tunnel
- `_domainconnect.shannonlabs.dev` -> GoDaddy DomainConnect

If Security Center continues reporting missing TLS for tunnel-backed names, fix the service behind the tunnel or add Cloudflare edge/header rules for those hostnames. If `_domainconnect.shannonlabs.dev` is not needed, remove it or make it DNS-only.
