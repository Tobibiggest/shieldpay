# Deploying to Railway

This repo is a monorepo with three independently deployable apps. Railway's
model fits that directly: **one Railway project, three services**, each
pointed at this same GitHub repo with a different **Root Directory**. Once
connected, Railway's native GitHub integration auto-redeploys each service
on every push to `master` -- there's no GitHub Actions workflow involved,
and none is needed for this setup.

Each service directory already has a `railway.json` that tells Railway how
to build and start it, so the only thing you configure by hand is the Root
Directory + environment variables per service.

## 1. Backend -- `AI_model_server_Flask/`

Create a Railway service, set **Root Directory** to `AI_model_server_Flask`.

Environment variables (Railway dashboard -> service -> Variables):

| Variable | Required | Notes |
|---|---|---|
| `PAYSTACK_SECRET_KEY` | for `/verify-account` | |
| `MONO_SECRET_KEY` | for `/mono/*` endpoints | |
| `ANTHROPIC_API_KEY` | for PDF statement upload | Needs a funded Anthropic API credit balance -- separate from a Claude.ai/Pro subscription. |
| `FRAUD_MODEL_DIR` | optional | Only relevant if you separately provide the trained ensemble bundle (see note below). |

Deploy this service **first** -- the frontend and landing page both need its
public Railway URL for their own env vars below.

**Known limitation:** `fraud_detection/artifacts/` (the trained graph
ensemble used by `/predict/v2`) is gitignored and is *not* in this repo, so
it won't exist in Railway's build. `/predict/v2` will return `503` in
production -- this is the same graceful-degradation behavior it already has
locally when the model isn't found, not a bug introduced by this deploy. The
legacy `/predict` endpoint (RandomForest) works fine regardless, since that
model file (`best_rf_model (1).pkl`) *is* tracked in git. If you want
`/predict/v2` live in production, you'll need to get the trained artifacts
onto the Railway instance yourself (e.g. a Railway volume), which isn't
something that can be set up from the repo alone.

## 2. Frontend -- `fraudAI_Frontend_React/`

Create a second service, **Root Directory** = `fraudAI_Frontend_React`.

Environment variables (all are **build-time** -- Vite bakes `VITE_*` vars
into the static bundle, so these must be set *before* the build runs, not
just at runtime):

| Variable | Required |
|---|---|
| `VITE_API_BASE_URL` | Yes -- set to the backend service's public Railway URL (e.g. `https://<backend>.up.railway.app`). Without this it falls back to `http://127.0.0.1:5000`, which will not work once deployed. |
| `VITE_FIREBASE_API_KEY` | Yes |
| `VITE_FIREBASE_AUTH_DOMAIN` | Yes |
| `VITE_FIREBASE_PROJECT_ID` | Yes |
| `VITE_FIREBASE_STORAGE_BUCKET` | Yes |
| `VITE_FIREBASE_MESSAGING_SENDER_ID` | Yes |
| `VITE_FIREBASE_APP_ID` | Yes |
| `VITE_FIREBASE_MEASUREMENT_ID` | Optional |

(Values for the `VITE_FIREBASE_*` keys are the same ones already in your
local `.env` for this app.)

## 3. Landing page -- `shieldpay-landing page/`

Create a third service, **Root Directory** = `shieldpay-landing page`.

This app is built with [TanStack Start](https://tanstack.com/start) and
defaults to targeting **Cloudflare Workers** (its `vite.config.ts` uses a
shared Lovable build config with `cloudflare-module` as the default preset).
That doesn't run on Railway's plain Node containers, so its `railway.json`
build command explicitly overrides this via `NITRO_PRESET=node-server`,
which produces a standard Node server at `.output/server/index.mjs` instead
-- this has been build-tested locally and confirmed to boot and serve
requests correctly under that preset.

Environment variables (same pattern as the frontend -- `VITE_*` vars are
build-time):

| Variable | Required |
|---|---|
| `VITE_FIREBASE_API_KEY` | Yes |
| `VITE_FIREBASE_AUTH_DOMAIN` | Yes |
| `VITE_FIREBASE_PROJECT_ID` | Yes |
| `VITE_FIREBASE_STORAGE_BUCKET` | Yes |
| `VITE_FIREBASE_MESSAGING_SENDER_ID` | Yes |
| `VITE_FIREBASE_APP_ID` | Yes |
| `VITE_FIREBASE_MEASUREMENT_ID` | Optional |
| `PAYSTACK_SECRET_KEY` | If the landing page's own Paystack flow is used |
| `VITE_MONO_PUBLIC_KEY` | If the landing page's own Mono Connect flow is used |

(Same source: this app's local `.env`, which is gitignored and was never
committed.)

## After connecting all three

1. In Railway's dashboard, connect this GitHub repo once per service (three
   times total), setting the Root Directory as above. Railway auto-detects
   each `railway.json` and uses its build/start commands.
2. Push to `master` -- Railway auto-deploys all three services on every
   push. No manual redeploy step, no CI workflow file.
3. Verify: hit the backend's `/` route (should return `Welcome to the
   ShieldPay API!`), load the frontend and confirm statement upload / login
   work against the live backend URL, and load the landing page.
