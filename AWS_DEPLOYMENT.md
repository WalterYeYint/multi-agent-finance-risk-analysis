# AWS Deployment Guide — Option B-lite-lite

> **TL;DR.** Keep the Flask backend on App Runner, run the worker on Fargate, store snapshots/filings in a single `t4g.micro` RDS Postgres (pgvector), serve the frontend from S3 + CloudFront. **~$28–38/mo idle**, with no Lambda port required.

This guide reflects the **current** code shape (post-`21216cc1`): a backend that only **reads or enqueues**, a **worker** that drains the queue and runs the chain + debate pipeline, and a Postgres + pgvector database that both processes share. The previous version of this file only deployed the backend container — it predates the worker and database, and was not actually runnable end-to-end.

For a comparison with the higher-ceiling "Option B-lite" (Lambda + Aurora + DynamoDB) from [`latest_plan.md`](latest_plan.md), see the [cost section](#cost-comparison-vs-option-b-lite) at the bottom.

---

## Architecture

```
                  CloudFront
                       │
        ┌──────────────┼──────────────┐
        │              │              │
   S3: frontend   App Runner     ── (no auth, no API Gateway)
                  Flask backend       
                       │              
                       │              ┌─────────────────────────────┐
                       └────POSTGRES──┤  Supabase  OR  RDS t4g.micro│
                                      │  (Postgres + pgvector)      │
                                      │  - snapshots                │
                                      │  - jobs                     │
                                      │  - filings, filing_chunks   │
                                      └─────────────────────────────┘
                                              ▲
                                              │ POSTGRES
                                              │
                                      ECS Fargate task
                                      `python -m src.worker`
                                        - claims jobs
                                        - runs chain + debate
                                        - writes snapshot
                                        - periodic stale-refresh
                                              │
                                              ▼
                                      OpenAI / Polygon / SEC EDGAR
                                      (outbound HTTPS only)
```

Two containers built from the same source tree (`Dockerfile.backend`, `Dockerfile.worker`), pushed to two ECR repos. The backend never runs the pipeline — it only `SELECT`s the snapshot if fresh, otherwise enqueues a job row and returns `202 {status, job_id}`. The Fargate worker is the only thing that calls the LLM.

---

## What's different from the old single-container plan

| Concern | Old plan | This plan |
|---|---|---|
| Postgres | _not mentioned_ | RDS `t4g.micro` + pgvector |
| Worker | _not mentioned_ | ECS Fargate (continuously running) |
| `Dockerfile.backend` referenced but missing | Yes | Now committed alongside this file |
| `Dockerfile.worker` | Missing | Committed alongside this file |
| Env vars enumerated | `OPENAI_API_KEY`, `FLASK_ENV` | Full list below |
| Cache-miss path | crashed | works |

---

## Pre-reqs

- AWS CLI v2 (`aws configure` with a deploy IAM user)
- Docker locally
- An AWS region (this guide uses `ap-southeast-2` — substitute yours)
- An OpenAI API key, optional Polygon key, a real `SEC_USER_AGENT` contact string

Set these once at the top of your shell session — every command below uses them:

```bash
export AWS_REGION=ap-southeast-2
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export PROJECT=finance-agents
```

---

## Step 1 — Database: pick one

The pipeline needs Postgres 14+ with `pgvector`. Two paths, picked on cost / ops trade-off:

| | **Option A — Supabase** | **Option B — RDS** |
|---|---|---|
| Cost | **$0** (free tier) or $25/mo (Pro) | ~$14/mo (`t4g.micro` + 20 GB gp3) |
| Setup time | ~5 min, point-and-click | ~30 min, VPC + SG + wait |
| Networking | Public endpoint, TLS | Private in your VPC |
| Connection latency from AWS | ~5–10 ms (same metro) | ~1–2 ms (same VPC) |
| pgvector | preinstalled, toggle in UI | install with `CREATE EXTENSION` |
| Backups | Daily, retained 7 d (Pro) | Configurable, paid storage |
| Free tier paused after | 1 week of inactivity | n/a |
| Best for | demos, portfolio, MVPs | production workloads, multi-region, compliance |

**My recommendation:** start with **Supabase** unless you already have a reason to need an in-VPC database. The free tier hosts the whole thing for $0/mo and the migration to RDS later is just changing one env var.

### Option A — Supabase

1. Create a free account at https://supabase.com and click **New project**.
2. Settings:
   - **Name**: `finance-agents`
   - **Database password**: generate a strong one — save it to your password manager.
   - **Region**: pick the one closest to your AWS region (e.g. AWS `ap-southeast-2` → Supabase **Sydney**, AWS `us-east-1` → Supabase **US East**). Cross-region adds 100+ ms per query — avoid it.
   - **Pricing plan**: Free (or Pro $25/mo for always-on + 8 GB storage + daily backups).
3. Wait ~2 min for provisioning.
4. **Enable pgvector**: Project → **Database** → **Extensions** → search `vector` → toggle on. (Or run `create extension if not exists vector;` from the SQL editor.) The app's `ensure_schema()` will create the tables on first run.
5. **Get the connection string**: Project → **Settings** → **Database** → **Connection string** → **URI** tab → copy the **"Session" / direct connection** URL (port 5432). It looks like:

   ```
   postgresql://postgres.<project-ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres
   ```

   ```bash
   export DATABASE_URL="postgresql://postgres.xxxxx:YOUR_PASSWORD@aws-0-ap-southeast-2.pooler.supabase.com:5432/postgres"
   ```

   > **Direct vs. pooled.** Supabase exposes a pgBouncer-pooled URL on port 6543 and a direct URL on 5432. The worker holds a long-lived connection and runs `LISTEN`-style queries (`FOR UPDATE SKIP LOCKED`) so it wants the **direct** connection (port 5432). The backend is also a long-lived process and works fine on either. Use direct for both unless you start running into the free-tier connection limit (60 connections), in which case put the *backend* on the pooled URL and keep the worker on direct.

6. **(Free tier only.)** Supabase pauses projects after 7 days of inactivity. The worker's 3-second poll loop counts as activity, so as long as the Fargate task is running you never sleep. If you `desired-count=0` the worker for a week, expect a one-time ~30 s cold start on the next request.

That's it for Option A — skip ahead to [Step 2](#step-2--container-registry-two-ecr-repos). The App Runner VPC connector (Step 4) and the in-VPC ingress rules in Step 5 become **unnecessary** with Supabase, because both containers reach the DB over the public internet via TLS.

### Option B — RDS Postgres + pgvector (in your VPC)

```bash
# 1a) Networking: pick the default VPC + its first two subnets
export VPC_ID=$(aws ec2 describe-vpcs --filters Name=is-default,Values=true \
   --query 'Vpcs[0].VpcId' --output text)
export SUBNET_IDS=$(aws ec2 describe-subnets --filters Name=vpc-id,Values=$VPC_ID \
   --query 'Subnets[0:2].SubnetId' --output text | tr '\t' ',')

# 1b) Security group: backend + worker → DB on 5432
aws ec2 create-security-group --group-name $PROJECT-db-sg \
   --description "Postgres access from backend + worker" --vpc-id $VPC_ID
export DB_SG_ID=$(aws ec2 describe-security-groups \
   --filters Name=group-name,Values=$PROJECT-db-sg \
   --query 'SecurityGroups[0].GroupId' --output text)
aws ec2 authorize-security-group-ingress --group-id $DB_SG_ID \
   --protocol tcp --port 5432 --source-group $DB_SG_ID

# 1c) Create the RDS instance (db.t4g.micro, free-tier-eligible for the first year)
aws rds create-db-instance \
   --db-instance-identifier $PROJECT-db \
   --db-instance-class db.t4g.micro \
   --engine postgres \
   --engine-version 16.4 \
   --master-username finance \
   --master-user-password "CHANGE-ME-USE-SECRETS-MANAGER" \
   --allocated-storage 20 \
   --storage-type gp3 \
   --db-name finance_rag \
   --vpc-security-group-ids $DB_SG_ID \
   --no-publicly-accessible \
   --backup-retention-period 1

# Wait ~5 min, then grab the endpoint
aws rds wait db-instance-available --db-instance-identifier $PROJECT-db
export DB_HOST=$(aws rds describe-db-instances --db-instance-identifier $PROJECT-db \
   --query 'DBInstances[0].Endpoint.Address' --output text)
export DATABASE_URL="postgresql://finance:CHANGE-ME-USE-SECRETS-MANAGER@$DB_HOST:5432/finance_rag"
echo "$DATABASE_URL"
```

**Enable pgvector.** Connect once and create the extension (the worker also calls `ensure_schema()` which does this idempotently, but RDS requires the master user to do it the first time):

```bash
# From a one-shot EC2 instance in the same VPC, or via a bastion / SSH-tunnel:
psql "$DATABASE_URL" -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

> **Production hardening (not in this minimal path):** store the password in AWS Secrets Manager and reference it from the App Runner / Fargate env config; rotate; enable storage encryption; turn on Performance Insights.

---

## Step 2 — Container registry (two ECR repos)

```bash
aws ecr create-repository --repository-name $PROJECT-backend --region $AWS_REGION
aws ecr create-repository --repository-name $PROJECT-worker  --region $AWS_REGION

aws ecr get-login-password --region $AWS_REGION \
   | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
```

## Step 3 — Build + push

The two images share `requirements.txt` but differ in what they `COPY` and `CMD`. Build for `linux/amd64` explicitly (matters when building from Apple Silicon):

```bash
docker buildx build --platform linux/amd64 -f Dockerfile.backend \
   -t $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$PROJECT-backend:latest --push .

docker buildx build --platform linux/amd64 -f Dockerfile.worker \
   -t $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$PROJECT-worker:latest --push .
```

---

## Step 4 — Backend (App Runner)

Create from the ECR image. Console flow:

1. **App Runner → Create service**
2. **Source:** Container registry → Amazon ECR → pick `$PROJECT-backend:latest`
3. **Deployment trigger:** Automatic (App Runner re-deploys on `latest` push)
4. **Service settings:**
   - CPU / Memory: **1 vCPU / 2 GB**
   - Port: **8000**
   - Health check: HTTP `GET /api/health` (every 20s)
   - Auto-scaling: min 1 / max 1 instance, concurrency 50
5. **VPC connector:**
   - **Option A (Supabase):** skip — App Runner's default egress reaches the public Supabase endpoint directly. Faster to provision, no NAT.
   - **Option B (RDS):** create one bound to the default VPC + the same security group as the DB. *(Required — App Runner needs explicit VPC egress to reach a private-subnet RDS.)*
6. **Environment variables** (see [full list below](#environment-variables)).

App Runner gives you `https://<random>.<region>.awsapprunner.com`. Record it — the frontend needs it.

> **Idle billing.** With min instances = 1 and auto-scaling concurrency tuned, App Runner *pauses provisioned instances* between requests and only charges for memory ($0.007/GB-hour). 2 GB paused = ~$10/mo. Active billing ($0.064/vCPU-hour + $0.007/GB-hour) is per-request.

## Step 5 — Worker (ECS Fargate)

The worker polls every 3 s and must run continuously — App Runner's pause-when-idle model is wrong for it. Fargate is.

```bash
# 5a) Create cluster
aws ecs create-cluster --cluster-name $PROJECT --region $AWS_REGION

# 5b) Task definition (save as task-worker.json then register)
cat > task-worker.json <<EOF
{
  "family": "$PROJECT-worker",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::$AWS_ACCOUNT_ID:role/ecsTaskExecutionRole",
  "containerDefinitions": [{
    "name": "worker",
    "image": "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$PROJECT-worker:latest",
    "essential": true,
    "environment": [
      {"name": "DATABASE_URL",         "value": "$DATABASE_URL"},
      {"name": "OPENAI_API_KEY",       "value": "sk-..."},
      {"name": "MODEL_PROVIDER",       "value": "openai"},
      {"name": "SEC_USER_AGENT",       "value": "Your Name <you@example.com>"},
      {"name": "WORKER_POLL_SECONDS",  "value": "3"},
      {"name": "WORKER_REFRESH_SCAN_SECONDS", "value": "300"}
    ],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/ecs/$PROJECT-worker",
        "awslogs-region": "$AWS_REGION",
        "awslogs-create-group": "true",
        "awslogs-stream-prefix": "worker"
      }
    }
  }]
}
EOF

aws ecs register-task-definition --cli-input-json file://task-worker.json

# 5c) Service: desired-count = 1 (single-worker by design — serial pipeline)
#
# Network config differs by DB choice:
#   Option A (Supabase): any SG that allows outbound 443/5432; the default VPC's
#                        default SG works. Public IP needed for internet egress.
#   Option B (RDS):      use $DB_SG_ID so the worker inherits the SG-to-SG
#                        ingress rule on the RDS security group.
export WORKER_SG_ID=$DB_SG_ID  # or substitute the default SG for Option A

aws ecs create-service \
   --cluster $PROJECT \
   --service-name $PROJECT-worker \
   --task-definition $PROJECT-worker \
   --desired-count 1 \
   --launch-type FARGATE \
   --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_IDS],securityGroups=[$WORKER_SG_ID],assignPublicIp=ENABLED}"
```

> **Fargate Spot variant.** Add `--capacity-provider-strategy capacityProvider=FARGATE_SPOT,weight=1` instead of `--launch-type FARGATE` to drop the worker bill by ~70 %. Spot can be interrupted; the worker handles that fine because `claim_next_job` is `FOR UPDATE SKIP LOCKED` — an interrupted job goes back to `queued` and the next instance picks it up.

## Step 6 — Frontend (S3 + CloudFront)

```bash
# 6a) Point the React app at the App Runner URL
cd frontend
echo "REACT_APP_API_URL=https://<your-apprunner-url>" > .env.production
npm run build
cd ..

# 6b) Bucket + upload
export FE_BUCKET=$PROJECT-ui-$AWS_ACCOUNT_ID
aws s3 mb s3://$FE_BUCKET --region $AWS_REGION
aws s3 sync frontend/build/ s3://$FE_BUCKET --delete

# 6c) CloudFront: console is faster than CLI here.
#     - Origin: $FE_BUCKET (use OAC, not the legacy OAI)
#     - Default root object: index.html
#     - Error 403 / 404 → /index.html with 200 (SPA routing)
#     - Cache policy: CachingOptimized
```

---

## Environment variables

Set on **both** the App Runner service and the Fargate task definition unless noted.

| Var | Required by | Notes |
|---|---|---|
| `DATABASE_URL` | both | Supabase: `postgresql://postgres.<ref>:<pw>@aws-0-<region>.pooler.supabase.com:5432/postgres`. RDS: `postgresql://finance:…@$DB_HOST:5432/finance_rag`. |
| `OPENAI_API_KEY` | worker (backend uses it for the legacy `/api/analyze`) | If unset, both fall back to Ollama or `MockLLM` — useless in prod. |
| `MODEL_PROVIDER` | both | `openai` for production. `auto` works but is fragile. |
| `SEC_USER_AGENT` | worker | Real contact string. SEC rate-limits anonymous traffic. |
| `POLYGON_API_KEY` | worker | Optional — falls back to synthetic news. |
| `ANALYZE_TIMEOUT_SECS` | backend | Default 900. |
| `WORKER_POLL_SECONDS` | worker | Default 3. |
| `WORKER_REFRESH_SCAN_SECONDS` | worker | Default 300. |
| `HOST`, `PORT` | backend | Set by Dockerfile.backend; App Runner reads PORT. |
| `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` | worker (optional) | LangSmith. |

> **Move secrets out of the task definition.** For anything non-demo, replace inline `OPENAI_API_KEY` / `DATABASE_URL` values with `"valueFrom": "<secret-arn>"` and create entries in Secrets Manager. The IAM role just needs `secretsmanager:GetSecretValue`.

---

## Smoke test

```bash
# Backend health
curl https://<your-apprunner-url>/api/health
# → {"status":"ok","model_provider":"openai","model":"gpt-4o", ...}

# Tickers list (probably empty on a fresh deploy)
curl https://<your-apprunner-url>/api/tickers

# Cache-miss path: request AAPL/short, get 202 + job_id, poll
curl https://<your-apprunner-url>/api/snapshot/AAPL/short
# → 202 {"status":"queued","job_id":"<uuid>"}
sleep 30
curl https://<your-apprunner-url>/api/jobs/<uuid>
# Expect status to flip queued → running → ready within ~3–5 min.
# If it stays "queued" forever, the worker isn't draining — check
# the Fargate task's CloudWatch logs.
```

If a job goes `queued → ready` end-to-end, the system is wired correctly. The CloudFront URL serving the React build is then the user-facing demo.

---

## Cost comparison vs Option B-lite

Both estimates are **idle / low-traffic baselines** (no real users, minimal LLM spend). LLM costs scale identically and are excluded.

| Line item | This plan (B-lite-lite) | Option B-lite (latest_plan.md) |
|---|---|---|
| **HTTP serving** | App Runner backend, 1 vCPU / 2 GB, paused: 2 GB × $0.007 × 730h | API Gateway + Lambda: ~few $ |
|  | ≈ **$10/mo** | ≈ **$3/mo** |
| **Worker** | Fargate 0.25 vCPU / 1 GB continuous | Fargate 0.5 vCPU / 1 GB continuous |
|  | ≈ **$9/mo** (or ~$3 on Spot) | ≈ **$18/mo** |
| **Database — Option A (Supabase Free)** | Hosted Postgres + pgvector, 500 MB | _n/a — B-lite assumes Aurora_ |
|  | ≈ **$0/mo** | |
| **Database — Option A (Supabase Pro)** | Hosted Postgres + pgvector, 8 GB, always-on, daily backups | _n/a_ |
|  | ≈ **$25/mo** | |
| **Database — Option B (RDS)** | RDS `t4g.micro` + 20 GB gp3 | Aurora Serverless v2, 0.5 ACU floor |
|  | ≈ **$14/mo** | ≈ **$45/mo** |
| **Hot cache** | _(reuses Postgres)_ | DynamoDB on-demand: pennies |
|  | ≈ **$0** | ≈ **$1/mo** |
| **Frontend (S3 + CloudFront)** | ≈ $1/mo | ≈ $1/mo |
| **Cron / scheduling** | _(worker self-schedules)_ | EventBridge: free tier |
| **Observability (CloudWatch basic)** | ≈ $1/mo | ≈ $2/mo |
| **WAF** | _not included_ | ≈ $5/mo + per-request |
| **NAT egress** | $0 (App Runner uses managed VPC connector; Fargate task has `assignPublicIp=ENABLED`) | depends |
| **Total idle (Supabase Free + Spot worker)** | **~$15/mo** | _n/a_ |
| **Total idle (Supabase Free + standard worker)** | **~$21/mo** | _n/a_ |
| **Total idle (Supabase Pro + standard worker)** | **~$46/mo** | _n/a_ |
| **Total idle (RDS + standard worker)** | **~$35/mo** | **~$75/mo** |
| **Total idle (RDS + Spot worker)** | **~$29/mo** | |

**Cheapest viable runnable demo: ~$15/mo** (Supabase Free + Fargate Spot worker + App Runner + S3). That's **~80 % off Option B-lite** and **~57 % off the RDS variant of this plan**. Caveats: 500 MB DB ceiling, no auto-backups on free Supabase, and pause-after-1-week on idle if you also `desired-count=0` the worker.

### When B-lite-lite stops winning

- **Above ~50 concurrent users.** The single App Runner instance + single Fargate worker tops out. B-lite's Lambda + scaling Fargate handles bursts more gracefully.
- **Above ~100 jobs/hour.** Both DB options start to hurt:
  - **Supabase Free** caps at ~60 concurrent connections and 500 MB storage. The pipeline's RAG queries cause connection churn; you'll see `too many clients already`. Move to Supabase Pro ($25/mo, ~200 connections, 8 GB) or migrate to RDS.
  - **RDS `t4g.micro`** has 1 vCPU / 1 GB with a sensible ceiling around 2 conn-per-vCPU; bump to `t4g.small` (~$28/mo) or move to Aurora Serverless v2.
- **You need WAF / per-user metering / API keys.** Phase 3 of the plan adds these — at that point you've grown into B-lite's shape anyway.

### What you give up

| | B-lite-lite | B-lite |
|---|---|---|
| Auto-scales request path | only within App Runner's auto-scaling | Lambda → effectively unbounded |
| DB headroom | t4g.micro ceiling ~50 connections | Aurora ACU scales 0.5 → 16 |
| Hot read path | Postgres | DynamoDB single-digit ms |
| Edge ACLs / rate limiting | missing (App Runner has no native WAF) | CloudFront + WAF |
| Operational complexity | 4 AWS services | 7+ AWS services |

For Phase 1 + the "honest backtest dashboard" of Phase 2, B-lite-lite is more than enough. The migration to B-lite is also incremental: swap RDS → Aurora first, then put API Gateway + Lambda *in front of* App Runner (or replace it), then add DynamoDB. Nothing about this plan paints you into a corner.

---

## Teardown

```bash
aws ecs update-service --cluster $PROJECT --service $PROJECT-worker --desired-count 0
aws ecs delete-service --cluster $PROJECT --service $PROJECT-worker --force
aws ecs delete-cluster --cluster $PROJECT
# Delete the App Runner service from the console (no CLI shortcut without ARN lookup).

# Option A (Supabase): delete the project from supabase.com → Project → Settings → General → Delete project.
# Option B (RDS):
aws rds delete-db-instance --db-instance-identifier $PROJECT-db --skip-final-snapshot

aws s3 rb s3://$FE_BUCKET --force
# Delete the CloudFront distribution from the console.
aws ecr delete-repository --repository-name $PROJECT-backend --force
aws ecr delete-repository --repository-name $PROJECT-worker  --force
```

---

## Troubleshooting

- **App Runner deploy stuck "Operation in progress"** — Option B (RDS) only: usually a VPC-connector misconfiguration. The connector needs subnets *and* a security group that the RDS instance's security group ingress allows. Option A (Supabase) should never hit this; if it does, the App Runner service has a misconfigured VPC connector it doesn't need — remove it.
- **Worker exits with `psycopg.OperationalError`** —
  - *Option A:* `DATABASE_URL` is wrong (most often you pasted the pooled URL on port 6543 and the worker needs the direct URL on 5432), or the password contains URL-unsafe characters that weren't percent-encoded.
  - *Option B:* Fargate task's security group isn't in the DB's allowed ingress list, or the DB endpoint isn't reachable from the task's subnet.
- **Supabase: `FATAL: too many connections for role`** — you hit the free tier's ~60 connection ceiling. Either move the backend to the pooled URL (port 6543), drop `WORKER_POLL_SECONDS` (each poll opens then returns a connection — verify the worker uses a connection pool), or upgrade to Pro.
- **Supabase: schema/extension errors on first run** — pgvector extension wasn't enabled in the dashboard. Go to Database → Extensions → search `vector` → toggle on, then redeploy the worker. The app's `ensure_schema()` does *not* `CREATE EXTENSION` (Supabase requires the dashboard toggle for the first install).
- **`/api/snapshot/...` returns 202 forever** — worker isn't draining. Most common cause: ECS service desired-count is 0, or the task is stuck pulling the image. Check `aws ecs describe-services` and the task's CloudWatch logs.
- **Backend image fails to push from Apple Silicon** — you skipped `--platform linux/amd64` on the `docker buildx` command. The image will be `arm64`-only and App Runner Fargate runners reject it.
- **SEC EDGAR returns 403 / rate-limit** — `SEC_USER_AGENT` is unset or generic. Set it to a real `Name <email>` string and redeploy the worker.
