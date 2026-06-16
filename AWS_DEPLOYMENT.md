# AWS Deployment Guide — Option B-lite-lite

> **TL;DR.** Run the Flask backend on **ECS Express Mode** (Fargate-backed; managed ALB + HTTPS URL), the worker on plain ECS Fargate, store snapshots/filings in Supabase (free) or a `t4g.micro` RDS Postgres (pgvector), serve the frontend from S3 + CloudFront. **~$41/mo idle (lean)** to ~$77/mo (default-sized + RDS), no Lambda port required.

This guide reflects the **current** code shape (post-`21216cc1`): a backend that only **reads or enqueues**, a **worker** that drains the queue and runs the chain + debate pipeline, and a Postgres + pgvector database that both processes share. The previous version of this file only deployed the backend container — it predates the worker and database, and was not actually runnable end-to-end.

For a comparison with the higher-ceiling "Option B-lite" (Lambda + Aurora + DynamoDB) from [`latest_plan.md`](latest_plan.md), see the [cost section](#cost-comparison-vs-option-b-lite) at the bottom.

---

## Architecture

```
                  CloudFront
                       │
        ┌──────────────┼─────────────────┐
        │              │                 │
   S3: frontend   ECS Express Mode    ── (no auth, no API Gateway)
                  (Fargate + auto-ALB
                   + auto-URL .ecs.<region>.on.aws)
                       │
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

## What's different from the previous versions of this guide

> Earlier revisions of this guide had the backend on **AWS App Runner**. App Runner stopped accepting new customers on 30 Apr 2026 — we're now on **ECS Express Mode**, AWS's named successor. The two have similar shapes (give it a container image, get a managed URL) but Express Mode is Fargate underneath with an auto-provisioned ALB, so the idle cost is meaningfully higher (no "pause when idle" billing).

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
export AWS_REGION=us-east-2
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

That's it for Option A — skip ahead to [Step 2](#step-2--container-registry-two-ecr-repos). The custom VPC config in Step 4 (Express Mode) and the in-VPC ingress rules in Step 5 (worker) become **unnecessary** with Supabase, because both containers reach the DB over the public internet via TLS.

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

> **Production hardening (not in this minimal path):** store the password in AWS Secrets Manager and reference it from the Express Mode / Fargate env config; rotate; enable storage encryption; turn on Performance Insights.

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

## Step 3.5 — Secrets (Secrets Manager)

The containers need two **secret** values at runtime — `DATABASE_URL` (your Supabase URL, which embeds the DB password) and `OPENAI_API_KEY`. These must **not** live in the task definition as plaintext, in the image, or in GitHub. They go in AWS Secrets Manager; the ECS task reads them at container start; only their (non-sensitive) ARNs ever appear in config.

> **Why not GitHub Actions secrets?** Those authenticate the *deploy pipeline* to AWS (build/push/deploy). They are a different plane from the *app's* runtime config. The app's secret **values** never need to touch GitHub — the workflow references only the Secrets Manager **ARNs**, which aren't sensitive.

### Create the secrets

Three runtime secrets: `DATABASE_URL` + `OPENAI_API_KEY` (needed by both backend and worker) and `POLYGON_API_KEY` (worker only — for real news; without it the pipeline falls back to *synthetic* news).

```bash
# Pull the values straight from your local .env (they never get committed).
source .env   # or paste the values inline

aws secretsmanager create-secret \
   --name $PROJECT/database-url \
   --secret-string "$DATABASE_URL" \
   --region $AWS_REGION

aws secretsmanager create-secret \
   --name $PROJECT/openai-key \
   --secret-string "$OPENAI_API_KEY" \
   --region $AWS_REGION

aws secretsmanager create-secret \
   --name $PROJECT/polygon-key \
   --secret-string "$POLYGON_API_KEY" \
   --region $AWS_REGION

# Capture the ARNs — the deploy step and the worker task definition reference these.
export DATABASE_URL_SECRET_ARN=$(aws secretsmanager describe-secret \
   --secret-id $PROJECT/database-url --query ARN --output text --region $AWS_REGION)
export OPENAI_KEY_SECRET_ARN=$(aws secretsmanager describe-secret \
   --secret-id $PROJECT/openai-key --query ARN --output text --region $AWS_REGION)
export POLYGON_KEY_SECRET_ARN=$(aws secretsmanager describe-secret \
   --secret-id $PROJECT/polygon-key --query ARN --output text --region $AWS_REGION)
echo "$DATABASE_URL_SECRET_ARN"
echo "$OPENAI_KEY_SECRET_ARN"
echo "$POLYGON_KEY_SECRET_ARN"
```

> To rotate a value later: `aws secretsmanager put-secret-value --secret-id $PROJECT/openai-key --secret-string "sk-new..."`. The next container start picks it up — no image rebuild.

### Grant the execution role read access

The **task execution role** (`ecsTaskExecutionRole`, created in [Step 4a](#4a--iam-roles-one-time)) is what ECS uses to fetch these at container start. Attach an inline policy scoped to exactly these two secrets — run this **after** Step 4a creates the role:

```bash
aws iam put-role-policy \
   --role-name ecsTaskExecutionRole \
   --policy-name ReadAppSecrets \
   --policy-document "$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": "secretsmanager:GetSecretValue",
    "Resource": ["$DATABASE_URL_SECRET_ARN", "$OPENAI_KEY_SECRET_ARN", "$POLYGON_KEY_SECRET_ARN"]
  }]
}
EOF
)"
```

Without this grant, the service creation/update succeeds but tasks fail to start with `ResourceInitializationError: unable to pull secrets`.

### Wire the ARNs into CI (for the GitHub Actions deploy)

If you deploy via `build-and-push.yml` (rather than the manual `create-express-gateway-service` in 4b), add the two ARNs as GitHub repo secrets so the deploy step can pass them as `valueFrom`:

```bash
# GitHub → repo → Settings → Secrets and variables → Actions → New repository secret
#   DATABASE_URL_SECRET_ARN = <the arn echoed above>
#   OPENAI_KEY_SECRET_ARN    = <the arn echoed above>
# (These ARNs are not sensitive; they're stored as secrets only to match the
#  existing ECS_EXEC_ROLE_ARN / ECS_INFRA_ROLE_ARN pattern. Repo *variables*
#  would work equally well.)
```

The deploy job already passes them as the `secrets:` input to `aws-actions/amazon-ecs-deploy-express-service`, and `MODEL_PROVIDER=openai` as a plain `environment-variables` entry — so a CI-created service comes up fully configured, and every subsequent deploy re-asserts the env (safe even if the action would otherwise replace the container definition).

---

## Step 3.6 — Account + CI bootstrap (one-time)

> **These two steps are easy to miss** because nothing in the repo references them — they're account-level prerequisites. They were the cause of the two errors most people hit on their first deploy:
> - `User ... is not authorized to perform: ecs:DescribeServices` → deploy user is missing ECS permissions (3.6a)
> - `Unable to assume the service linked role. Please verify that the ECS service linked role exists.` → fresh account has no ECS/ELB/autoscaling service-linked roles (3.6b)

### 3.6a — Grant the deploy IAM user ECS Express permissions

The IAM user behind your GitHub Actions keys (`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`) needs **two** permission sets:

1. **ECR** (for `build` / push) — attach the AWS-managed `AmazonEC2ContainerRegistryPowerUser`, or an equivalent scoped policy. (If your build+push already succeeds, this is done.)
2. **ECS Express + PassRole** (for the `deploy` job) — not covered by any single managed policy, so attach this inline policy:

```bash
export DEPLOY_IAM_USER=<your-ci-iam-user>   # e.g. the user whose keys are in GitHub secrets

aws iam put-user-policy \
  --user-name "$DEPLOY_IAM_USER" \
  --policy-name GitHubActionsECSExpressDeploy \
  --policy-document "$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ECSExpressDeploy",
      "Effect": "Allow",
      "Action": [
        "ecs:CreateCluster",
        "ecs:RegisterTaskDefinition",
        "ecs:CreateExpressGatewayService",
        "ecs:UpdateExpressGatewayService",
        "ecs:DescribeExpressGatewayService",
        "ecs:DescribeClusters",
        "ecs:DescribeServices",
        "ecs:ListServiceDeployments",
        "ecs:DescribeServiceDeployments",
        "ecs:UpdateService"
      ],
      "Resource": "*"
    },
    {
      "Sid": "PassExpressRoles",
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": [
        "arn:aws:iam::${AWS_ACCOUNT_ID}:role/ecsTaskExecutionRole",
        "arn:aws:iam::${AWS_ACCOUNT_ID}:role/ecsInfrastructureRoleForExpressServices"
      ]
    }
  ]
}
EOF
)"
```

The `iam:PassRole` block references the two roles from [Step 4a](#4a--iam-roles-one-time). The policy attaches fine before those roles exist — but the roles must exist by deploy time, so do Step 4a too (order between 3.6a and 4a doesn't matter).

### 3.6b — Create the ECS service-linked roles

On a **fresh account that has never used ECS via the console**, the service-linked roles ECS needs to manage networking, the load balancer, and auto-scaling don't exist yet. Create all three once (account-wide, free; an "already exists / has been taken" error is harmless):

```bash
aws iam create-service-linked-role --aws-service-name ecs.amazonaws.com
aws iam create-service-linked-role --aws-service-name elasticloadbalancing.amazonaws.com
aws iam create-service-linked-role --aws-service-name ecs.application-autoscaling.amazonaws.com
```

> These are distinct from the two roles in Step 4a. The Step 4a roles are what your *containers* assume (pull images, read secrets, provision the ALB). The service-linked roles are what the ECS *service itself* assumes to operate — you never reference them directly, AWS just needs them to exist.

---

## Step 4 — Backend (ECS Express Mode)

> **Why not App Runner?** AWS announced on 30 Apr 2026 that App Runner stopped accepting new customers, and the recommended replacement is **Amazon ECS Express Mode** (launched 21 Nov 2025). Express Mode is the spiritual successor: provide a container image and two IAM roles, and ECS provisions a Fargate service + Application Load Balancer + auto-scaling + a `*.ecs.<region>.on.aws` HTTPS URL with a single API call. No extra charge for Express Mode itself — you pay only for the Fargate compute + ALB underneath.

### 4a — IAM roles (one-time)

Express Mode needs two roles. Both attach AWS-managed policies; no custom policy authoring required.

```bash
# Task execution role — lets ECS pull from ECR and write CloudWatch logs
aws iam create-role --role-name ecsTaskExecutionRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "ecs-tasks.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

aws iam attach-role-policy --role-name ecsTaskExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

# Infrastructure role — lets Express Mode provision the ALB + scaling on your behalf
aws iam create-role --role-name ecsInfrastructureRoleForExpressServices \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Sid": "AllowAccessInfrastructureForECSExpressServices",
      "Effect": "Allow",
      "Principal": {"Service": "ecs.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

aws iam attach-role-policy --role-name ecsInfrastructureRoleForExpressServices \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSInfrastructureRoleforExpressGatewayServices
```

Record both ARNs — the deploy step needs them:

```bash
export EXEC_ROLE_ARN=$(aws iam get-role --role-name ecsTaskExecutionRole --query 'Role.Arn' --output text)
export INFRA_ROLE_ARN=$(aws iam get-role --role-name ecsInfrastructureRoleForExpressServices --query 'Role.Arn' --output text)
```

### 4b — Deploy the backend

There are two ways to create/update the Express service. **Path A (GitHub Actions) is the one this repo is wired for and the recommended path** — it builds, smoke-tests, and deploys on every push to `main`. Path B is the manual equivalent, kept for reference and one-off debugging.

#### Path A — GitHub Actions (recommended)

The [`.github/workflows/build-and-push.yml`](.github/workflows/build-and-push.yml) workflow runs `build → smoke-test → deploy`. The `deploy` job calls `aws-actions/amazon-ecs-deploy-express-service`, which **creates the service on the first run and updates it (new image + re-asserted env) on every run after**. No manual `create-express-gateway-service` needed.

Set these **GitHub repo secrets** (Settings → Secrets and variables → Actions) so the deploy job is fully configured — if any is missing, the deploy step skips with a warning naming the gap:

| GitHub repo secret | Value | From |
|---|---|---|
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | the deploy IAM user's keys | [3.6a](#36a--grant-the-deploy-iam-user-ecs-express-permissions) |
| `ECS_EXEC_ROLE_ARN` | `ecsTaskExecutionRole` ARN | [4a](#4a--iam-roles-one-time) (`$EXEC_ROLE_ARN`) |
| `ECS_INFRA_ROLE_ARN` | `ecsInfrastructureRoleForExpressServices` ARN | [4a](#4a--iam-roles-one-time) (`$INFRA_ROLE_ARN`) |
| `DATABASE_URL_SECRET_ARN` | Secrets Manager ARN | [3.5](#step-35--secrets-secrets-manager) |
| `OPENAI_KEY_SECRET_ARN` | Secrets Manager ARN | [3.5](#step-35--secrets-secrets-manager) |
| `POLYGON_KEY_SECRET_ARN` | Secrets Manager ARN | [3.5](#step-35--secrets-secrets-manager) — backend needs it because `/api/price` serves from Polygon |

Then push to `main` (or run the workflow manually from the Actions tab). On success the deploy log prints the service URL. The full prerequisite chain for a green deploy, in order:

1. ECR repos exist ([Step 2](#step-2--container-registry-two-ecr-repos))
2. Secrets Manager secrets + execution-role grant ([Step 3.5](#step-35--secrets-secrets-manager))
3. Deploy-user ECS policy ([3.6a](#36a--grant-the-deploy-iam-user-ecs-express-permissions)) + service-linked roles ([3.6b](#36b--create-the-ecs-service-linked-roles))
4. The two IAM roles ([4a](#4a--iam-roles-one-time))
5. All five GitHub repo secrets above

The `container-port` (8000), `environment-variables` (`MODEL_PROVIDER=openai`), and `secrets` (DB + OpenAI ARNs) are already set in the workflow's deploy step — see [build-and-push.yml](.github/workflows/build-and-push.yml).

#### Path B — Manual CLI (alternative)

> Demo-grade: this inlines the secret **values** as plaintext env. For anything real, swap `environment` for the `secrets`/`valueFrom` form (see Step 3.5) so the values come from Secrets Manager instead.

```bash
aws ecs create-express-gateway-service \
  --service-name finance-agents-backend \
  --execution-role-arn "$EXEC_ROLE_ARN" \
  --infrastructure-role-arn "$INFRA_ROLE_ARN" \
  --primary-container '{
    "image": "'"$AWS_ACCOUNT_ID"'.dkr.ecr.'"$AWS_REGION"'.amazonaws.com/finance-agents-backend:latest",
    "containerPort": 8000,
    "environment": [
      {"name": "DATABASE_URL",         "value": "'"$DATABASE_URL"'"},
      {"name": "OPENAI_API_KEY",       "value": "sk-..."},
      {"name": "MODEL_PROVIDER",       "value": "openai"},
      {"name": "ANALYZE_TIMEOUT_SECS", "value": "900"}
    ]
  }' \
  --cpu 1 \
  --memory 2 \
  --health-check-path "/api/health" \
  --scaling-target '{"minTaskCount":1,"maxTaskCount":3}' \
  --monitor-resources
```

`--monitor-resources` streams provisioning status to stdout (ALB, target group, service, etc.) and exits once `ACTIVE`. Total time: ~3–5 min on a fresh region.

The service URL is printed on completion. **It is an opaque, auto-generated hostname — NOT derived from the service name** — of the form:

```
https://fi-<32-hex-chars>.ecs.<region>.on.aws/
# e.g. https://fi-b4d4a94c3b834d5faaa5d465255181de.ecs.us-east-2.on.aws/
```

**Record this exact hostname** — you cannot guess or reconstruct it from `finance-agents-backend`. It's the CloudFront `/api/*` origin in Step 6. Retrieve it any time from the ECS console (the Express service page) or:

```bash
aws ecs describe-express-gateway-service \
  --service-arn arn:aws:ecs:$AWS_REGION:$AWS_ACCOUNT_ID:service/default/finance-agents-backend \
  --region $AWS_REGION
```

(Look for the URL/endpoint field in the output.) Export it for the steps below: `export BACKEND_URL=fi-xxxxxxxx.ecs.$AWS_REGION.on.aws`.

### 4c — Note on networking by DB choice

- **Option A (Supabase):** no extra config — Express Mode's auto-provisioned VPC has outbound internet egress through a managed NAT, which reaches the public Supabase endpoint over TLS.
- **Option B (RDS):** create the Express service in the same VPC + subnets as the RDS instance, and add the Express Mode service's security group to the RDS security group's ingress on port 5432. The `aws ecs create-express-gateway-service` command takes `--network-configuration` for this; see the [Express Mode docs](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/express-service-overview.html) for the syntax.

### 4d — Cost shape

Express Mode is free; you pay for what it provisions:

| Resource | Default (1 vCPU / 2 GB / 1 task) | Idle cost |
|---|---|---|
| Fargate compute | 1 vCPU × $0.04048/hr × 730 + 2 GB × $0.004445/hr × 730 | ~$36/mo |
| Application Load Balancer | 1 ALB-hour × 730 | ~$16/mo |
| **Backend total (idle)** | | **~$52/mo** |

Smaller settings cut this meaningfully:

| Profile | CPU / Mem | Backend total idle |
|---|---|---|
| Lean (`--cpu 0.5 --memory 1`) | 0.5 vCPU / 1 GB | ~$36/mo |
| **Default** | 1 vCPU / 2 GB | ~$52/mo |
| Production-ish (`--cpu 2 --memory 4`) | 2 vCPU / 4 GB | ~$93/mo |

> **ALB consolidation.** AWS automatically packs up to 25 Express Mode services behind a single ALB when possible — so if you eventually run multiple Express services in this account, the $16 ALB cost is amortized across them. For a single-service deploy it's a fixed line item.

## Step 5 — Worker (ECS Fargate)

The worker polls every 3 s and must run continuously. Unlike the backend, it doesn't expose HTTP — so it doesn't belong on Express Mode. Plain ECS Fargate with `desired-count=1` is the right shape.

> **Worker env note.** The worker now also ingests SEC EDGAR filings: on-demand before each pipeline run (only when a ticker has no stored filings — there are no local PDFs on AWS) and on a weekly sweep across all tracked tickers for newly-released filings. So `SEC_USER_AGENT` must be a real contact string on the worker or SEC throttles the downloads. The worker also fetches Polygon price series for snapshots, so `POLYGON_API_KEY` is required there too (both are in the task def below). Tunables: `WORKER_FILING_SCAN_SECONDS` (weekly sweep interval, default `604800`), `EDGAR_FORMS` (default `10-K,10-Q`), `EDGAR_LIMIT` (max filings per form per ticker, default `4`).

```bash
# 5a) Create cluster
aws ecs create-cluster --cluster-name $PROJECT --region $AWS_REGION

# 5a-2) The task def below uses "awslogs-create-group": "true", which makes the
# log driver create the CloudWatch group at task start. The AWS-managed
# AmazonECSTaskExecutionRolePolicy grants CreateLogStream/PutLogEvents but NOT
# CreateLogGroup — so without this grant every task dies at init with
# "ResourceInitializationError ... not authorized to perform: logs:CreateLogGroup".
aws iam put-role-policy \
  --role-name ecsTaskExecutionRole \
  --policy-name CreateEcsLogGroups \
  --policy-document "$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": "logs:CreateLogGroup",
    "Resource": "arn:aws:logs:${AWS_REGION}:${AWS_ACCOUNT_ID}:log-group:/ecs/*"
  }]
}
EOF
)"

# 5b) Task definition (save as task-worker.json then register)
cat > task-worker.json <<EOF
{
  "family": "$PROJECT-worker",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::${AWS_ACCOUNT_ID}:role/ecsTaskExecutionRole",
  "containerDefinitions": [{
    "name": "worker",
    "image": "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$PROJECT-worker:latest",
    "essential": true,
    "environment": [
      {"name": "MODEL_PROVIDER",       "value": "openai"},
      {"name": "SEC_USER_AGENT",       "value": "Your Name <you@example.com>"},
      {"name": "WORKER_POLL_SECONDS",  "value": "3"},
      {"name": "WORKER_REFRESH_SCAN_SECONDS", "value": "300"},
      {"name": "WORKER_FILING_SCAN_SECONDS",  "value": "604800"},
      {"name": "EDGAR_FORMS",          "value": "10-K,10-Q"},
      {"name": "EDGAR_LIMIT",          "value": "4"}
    ],
    "secrets": [
      {"name": "DATABASE_URL",    "valueFrom": "$DATABASE_URL_SECRET_ARN"},
      {"name": "OPENAI_API_KEY",  "valueFrom": "$OPENAI_KEY_SECRET_ARN"},
      {"name": "POLYGON_API_KEY", "valueFrom": "$POLYGON_KEY_SECRET_ARN"}
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
#   Option A (Supabase): any SG with outbound internet; the default VPC's default
#                        SG works. Public IP needed so the task can reach Supabase
#                        + OpenAI over the internet.
#   Option B (RDS):      use $DB_SG_ID so the worker inherits the SG-to-SG
#                        ingress rule on the RDS security group.
#
# Option A only — derive the default VPC's subnets + default SG (Supabase users
# skipped Step 1's Option B block, so $SUBNET_IDS / $DB_SG_ID aren't set yet):
export VPC_ID=$(aws ec2 describe-vpcs --filters Name=is-default,Values=true \
   --query 'Vpcs[0].VpcId' --output text --region $AWS_REGION)
export SUBNET_IDS=$(aws ec2 describe-subnets --filters Name=vpc-id,Values=$VPC_ID \
   --query 'Subnets[0:2].SubnetId' --output text --region $AWS_REGION | tr '\t' ',')
export WORKER_SG_ID=$(aws ec2 describe-security-groups \
   --filters Name=vpc-id,Values=$VPC_ID Name=group-name,Values=default \
   --query 'SecurityGroups[0].GroupId' --output text --region $AWS_REGION)

# Option B (RDS) instead: export WORKER_SG_ID=$DB_SG_ID  (reuse Step 1's SG + subnets)

aws ecs create-service \
   --cluster $PROJECT \
   --service-name $PROJECT-worker \
   --task-definition $PROJECT-worker \
   --desired-count 1 \
   --launch-type FARGATE \
   --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_IDS],securityGroups=[$WORKER_SG_ID],assignPublicIp=ENABLED}"
```

> **Fargate Spot variant.** Add `--capacity-provider-strategy capacityProvider=FARGATE_SPOT,weight=1` instead of `--launch-type FARGATE` to drop the worker bill by ~70 %. Spot can be interrupted; the worker handles that fine because `claim_next_job` is `FOR UPDATE SKIP LOCKED` — an interrupted job goes back to `queued` and the next instance picks it up.

### 5d — Updating the worker's env/secrets later

> **The worker is NOT CI-managed** (unlike the backend, whose env lives in the GitHub Actions deploy step). The worker's env/secrets live in the **task definition**, and the `build-and-push.yml` worker step only does `update-service --force-new-deployment` — it restarts the worker with the latest image but **reuses the current task-def revision and never changes env**. Adding a value to GitHub repo secrets does nothing for the worker.

To add or change a worker env var or secret (e.g. add `POLYGON_API_KEY`):

```bash
# 1. (secrets only) create it + grant the execution role — see Step 3.5
# 2. Edit task-worker.json: add to "environment" (plain) or "secrets" (valueFrom)
# 3. Register a new revision and roll the service onto it:
aws ecs register-task-definition --cli-input-json file://task-worker.json --region $AWS_REGION
aws ecs update-service --cluster $PROJECT --service $PROJECT-worker \
   --task-definition $PROJECT-worker --region $AWS_REGION
```

`update-service --task-definition $PROJECT-worker` (family name, no `:revision`) rolls onto the latest ACTIVE revision. *(To remove this manual dance entirely, give the worker its own CI deploy job that renders + registers the task def on push — a TODO, not yet wired.)*

## Step 6 — Frontend (S3 + CloudFront)

> **The frontend calls the API with relative `/api/...` paths** (no `REACT_APP_API_URL` in the code). That means CloudFront must serve **both** the static app *and* the API under one domain — S3 for `/*`, the Express backend for `/api/*`. Do NOT set `REACT_APP_API_URL`; the code ignores it. The two-origins setup below is what makes the relative paths resolve (and it's what the [scaling plan](#cost-comparison-vs-option-b-lite) builds on — edge-caching the API later is just swapping the `/api/*` cache policy).

```bash
# 6a) Build the static app (no API URL baked in — paths are relative)
cd frontend && npm run build && cd ..

# 6b) Bucket + upload
export FE_BUCKET=$PROJECT-ui-$AWS_ACCOUNT_ID
aws s3 mb s3://$FE_BUCKET --region $AWS_REGION
aws s3 sync frontend/build/ s3://$FE_BUCKET --delete
```

**6c) CloudFront — two origins (console is far easier than the CLI for this):**

*Origin 1 — the static app:*
- Origin domain: `$FE_BUCKET` (S3), access via **OAC** (not the legacy OAI)
- Default root object: `index.html`
- Default behavior (`*`): cache policy `CachingOptimized`
- Custom error responses: `403 → /index.html (200)` and `404 → /index.html (200)` — SPA routing so deep links like `/t/AAPL` work

*Origin 2 — the backend API:*
- Create a second origin: domain = **your Express service's opaque hostname** `fi-<hex>.ecs.<region>.on.aws` (the `$BACKEND_URL` from Step 4b — host only, no `https://`, no trailing `/`), **HTTPS only**, name `backend-express`. ⚠️ This is *not* `finance-agents-backend.ecs...` — Express generates a random hostname; a guessed/typo'd value gives CloudFront a `502: couldn't resolve the origin domain name`.
- Add a behavior: **path pattern `/api/*`** → origin `backend-express`
  - Allowed methods: GET, HEAD, OPTIONS, PUT, POST, PATCH, DELETE (POST for `/api/analyze`)
  - Cache policy: **`CachingDisabled`** (the API doesn't send cache headers yet; revisit for the scaling plan)
  - Origin request policy: **`AllViewerExceptHostHeader`** — forwards query strings (`?period=` is load-bearing for the price chart) and sets `Host` to the origin, which a custom/ALB origin requires

> `/api/*` is more specific than the default `*`, so CloudFront routes API calls to the backend and everything else to S3 automatically.
>
> **Common failure:** a frontend that loads but every API call returns **403** means the `/api/*` behavior is missing — `/api/*` is falling through to the S3 origin, which 403s on the nonexistent key. **Sparklines all identical** means the `/api/*` behavior's policy is stripping query strings — use `AllViewerExceptHostHeader`.

### 6d — Automating with GitHub Actions

Once the bucket + distribution exist (the manual 6a–6c above), [`.github/workflows/deploy-frontend.yml`](.github/workflows/deploy-frontend.yml) takes over: on every push to `main` touching `frontend/**`, it runs `npm ci && npm run build`, `s3 sync --delete` (with correct cache headers — immutable for fingerprinted assets, no-cache for `index.html`), and a CloudFront `/*` invalidation. It's a separate workflow from `build-and-push.yml` so frontend changes don't trigger backend Docker builds (and vice versa).

**Deploy IAM user permissions** — the user behind the GitHub keys needs S3 write + CloudFront invalidation:

```bash
export FE_BUCKET=...          # your bucket name from 6b
export DIST_ID=...            # see below
aws iam put-user-policy --user-name "$DEPLOY_IAM_USER" \
  --policy-name GitHubActionsFrontendDeploy \
  --policy-document "$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect":"Allow","Action":["s3:PutObject","s3:DeleteObject","s3:ListBucket"],
     "Resource":["arn:aws:s3:::$FE_BUCKET","arn:aws:s3:::$FE_BUCKET/*"]},
    {"Effect":"Allow","Action":"cloudfront:CreateInvalidation",
     "Resource":"arn:aws:cloudfront::${AWS_ACCOUNT_ID}:distribution/$DIST_ID"}
  ]
}
EOF
)"
```

**Find the CloudFront distribution ID** (it's not the `d…cloudfront.net` domain — it's the `E…` ID):

```bash
aws cloudfront list-distributions \
  --query "DistributionList.Items[?DomainName=='<your-dist>.cloudfront.net'].Id" --output text
```

**Add two GitHub repo secrets:** `FRONTEND_S3_BUCKET` (the bucket name) and `CLOUDFRONT_DISTRIBUTION_ID` (the `E…` ID). If either is unset, the workflow skips with a warning rather than failing.

---

## Environment variables

Set on **both** the Express Mode service (via `--primary-container '...environment...'` at create time, or `aws ecs update-express-gateway-service` later) and the Fargate worker task definition unless noted.

| Var | Required by | Notes |
|---|---|---|
| `DATABASE_URL` | both | Supabase: `postgresql://postgres.<ref>:<pw>@aws-0-<region>.pooler.supabase.com:5432/postgres`. RDS: `postgresql://finance:…@$DB_HOST:5432/finance_rag`. |
| `OPENAI_API_KEY` | worker (backend uses it for the legacy `/api/analyze`) | If unset, both fall back to Ollama or `MockLLM` — useless in prod. |
| `MODEL_PROVIDER` | both | `openai` for production. `auto` works but is fragile. |
| `SEC_USER_AGENT` | worker | Real contact string. SEC rate-limits anonymous traffic. |
| `POLYGON_API_KEY` | **worker + backend** | Worker: real news (else synthetic). Backend: powers `/api/price` (the chart). Both via Secrets Manager `valueFrom`. Without it: synthetic news + empty price charts (no crash). |
| `ANALYZE_TIMEOUT_SECS` | backend | Default 900. |
| `WORKER_POLL_SECONDS` | worker | Default 3. |
| `WORKER_REFRESH_SCAN_SECONDS` | worker | Default 300. |
| `HOST`, `PORT` | backend | Set by Dockerfile.backend; Express Mode reads `containerPort` from the service config (set to 8000). |
| `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` | worker (optional) | LangSmith. |

> **Move secrets out of the task definition.** For anything non-demo, replace inline `OPENAI_API_KEY` / `DATABASE_URL` values with `"valueFrom": "<secret-arn>"` and create entries in Secrets Manager. The IAM role just needs `secretsmanager:GetSecretValue`.

---

## Smoke test

```bash
# Use the opaque Express hostname from Step 4b (NOT a service-name-based URL):
export BACKEND_URL=fi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.ecs.us-east-2.on.aws

# Backend health
curl https://$BACKEND_URL/api/health
# → {"status":"healthy","message":"Finance Risk Analysis API is running"}
# (and GET / returns {"status":"ok",...} — the ALB health-check route)

# Tickers list (probably empty on a fresh deploy)
curl https://$BACKEND_URL/api/tickers

# Cache-miss path: request AAPL/short, get 202 + job_id, poll
curl https://$BACKEND_URL/api/snapshot/AAPL/short
# → 202 {"status":"queued","job_id":"<uuid>"}
sleep 30
curl https://$BACKEND_URL/api/jobs/<uuid>
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
| **HTTP serving** | ECS Express Mode (Fargate 1 vCPU / 2 GB + auto-ALB) | API Gateway + Lambda: ~few $ |
|  | ≈ **$52/mo** (lean: ~$36) | ≈ **$3/mo** |
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
| **NAT egress** | $0 (Express Mode manages its own VPC; Fargate worker has `assignPublicIp=ENABLED`) | depends |
| **Total idle (Supabase Free + Spot worker + lean Express)** | **~$41/mo** | _n/a_ |
| **Total idle (Supabase Free + Spot worker + default Express)** | **~$57/mo** | _n/a_ |
| **Total idle (Supabase Free + standard worker + default Express)** | **~$63/mo** | _n/a_ |
| **Total idle (Supabase Pro + standard worker + default Express)** | **~$88/mo** | _n/a_ |
| **Total idle (RDS + standard worker + default Express)** | **~$77/mo** | **~$75/mo** |

**Cheapest viable runnable demo: ~$41/mo** (Supabase Free + Fargate Spot worker + Express Mode lean + S3+CF). That's ~45 % off Option B-lite. The App Runner-paused trick that previously got us to ~$15/mo is gone with App Runner — Express Mode keeps Fargate tasks running, no pause-when-idle. If $41/mo is too much, the architectural levers are: smaller Express CPU (already at 0.5), drop the worker to scheduled-only via EventBridge instead of continuous, or move HTTP off Express to a cheaper container service (Lightsail Containers ~$10/mo, no ALB).

### When B-lite-lite stops winning

- **Above ~50 concurrent users.** Express Mode auto-scales the backend (default cap: 3 tasks), and the single Fargate worker tops out. B-lite's Lambda + scaling Fargate handles bursts more gracefully and bills per-request.
- **Above ~100 jobs/hour.** Both DB options start to hurt:
  - **Supabase Free** caps at ~60 concurrent connections and 500 MB storage. The pipeline's RAG queries cause connection churn; you'll see `too many clients already`. Move to Supabase Pro ($25/mo, ~200 connections, 8 GB) or migrate to RDS.
  - **RDS `t4g.micro`** has 1 vCPU / 1 GB with a sensible ceiling around 2 conn-per-vCPU; bump to `t4g.small` (~$28/mo) or move to Aurora Serverless v2.
- **You need WAF / per-user metering / API keys.** Phase 3 of the plan adds these — at that point you've grown into B-lite's shape anyway.

### What you give up

| | B-lite-lite | B-lite |
|---|---|---|
| Auto-scales request path | Express Mode scales 1 → maxTaskCount (3 default) | Lambda → effectively unbounded |
| DB headroom | t4g.micro ceiling ~50 connections | Aurora ACU scales 0.5 → 16 |
| Hot read path | Postgres | DynamoDB single-digit ms |
| Edge ACLs / rate limiting | missing (Express Mode's auto-ALB has no WAF by default) | CloudFront + WAF |
| Operational complexity | 4 AWS services | 7+ AWS services |

For Phase 1 + the "honest backtest dashboard" of Phase 2, B-lite-lite is more than enough. The migration to B-lite is also incremental: swap RDS → Aurora first, then put API Gateway + Lambda *in front of* Express Mode (or replace it), then add DynamoDB. Nothing about this plan paints you into a corner.

---

## Teardown

```bash
aws ecs update-service --cluster $PROJECT --service $PROJECT-worker --desired-count 0
aws ecs delete-service --cluster $PROJECT --service $PROJECT-worker --force
aws ecs delete-cluster --cluster $PROJECT
# Delete the Express Mode service (this also tears down its auto-provisioned ALB + target group)
aws ecs delete-express-gateway-service --service-arn "$EXPRESS_SERVICE_ARN" --force

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

- **`User ... is not authorized to perform: ecs:DescribeServices`** (or any other `ecs:*` action) in the GitHub Actions deploy job — the deploy IAM user lacks ECS Express permissions. Attach the policy in [Step 3.6a](#36a--grant-the-deploy-iam-user-ecs-express-permissions). The error names the *first* missing action; the policy covers them all so you won't hit them one at a time.
- **`Unable to assume the service linked role. Please verify that the ECS service linked role exists.`** — fresh account with no ECS/ELB/autoscaling service-linked roles. Run the three `create-service-linked-role` commands in [Step 3.6b](#36b--create-the-ecs-service-linked-roles).
- **Worker tasks fail at init with `ResourceInitializationError ... not authorized to perform: logs:CreateLogGroup`** — the task def's `awslogs-create-group: true` needs `logs:CreateLogGroup`, which the managed execution-role policy doesn't grant. Run the `put-role-policy` in [Step 5a-2](#step-5--worker-ecs-fargate). The service self-heals on the next task retry — no redeploy needed.
- **`aws ecs create-express-gateway-service` stuck at PROVISIONING** — `--monitor-resources` prints exactly which sub-resource (target group, ALB, listener, service) is taking time. Most stalls are VPC-route issues; Option B (RDS) hits these when the service is created in a subnet without a route to the RDS subnet. Option A (Supabase) hits this rarely — the default-VPC config Express Mode picks works out-of-the-box for public-internet egress.
- **`AccessDeniedException` on `iam:PassRole`** — your deploy IAM user can pass the ExecutionRole and InfrastructureRole ARNs. Add an `iam:PassRole` statement scoped to those two role ARNs, or attach `AmazonECS_FullAccess` to the deploy user temporarily.
- **Express Mode service URL returns 503 with the task RUNNING (no stopped tasks)** — the ALB health check is failing, so the only target is marked unhealthy. The most common cause on the **GitHub Actions deploy path**: the `aws-actions/amazon-ecs-deploy-express-service` action doesn't expose a `health-check-path` input, so the ALB defaults to `/` — and the Flask app 404s on `/` unless it has a root route. The backend defines a lightweight `@app.route('/')` returning 200 (no DB) for exactly this; if you removed it, the 503 returns. (The manual CLI path in 4b sets `--health-check-path "/api/health"` instead.) Confirm `containerPort: 8000` matches Dockerfile.backend, and inspect with `aws ecs describe-express-gateway-service --service-arn ...`.
- **Worker exits with `psycopg.OperationalError`** —
  - *Option A:* `DATABASE_URL` is wrong (most often you pasted the pooled URL on port 6543 and the worker needs the direct URL on 5432), or the password contains URL-unsafe characters that weren't percent-encoded.
  - *Option B:* Fargate task's security group isn't in the DB's allowed ingress list, or the DB endpoint isn't reachable from the task's subnet.
- **Supabase: `FATAL: too many connections for role`** — you hit the free tier's ~60 connection ceiling. Either move the backend to the pooled URL (port 6543), drop `WORKER_POLL_SECONDS` (each poll opens then returns a connection — verify the worker uses a connection pool), or upgrade to Pro.
- **Supabase: schema/extension errors on first run** — pgvector extension wasn't enabled in the dashboard. Go to Database → Extensions → search `vector` → toggle on, then redeploy the worker. The app's `ensure_schema()` does *not* `CREATE EXTENSION` (Supabase requires the dashboard toggle for the first install).
- **`/api/snapshot/...` returns 202 forever** — worker isn't draining. Most common cause: ECS service desired-count is 0, or the task is stuck pulling the image. Check `aws ecs describe-services` and the task's CloudWatch logs.
- **Backend image fails to run from Apple Silicon push** — you skipped `--platform linux/amd64` on the `docker buildx` command. The image will be `arm64`-only and Fargate (x86_64 by default) rejects it.
- **SEC EDGAR returns 403 / rate-limit** — `SEC_USER_AGENT` is unset or generic. Set it to a real `Name <email>` string and redeploy the worker.
