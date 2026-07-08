#!/usr/bin/env bash
#
# I19 — Replace long-lived CI IAM-user keys with GitHub OIDC + a short-lived role.
#
# Run this ONCE with an AWS admin identity (`aws configure` / SSO). It:
#   1. Registers GitHub Actions as an OIDC identity provider in your account
#      (idempotent — skipped if it already exists).
#   2. Creates an IAM role that GitHub Actions can assume via OIDC, trusted ONLY
#      for this repo (no static keys, credentials expire when the job ends).
#   3. Attaches the same permissions the old CI IAM user had:
#        - AmazonEC2ContainerRegistryPowerUser  (build/push + pull images)
#        - inline ECSExpressDeploy               (deploy job — mirrors Step 3.6a)
#        - inline FrontendDeploy                 (S3 sync + CloudFront invalidate)
#
# After it prints the role ARN:
#   - Add it as the repo secret AWS_OIDC_ROLE_ARN
#     (GitHub → Settings → Secrets and variables → Actions).
#   - DELETE the old AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY repo secrets, and
#     delete the CI IAM user's access keys in AWS (they are no longer used).
#
# Re-running is safe: the provider/role/policies are created-or-updated in place.
set -euo pipefail

# ---- Config (override via env) -----------------------------------------------
GITHUB_ORG="${GITHUB_ORG:-WalterYeYint}"
GITHUB_REPO="${GITHUB_REPO:-multi-agent-finance-risk-analysis}"
ROLE_NAME="${ROLE_NAME:-github-actions-oidc-deploy}"
# Restrict which git refs may assume the role. Default `*` = any branch/tag in
# THIS repo (needed because build-and-push runs on workflow_dispatch from any
# branch; the deploy job itself is still gated to main inside the workflow).
# Tighten to a single branch with:  SUBJECT_REF='ref:refs/heads/main'
SUBJECT_REF="${SUBJECT_REF:-*}"
# Optional: scope the frontend-deploy policy. Leave empty to use "*".
FRONTEND_S3_BUCKET="${FRONTEND_S3_BUCKET:-}"
CLOUDFRONT_DISTRIBUTION_ID="${CLOUDFRONT_DISTRIBUTION_ID:-}"

AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
OIDC_HOST="token.actions.githubusercontent.com"
OIDC_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:oidc-provider/${OIDC_HOST}"
SUBJECT="repo:${GITHUB_ORG}/${GITHUB_REPO}:${SUBJECT_REF}"

echo "Account:  ${AWS_ACCOUNT_ID}"
echo "Repo:     ${GITHUB_ORG}/${GITHUB_REPO}"
echo "Role:     ${ROLE_NAME}"
echo "Trust sub: ${SUBJECT}"
echo

# ---- 1. OIDC provider (idempotent) -------------------------------------------
if aws iam get-open-id-connect-provider --open-id-connect-provider-arn "$OIDC_ARN" >/dev/null 2>&1; then
  echo "✓ OIDC provider already registered"
else
  echo "→ registering GitHub OIDC provider…"
  # AWS validates the GitHub OIDC endpoint against its trusted-CA library; the
  # thumbprints below are still required by the API but are no longer the trust
  # anchor. Both current GitHub values are supplied.
  aws iam create-open-id-connect-provider \
    --url "https://${OIDC_HOST}" \
    --client-id-list "sts.amazonaws.com" \
    --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1 1c58a3a8518e8759bf075b76b750d4f2df264fca >/dev/null
  echo "✓ OIDC provider created"
fi

# ---- 2. Role + repo-scoped trust policy --------------------------------------
TRUST_POLICY="$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Federated": "${OIDC_ARN}" },
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": { "${OIDC_HOST}:aud": "sts.amazonaws.com" },
      "StringLike":   { "${OIDC_HOST}:sub": "${SUBJECT}" }
    }
  }]
}
EOF
)"

if aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  echo "→ role exists — updating trust policy…"
  aws iam update-assume-role-policy --role-name "$ROLE_NAME" --policy-document "$TRUST_POLICY"
else
  echo "→ creating role…"
  aws iam create-role \
    --role-name "$ROLE_NAME" \
    --assume-role-policy-document "$TRUST_POLICY" \
    --description "GitHub Actions OIDC deploy role for ${GITHUB_ORG}/${GITHUB_REPO} (I19)" \
    --max-session-duration 3600 >/dev/null
fi
echo "✓ role trust policy set"

# ---- 3a. ECR (build/push/pull) -----------------------------------------------
aws iam attach-role-policy --role-name "$ROLE_NAME" \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser
echo "✓ attached AmazonEC2ContainerRegistryPowerUser"

# ---- 3b. ECS Express + PassRole (deploy job — mirrors AWS_DEPLOYMENT.md 3.6a) -
aws iam put-role-policy --role-name "$ROLE_NAME" --policy-name ECSExpressDeploy \
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
echo "✓ put inline policy ECSExpressDeploy"

# ---- 3c. Frontend deploy (S3 sync + CloudFront invalidate) --------------------
if [ -n "$FRONTEND_S3_BUCKET" ]; then
  S3_RES="[\"arn:aws:s3:::${FRONTEND_S3_BUCKET}\",\"arn:aws:s3:::${FRONTEND_S3_BUCKET}/*\"]"
else
  echo "  ⚠ FRONTEND_S3_BUCKET unset — S3 actions scoped to \"*\" (tighten later)."
  S3_RES="\"*\""
fi
if [ -n "$CLOUDFRONT_DISTRIBUTION_ID" ]; then
  CF_RES="\"arn:aws:cloudfront::${AWS_ACCOUNT_ID}:distribution/${CLOUDFRONT_DISTRIBUTION_ID}\""
else
  echo "  ⚠ CLOUDFRONT_DISTRIBUTION_ID unset — CloudFront invalidation scoped to \"*\" (tighten later)."
  CF_RES="\"*\""
fi
aws iam put-role-policy --role-name "$ROLE_NAME" --policy-name FrontendDeploy \
  --policy-document "$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "S3SyncFrontend",
      "Effect": "Allow",
      "Action": ["s3:ListBucket", "s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
      "Resource": ${S3_RES}
    },
    {
      "Sid": "CloudFrontInvalidate",
      "Effect": "Allow",
      "Action": "cloudfront:CreateInvalidation",
      "Resource": ${CF_RES}
    }
  ]
}
EOF
)"
echo "✓ put inline policy FrontendDeploy"

echo
echo "=============================================================="
echo "Done. Add this as the repo secret AWS_OIDC_ROLE_ARN:"
echo
echo "  arn:aws:iam::${AWS_ACCOUNT_ID}:role/${ROLE_NAME}"
echo
echo "Then DELETE the old repo secrets AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY"
echo "and delete the CI IAM user's access keys in AWS."
echo "=============================================================="
