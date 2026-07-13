# deploy/iam — IAM policy documents

> IAM policy JSON supports **no comments** and must be **pure printable ASCII**
> (`create-role` rejects anything else), so these files are bare policies and
> the documentation lives here instead.

## Bedrock task role (I19/N3): `financeAgentsTaskRole`

The **task role** the backend/worker CONTAINER assumes at runtime to call
Bedrock — distinct from `ecsTaskExecutionRole`, which only pulls the image and
reads secrets at startup. Attach it as the task definition's `taskRoleArn`
(AWS_DEPLOYMENT.md, "Using AWS Bedrock" section).

- **`ecs-task-trust.json`** — trust policy: lets the ECS tasks service
  principal (`ecs-tasks.amazonaws.com`) assume the role.
- **`bedrock-task-role-policy.json`** — least-privilege permissions:
  `bedrock:InvokeModel` + `InvokeModelWithResponseStream` only, scoped to
  `anthropic.*` foundation models and inference profiles. Tighten `Resource`
  to the exact model IDs you invoke before production use.

Apply (from this directory, so the `file://` paths resolve):

```bash
aws iam create-role --role-name financeAgentsTaskRole \
  --assume-role-policy-document file://ecs-task-trust.json

aws iam put-role-policy --role-name financeAgentsTaskRole \
  --policy-name BedrockInvoke \
  --policy-document file://bedrock-task-role-policy.json

export TASK_ROLE_ARN=$(aws iam get-role --role-name financeAgentsTaskRole \
  --query 'Role.Arn' --output text)
```

Then add `"taskRoleArn": "<TASK_ROLE_ARN>"` to `task-worker.json`, register a
new revision, and roll the service (AWS_DEPLOYMENT.md Step 5d).

## GitHub OIDC deploy role (I19)

- **`setup-github-oidc.sh`** — idempotent script that creates the GitHub OIDC
  identity provider + the `github-actions-oidc-deploy` role CI assumes
  (replaces static IAM-user keys). See AWS_DEPLOYMENT.md Step 3.6a.
