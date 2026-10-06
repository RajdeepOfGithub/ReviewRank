# Deploying ReviewRank on AWS (free-tier-minded runbook)

**Status: NOT deployed.** The AWS CLI on the development machine has no credentials configured, the Docker image has not been built (Docker Desktop was not running), and the model is a candidate until the Stage 7 freeze. Describe the project as deployed on AWS only after the steps below are done and the live URL has been checked.

## What AWS says (read from official pages on 2026-09-30)

- Free plan for new accounts: up to $200 in credits ($100 immediately, up to $100 more for exploring), valid 6 months; the account **closes on its own after 6 months or when credits run out**; no charge unless you convert to a Paid plan. Source: https://aws.amazon.com/free/
- Lambda free tier: 1 million requests and 400,000 GB-seconds per month, not time-limited. Memory 128 MB to 10,240 MB; 512 MB ephemeral storage included. Source: https://aws.amazon.com/lambda/pricing/
- ECR private repositories: $0.10 per GB-month; free tier 500 MB for one year, new ECR customers only. Source: https://aws.amazon.com/ecr/pricing/
- Not verified: whether Lambda Function URLs add any charge (the pricing page does not say). Check before relying on it.
- Consequence: on a Free-plan account a live demo is **temporary**. Keep the README, tests, Docker image, recorded demo output and latency numbers as the lasting evidence.

## Before any command

1. Decide which account/plan you are on (Free plan, Paid plan, or an older account). Pick a region (for example `us-east-1`).
2. Create an IAM user or SSO profile with only the permissions needed (ECR, Lambda, IAM role pass for Lambda, Budgets, CloudWatch Logs). Configure it yourself with `aws configure --profile reviewrank` (type the keys into the terminal yourself; never paste them into a chat or a file in this repo).
3. Create a **budget alarm first** (for example $5 with an email alert) in the Billing console or with `aws budgets create-budget`.

## Build and test locally

```bash
docker build -t reviewrank:dev .            # local image (port 8000)
docker run --rm -p 8000:8000 reviewrank:dev
curl localhost:8000/health
docker build -f Dockerfile.lambda -t reviewrank:lambda .   # Lambda image (adapter, port 8080)
docker images reviewrank                    # record the image size in 03_METRICS_AND_DECISIONS.md
```

## Push to ECR and create the function (sketch; adjust names and the account id)

```bash
aws ecr create-repository --repository-name reviewrank --profile reviewrank
aws ecr get-login-password --profile reviewrank | docker login --username AWS --password-stdin <ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com
docker tag reviewrank:lambda <ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com/reviewrank:lambda
docker push <ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com/reviewrank:lambda
aws lambda create-function --function-name reviewrank --package-type Image \
  --code ImageUri=<ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com/reviewrank:lambda \
  --role arn:aws:iam::<ACCOUNT>:role/<lambda-execution-role> --memory-size 1024 --timeout 15 --profile reviewrank
aws lambda put-function-concurrency --function-name reviewrank --reserved-concurrent-executions 2 --profile reviewrank
aws lambda create-function-url-config --function-name reviewrank --auth-type NONE --profile reviewrank
```

A public Function URL (`NONE` auth) can be called by anyone: keep reserved concurrency low, keep the budget alarm, and delete it after the demo. Use `--auth-type AWS_IAM` if you do not need it public.

## Check, record, tear down

1. `curl <function-url>/health`, then `POST /rank` with a sample request. Record cold-start and warm latency (several calls) and the image size in the ledger.
2. Record in `03_METRICS_AND_DECISIONS.md` (AWS / cost record): date, region, resource names, free-tier terms read (URL + date), budget alarm, spend.
3. Tear down when done: `aws lambda delete-function-url-config`, `aws lambda delete-function`, `aws ecr delete-repository --force`, and confirm in the console that nothing is left.

## What can go wrong

- Cold starts: the first request loads the model (about 760 KB) and Python libraries; measure it rather than guessing.
- Image size near the ECR 500 MB free tier; cost beyond that is a few cents per month.
- A Free-plan account closes after 6 months; export anything you need before then.
