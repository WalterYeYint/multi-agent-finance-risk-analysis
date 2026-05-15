# AWS Deployment Guide: Cost Optimized on App Runner

This outlines deploying the Multi-Agent Finance Risk Analysis platform to AWS extremely affordably using **AWS App Runner** and **Amazon S3 + CloudFront**. By avoiding an EC2 or multi-container Elastic Beanstalk cluster, you can heavily optimize startup and idle costs while maintaining a managed, production-ready environment.

## Cost Saving Strategy Architecture

1. **Frontend (S3 + CloudFront)**: Serverless static hosting charges mere pennies per month. Pay strictly by request/traffic.
2. **Backend (AWS App Runner)**: Managed container engine that *pauses instances* when there is no incoming traffic, eliminating persistent CPU uptime and charging only a tiny fraction for provisioned memory while paused. We avoid deploying the React application in docker alongside it specifically to keep the compute workload clean.

---

## Step 1: Deploy Backend to AWS App Runner

1. **Build and push the Docker image to Amazon ECR:**
   ```bash
   # Create a repository
   aws ecr create-repository --repository-name finance-app-backend
   
   # Authenticate Docker to ECR
   aws ecr get-login-password | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com
   
   # Build the backend-only image
   docker build -f Dockerfile.backend -t finance-app-backend .
   
   # Tag and Push
   docker tag finance-app-backend:latest YOUR_ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/finance-app-backend:latest
   docker push YOUR_ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/finance-app-backend:latest
   ```

2. **Create the App Runner Service:**
   - Go to the AWS App Runner Console -> **Create an App Runner service**.
   - Input the **Amazon ECR** image URI for `finance-app-backend`.
   - **Service settings**:
     - *CPU & Memory*: Pick **1 vCPU, 2 GB** (Minimum footprint compute size to minimize costs).
     - *Environment variables*: Supply your keys (`OPENAI_API_KEY`, `FLASK_ENV=production`).
     - *Auto-scaling configuration*: Modify Max concurrency/Max instances to scale constraints downward, preventing rapid scaling charges if bombarded.

3. **Get Endpoint**: After provisioning, record your AWS service URL (e.g., `https://xxxxx.REGION.awsapprunner.com`).

---

## Step 2: Deploy Frontend Serverless (S3 -> CloudFront)

1. **Point React to App Runner**: 
   Inside `frontend/src/App.jsx` (or `.env`), update API requests to point to your new backend URL.

2. **Build the Application natively**:
   ```bash
   cd frontend
   npm run build
   ```

3. **Upload to S3**:
   - Create a new S3 Bucket (e.g., `finance-ui-demo`).
   - Enable "Static Website Hosting".
   - Upload your `/build` folder.

4. **CloudFront**:
   Set up Amazon CloudFront forwarding requests to S3, bypassing direct internet access to S3 objects. This ensures free SSL Certificates (ACM) and caching.

## Summary of Costs
- **S3 / CloudFront**: ~$0.01 to $0.10/month depending on HTTP caching/traffic.
- **AWS App Runner**: Paused state costs ~$0.007/GB-hour. Active compute is billed strictly for the minutes it executes requests.
- **Estimated Idle Demo usage**: < $2-4.00 / month.