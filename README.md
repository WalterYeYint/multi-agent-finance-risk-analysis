
# 🤖 Multi-Agent Finance Risk Analysis

Advanced multi-agent system for comprehensive financial analysis using LangChain, RAG (Retrieval-Augmented Generation), and real-time data. Executes the following:
- multi-agent collaboration: allows each agent to perform sentiment, fundamental, valuation and risk analysis.
- multi-agent debate: Round-robin debate rounds between agents supervised by debate manager for final concensus on financial risks and investment recommendations.

## 🤖 Agent Overview

The system employs 6 specialized agents that work together:

| Agent | Purpose | Key Features |
|-------|---------|--------------|
| **Data Agent** | Market data & news retrieval | Real-time price data, news extraction, content enhancement |
| **Fundamental Agent** | 10-K/10-Q analysis | RAG-powered SEC filing analysis, financial metrics, business insights |
| **Sentiment Agent** | News sentiment analysis | Reflection-enhanced prompting, confidence scoring, investment recommendations |
| **Valuation Agent** | Technical analysis | Volatility calculations, trend analysis, price metrics |
| **Risk Agent** | Risk assessment | VaR calculations, drawdown analysis, risk flags |
| **Writer Agent** | Report generation | Comprehensive markdown reports, final analysis compilation |

## 📋 Prerequisites

- Python 3.8+
- macOS/Linux/Windows
- Internet connection (for stock data)

## 🛠 Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd multi-agent-finance-risk-analysis
```

### 2. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install -U pip
# python -m pip install -r requirements.txt
bash setup.sh
```

### 4. Start the Postgres + pgvector database (RAG store)

The fundamental-analysis RAG stores 10-K/10-Q filings in Postgres with the
`pgvector` extension. A `docker-compose.yml` is provided to run it locally:

```bash
# Start the database in the background
docker compose up -d

# Verify it is running
docker compose ps

# Stop it (data persists in the `pgdata` volume)
docker compose down

# Stop it and wipe all stored filings
docker compose down -v
```

Set the connection string in your `.env` (defaults shown):

```bash
DATABASE_URL=postgresql://finance:finance@localhost:5432/finance_rag
```

The `filings` and `filing_chunks` tables are created automatically on first use.

### 5. Setting up Ollama (Free Local Models)

1. **Install Ollama**:
   ```bash
   # macOS
   brew install ollama
   
   # Linux
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Windows: Download from https://ollama.com/
   ```

2. **Start Ollama service**:
   ```bash
   ollama serve
   ```

3. **Pull a model** (choose based on your system):
   ```bash
    # Lightweight models (good for testing)
    ollama pull llama3.2:1b      # 1.3GB")
    ollama pull llama3.2:3b      # 2.0GB")
    
    # More capable models
    ollama pull llama3.1:8b      # 4.7GB")
    ollama pull llama3.1:70b     # 40GB (requires 64GB+ RAM)")
    ollama pull qwen2.5:7b       # 4.4GB")
    
    # Specialized models
    ollama pull codellama:7b     # 3.8GB (for code)")
    ollama pull mistral:7b       # 4.1GB (general purpose)")
   ```

4. **Use with analysis**:
   ```bash
   export OLLAMA_MODEL=llama3.2:3b  # Optional: specify model
   MODEL_PROVIDER=ollama python -m src.main
   ```

## 🔑 LangSmith & Environment variables Setup

### Step 1: Get LangSmith API Key

1. Go to [https://smith.langchain.com/](https://smith.langchain.com/)
2. Sign up/Login with your account
3. Navigate to **Settings** → **API Keys**
4. Create a new API key
5. Copy the key (starts with `ls_`)

### Step 2: Configure Environment Variables

<!-- #### Option A: Export in Terminal (Temporary)

```bash
export LANGCHAIN_API_KEY="ls_your_api_key_here"
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_PROJECT="Multi-Agent Finance Bot"

export OPENAI_API_KEY=ls_your_api_key_here
# OR use local Ollama (no API key needed)
export MODEL_PROVIDER="ollama"             # Force Ollama usage
export OLLAMA_MODEL="llama3.2:3b"         # Optional: specify 
export POLYGON_API_KEY=ls_your_api_key_here
```

#### Option B: Create .env File (Recommended) -->

Create a `.env` file in your project root:

```bash
# .env
LANGCHAIN_API_KEY=ls_your_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT="Multi-Agent Finance Bot"

OPENAI_API_KEY=ls_your_api_key_here
# OR use local Ollama (no API key needed)
OLLAMA_MODEL="llama3.2:3b"

POLYGON_API_KEY=ls_your_api_key_here
```

Then load it before running:

```bash
source .env
```

<!-- #### Option C: Set in main.py (Code-based)

Edit `src/main.py` and ensure these lines are uncommented:

```python
# LangSmith configuration
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "Multi-Agent Finance Bot")
``` -->

## 🛠 Running the System

```bash
# In one terminal, start ollama service
ollama serve
```

```bash
# In another terminal, activate virtual environment & run backend
source .venv/bin/activate
python -m backend.app

# # Run the multi-agent system
# python -m src.main
```

```bash
# In another terminal, activate virtual environment & run frontend
source .venv/bin/activate
cd frontend
npm start
```
🔸 See pdf files in `sample_outputs/` for sample  outputs.

## Saving 10K/10Q Documents

Filings are added to the RAG store (Postgres + pgvector — make sure the database
is running, see installation step 4) **exclusively from SEC EDGAR**. Downloads
filings straight from the free SEC EDGAR API:

```bash
python -m src.utils.edgar_ingest --tickers AAPL,MSFT,GOOGL --forms 10-K,10-Q --limit 4
```

It is idempotent (filings already stored are skipped by accession number), so it
is safe to re-run or schedule on a cron. Set `SEC_USER_AGENT` in `.env` to a real
contact string (e.g. `"Your Name your-email@example.com"`) — SEC throttles
requests without one.

You usually don't need to run this manually: the **worker** now ingests filings
automatically — on-demand the first time a ticker is analyzed (if it has none
stored), and on a weekly sweep over all tracked tickers for newly-released
filings. Both paths share `SEC_USER_AGENT` and the `EDGAR_FORMS` (default
`10-K,10-Q`) / `EDGAR_LIMIT` (default `4`) / `WORKER_FILING_SCAN_SECONDS`
(default `604800`, i.e. 7 days) tunables.

Extracted filing text is archived under `data/filings_raw/` at runtime
(configurable via `FILINGS_RAW_DIR`); that folder is gitignored and regenerated
on each ingest, and retrieval always reads from Postgres regardless. There is no
local-PDF seed — drop-a-PDF ingestion has been retired in favour of EDGAR.

## Running Integration Test
Integration test can be run to verify how accurate the system's recommendations are.

Run the following:
```bash
# Runs the main program five times with different parameters, generate their outputs and perform test for each (Checks whether the actual recommendation is equal to expected result based on stock's average close price).
source .venv/bin/activate
bash test_program.sh
```

The five test cases are already saved in src/tests/json folder from prev. execution. To run tests using these files without rerunning the main program, run the following instead:
```bash
cd src/tests
pytest integration_test.py -v -s
```

## Running the Streamlit ChatGPT webapp clone
Run the following:
```bash
streamlit run chatgpt_ui.py
```

<!-- ## 📊 What You'll See

### Console Output

The system will display:
- **News Retrieval**: Real-time news fetching with URL content extraction
- **Fundamental Analysis**: RAG-powered analysis of 10-K/10-Q SEC filings with comprehensive insights
- **Valuation Analysis**: Computational metrics (annualized returns, volatility using 252-day formula)
- **Sentiment Analysis**: Reflection-enhanced prompting with confidence scores
- **Risk Metrics**: Traditional financial risk calculations and flags
- **Debug Information**: Data processing steps and agent workflow
- **Final Report**: Complete Markdown risk analysis with all sections
- **Final State**: Complete state object with all agent outputs

### LangSmith Platform (if enabled)

Visit [https://smith.langchain.com/](https://smith.langchain.com/) to see:
- **Project Dashboard**: "Multi-Agent Finance Bot"
- **Execution Traces**: Complete workflow runs
- **Agent Performance**: Timing and success rates
- **Data Flow**: Input/output between agents
- **Debugging**: Detailed execution logs

## � Fundamental Analysis & RAG System

The system includes a sophisticated Fundamental Analysis Agent that uses RAG (Retrieval-Augmented Generation) to analyze 10-K and 10-Q SEC filings.

### Features

- **Document Ingestion**: Custom tools to ingest PDF financial documents
- **Vector Search**: ChromaDB-powered semantic search across financial filings
- **Agent-Based Analysis**: LangChain agent with RAG tools for comprehensive analysis
- **Batch Queries**: Efficient single-call analysis with multiple queries
- **Sample Data**: Pre-loaded sample data for AAPL, MSFT, and GOOGL

### Quick Start with Sample Data

```bash
# Run analysis (pulls filings from SEC EDGAR on demand, stores chunks in Postgres)
python -m src.main
```

### What the Fundamental Agent Analyzes

The agent performs comprehensive analysis including:
- **Executive Summary**: High-level company overview
- **Key Financial Metrics**: Revenue, margins, profitability metrics
- **Business Highlights**: Core business segments and products
- **Risk Assessment**: Identified risk factors from filings
- **Competitive Position**: Market positioning analysis
- **Growth Prospects**: Future outlook and opportunities
- **Financial Health Score**: Numerical rating (0-10)
- **Investment Thesis**: Comprehensive investment recommendation

### RAG System Architecture

1. **Document Ingestion**: PDF documents are loaded and chunked
2. **Vector Storage**: ChromaDB stores embeddings for semantic search
3. **Query Processing**: Natural language queries are converted to vector searches
4. **Context Retrieval**: Relevant document chunks are retrieved
5. **LLM Analysis**: OpenAI models analyze context to provide insights

### Advanced Usage

#### Custom Queries
The system supports sophisticated queries like:
- "What are the main revenue streams and their growth rates?"
- "Analyze the risk factors and their potential impact"
- "Compare business segments and their profitability"
- "What regulatory challenges does the company face?"

#### Programmatic Access
```python
from src.rag_utils import FundamentalRAG

# Initialize RAG system
rag = FundamentalRAG()

# Query documents
results = rag.retrieve_relevant_chunks("AAPL", "revenue growth analysis")

# Process results
for chunk in results:
    print(chunk.page_content)
``` -->

## ☁️ Deployment (AWS)

Full deployment guide: **[AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md)** · architecture diagram: [AWS_DEPLOYMENT.drawio.xml](AWS_DEPLOYMENT.drawio.xml).

**Shape** (cost-optimized "Option B-lite-lite", ~$41/mo idle):

| Component | Hosted on |
|---|---|
| Flask backend (reads / enqueues) | ECS Express Mode (Fargate + managed ALB + HTTPS URL) |
| Worker (runs the chain + debate pipeline) | ECS Fargate (continuous) |
| Postgres + pgvector (snapshots, jobs, filings) | Supabase (free tier) or RDS `t4g.micro` |
| React frontend | S3 + CloudFront |

**Deploys are CI-driven**, split into two workflows by what changed:
- [`.github/workflows/build-and-push.yml`](.github/workflows/build-and-push.yml) — on push touching `backend/**`, `src/**`, or the Dockerfiles: `build → smoke-test → deploy`. Builds both images, pushes to ECR, smoke-tests against a throwaway Postgres, then creates/updates the Express service. The worker is redeployed in the same job.
- [`.github/workflows/deploy-frontend.yml`](.github/workflows/deploy-frontend.yml) — on push touching `frontend/**`: `npm build → s3 sync → CloudFront invalidation`. Separate so a UI tweak doesn't trigger an 8-minute backend Docker build.

**One-time manual setup** (the parts CI can't or shouldn't do for you — full commands in [AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md)):

1. **Database** — create the Supabase project, enable the `vector` extension, grab `DATABASE_URL` (Step 1).
2. **ECR repos** — `aws ecr create-repository` for `finance-agents-backend` and `finance-agents-worker` (Step 2).
3. **Secrets Manager** — store `DATABASE_URL`, `OPENAI_API_KEY`, `POLYGON_API_KEY`; grant the execution role `GetSecretValue` (Step 3.5).
4. **Deploy IAM user** — attach ECR + ECS Express + `iam:PassRole` permissions to the user whose keys are in GitHub (Step 3.6a). *(This is the `ecs:DescribeServices ... not authorized` fix.)*
5. **Service-linked roles** — `create-service-linked-role` for ECS, ELB, and app-autoscaling, once per fresh account (Step 3.6b). *(This is the `Unable to assume the service linked role` fix.)*
6. **IAM roles** — `ecsTaskExecutionRole` + `ecsInfrastructureRoleForExpressServices` (Step 4a).
7. **GitHub repo secrets (backend/worker)** — `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `ECS_EXEC_ROLE_ARN`, `ECS_INFRA_ROLE_ARN`, `DATABASE_URL_SECRET_ARN`, `OPENAI_KEY_SECRET_ARN`, `POLYGON_KEY_SECRET_ARN` (Step 4b).
8. **Frontend** — deploy once manually (S3 bucket + CloudFront with the two-origins setup, Step 6a–6c); add S3 + `cloudfront:CreateInvalidation` perms to the deploy user; add repo secrets `FRONTEND_S3_BUCKET` + `CLOUDFRONT_DISTRIBUTION_ID` (Step 6d). After that, `frontend/**` pushes auto-deploy.

Once these are done, every push to `main` deploys (backend/worker via one workflow, frontend via the other). The `.env` file is local-dev only — it is **not** shipped to AWS; runtime config is injected via the ECS task definition and Secrets Manager.

## 📚 Additional Documentation

- **LangSmith Traces**: Monitor agent execution at [smith.langchain.com](https://smith.langchain.com/)

## 🔧 Troubleshooting

### Common Issues

#### Fundamental Agent Issues
- **"OpenAI API key not set"**: Ensure `OPENAI_API_KEY` is set in your environment
- **"ChromaDB errors"**: Delete `./data/chroma_db/` and reinitialize with sample data
- **"Agent timeout"**: Increase `max_iterations` in the fundamental agent configuration

#### General Issues
- **Import errors**: Ensure virtual environment is activated and dependencies are installed
- **No news data**: Check `POLYGON_API_KEY` is set and valid
- **LangSmith not working**: Verify `LANGCHAIN_API_KEY` and `LANGCHAIN_TRACING_V2=true`

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details., technical analysis, and risk assessment.