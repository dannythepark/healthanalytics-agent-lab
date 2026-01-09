# 🚀 Quick Start: Import to Databricks

## Option 1: Import via Databricks Repos (Recommended)

### Step 1: Initialize Git Repository (Run Locally)

```bash
cd /Users/danny.park/agentic-ai/healthanalytics_agent_lab

# Initialize git repo
git init
git add .
git commit -m "Initial commit: HealthAnalytics AI Clinical Agent Workshop"

# Push to GitHub (replace with your repo URL)
# git remote add origin https://github.com/YOUR_USERNAME/healthanalytics-agent-lab.git
# git push -u origin main
```

### Step 2: Import to Databricks

1. **In Databricks UI:**
   - Click **Repos** in the left sidebar
   - Click **Add Repo**
   - Paste your GitHub URL
   - Click **Create Repo**

2. **Your workshop is now in Databricks!**

---

## Option 2: Manual Upload (Faster for Testing)

### Step 1: Create Workspace Folder

1. In Databricks UI, click **Workspace** → **Users** → **[your email]**
2. Click the dropdown → **Create** → **Folder**
3. Name it: `healthanalytics_agent_lab`

### Step 2: Upload Files

**Upload each file manually:**

1. **Config file:**
   - Right-click `healthanalytics_agent_lab` folder
   - Create → Folder → `00_setup`
   - Upload `00_config.py` and `01_dlt_synthetic_data.py`

2. **Tools:**
   - Create folder `01_create_tools`
   - Upload `01_create_tools.py`

3. **Agent & Eval:**
   - Create folder `02_agent_eval`
   - Upload `agent.py`, `eval_dataset.py`, `driver.py`

4. **Deployment:**
   - Create folder `03_deployment`
   - Upload `deploy_agent.py`

5. **Assets:**
   - Create folder `assets`
   - Upload `architecture_diagram.html`

---

## Step 3: Configure Your Catalog

### CRITICAL: Update Configuration

1. Open `/Workspace/Users/[your-email]/healthanalytics_agent_lab/00_setup/00_config.py`

2. **Change this line:**
   ```python
   CATALOG = "your_catalog_name"  # ← CHANGE THIS!
   ```

3. **To your actual catalog name:**
   ```python
   CATALOG = "main"  # or "healthcare_dev", "workshop", etc.
   ```

   **Don't have a catalog?** Create one:
   ```sql
   CREATE CATALOG IF NOT EXISTS healthcare_workshop;
   USE CATALOG healthcare_workshop;
   ```

4. **Run the config notebook** to verify:
   - Click **Run All**
   - Should see: ✅ Configuration validated successfully!

---

## Step 4: Run the Workshop (In Order!)

### Notebook 1: Generate Data (15 minutes)

**File:** `00_setup/01_dlt_synthetic_data.py`

1. Open the notebook
2. **Important:** This is a DLT notebook, so you need to:
   - Option A: Create a DLT Pipeline (recommended)
   - Option B: Convert to regular notebook (quick test)

#### Option A: Create DLT Pipeline (Production Way)

1. Click **Workflows** → **Delta Live Tables**
2. Click **Create Pipeline**
3. **Settings:**
   - Name: `HealthAnalytics Synthetic Data`
   - Notebook: `/Workspace/.../00_setup/01_dlt_synthetic_data.py`
   - Target: `healthcare_workshop.healthanalytics_ai`
   - Cluster: Serverless
4. Click **Create** → **Start**
5. Wait ~5 minutes for 7 tables to be created

#### Option B: Quick Test (Remove @dlt decorators)

If you want to test quickly without DLT, I can convert this to a standard notebook.
Would you like me to create a non-DLT version?

### Notebook 2: Create Tools (5 minutes)

**File:** `01_create_tools/01_create_tools.py`

1. Open notebook
2. **Run All**
3. Should see: ✅ Successfully created 9 functions

### Notebook 3: Build Agent (5 minutes)

**File:** `02_agent_eval/agent.py`

1. Open notebook
2. **Run All**
3. Test with the Monday morning question
4. Should see agent response in ~60 seconds

### Notebook 4: Evaluation Dataset (2 minutes)

**File:** `02_agent_eval/eval_dataset.py`

1. Open notebook
2. **Run All**
3. Creates 20 evaluation questions

### Notebook 5: Run Evaluation (10 minutes)

**File:** `02_agent_eval/driver.py`

1. Open notebook
2. **Run All**
3. Compares basic vs improved agent
4. View results in MLflow

### Notebook 6: Deploy to Production (5 minutes)

**File:** `03_deployment/deploy_agent.py`

1. Open notebook
2. **Run All**
3. Creates Model Serving endpoint
4. Test the deployed agent

---

## Troubleshooting

### Error: "Catalog 'your_catalog_name' does not exist"

**Fix:** You forgot to update `00_config.py`!

1. Open `00_setup/00_config.py`
2. Change `CATALOG = "your_catalog_name"` to your actual catalog
3. Re-run the config notebook

### Error: "Module 'config' not found"

**Fix:** The imports need the correct path.

In each notebook that has:
```python
import sys
sys.path.append("../00_setup")
from config import CATALOG
```

Change to:
```python
import sys
sys.path.append("/Workspace/Users/[YOUR_EMAIL]/healthanalytics_agent_lab/00_setup")
from config import CATALOG
```

### Error: "DLT decorators not recognized"

**Fix:** You're running a DLT notebook as a standard notebook.

Either:
- Create a proper DLT Pipeline (see Step 4 → Option A)
- Or ask me to create a non-DLT version

### Error: "Foundation model not available"

**Fix:** Your workspace needs access to Foundation Models.

1. Contact your Databricks admin
2. Enable Foundation Model APIs
3. Or change the model in `00_config.py`:
   ```python
   FOUNDATION_MODEL = "databricks-dbrx-instruct"  # Alternative model
   ```

---

## Expected Timeline

| Step | Time | Outcome |
|------|------|---------|
| Import & Config | 5 min | Workspace folder created, config updated |
| Generate Data | 15 min | 7 tables with 233K+ records |
| Create Tools | 5 min | 9 Unity Catalog functions |
| Build Agent | 5 min | Agent created and tested |
| Evaluation | 10 min | Agent performance scored |
| Deployment | 5 min | Production endpoint live |
| **TOTAL** | **45 min** | **End-to-end working demo** |

---

## Verification Checklist

After running all notebooks, verify:

- [ ] 7 tables in `[catalog].healthanalytics_ai`
- [ ] 9 functions in Unity Catalog
- [ ] 2 agents registered in MLflow
- [ ] 1 evaluation dataset table
- [ ] 1 Model Serving endpoint
- [ ] Endpoint responds to test queries

---

## What to Show Your Customers

### 1. The Architecture Diagram
   - Open `assets/architecture_diagram.html` in browser
   - Professional, Databricks-branded visual

### 2. The Data
   ```sql
   USE CATALOG your_catalog;
   USE SCHEMA healthanalytics_ai;

   -- Show recent discharges
   SELECT * FROM get_recent_discharges(7) LIMIT 10;

   -- Show CHF patients
   SELECT * FROM get_diagnoses_by_condition('CHF') LIMIT 10;
   ```

### 3. The Agent in Action
   - Open `02_agent_eval/agent.py`
   - Run the Monday morning question
   - Show the 60-second response

### 4. The Evaluation
   - Open MLflow UI
   - Compare basic vs improved agent
   - Show custom healthcare judges

### 5. The Deployed Endpoint
   - Show Model Serving page
   - Test with REST API
   - Demonstrate production integration

---

## Next Steps

Once working in your Databricks workspace:

1. **Customize for your customer:**
   - Change patient counts
   - Add their specific conditions
   - Use their actual risk models

2. **Extend the demo:**
   - Add more tools
   - Create additional agents
   - Build a Streamlit dashboard

3. **Share with the team:**
   - Export notebooks
   - Create a Git repo
   - Schedule automated reports

---

## Support

- **Documentation:** [README.md](./README.md)
- **Project Context:** [CLAUDE.yaml](./CLAUDE.yaml)
- **Databricks Docs:** https://docs.databricks.com/generative-ai/agent-framework

---

**Ready to transform Maria's Monday morning? Let's go! 🚀**
