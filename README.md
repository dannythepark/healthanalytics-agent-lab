# HealthAnalytics AI: Clinical Agent Workshop

**From 4-6 hours to 60 seconds: Clinical insights at the speed of care**

![Workshop Type](https://img.shields.io/badge/Workshop-Healthcare%20Analytics-red)
![Databricks](https://img.shields.io/badge/Platform-Databricks-orange)
![HIPAA](https://img.shields.io/badge/Compliance-HIPAA-blue)
![Llama](https://img.shields.io/badge/Model-Llama%203.1%2070B-green)

---

## 📖 The Story

### Meet HealthAnalytics AI

**HealthAnalytics AI** is the internal data science team at a regional health system serving **500,000+ patients** across **12 hospitals and 80+ clinics**. They're responsible for clinical analytics, population health insights, and operational reporting.

### The Problem: Maria's Monday Morning

Every Monday morning, the executive team asks the same critical question:

> *"Which patients are at highest risk of readmission this week, and what should our care coordinators prioritize?"*

**Meet Maria**, a senior clinical analyst with 10 years of healthcare analytics experience and an RN background. This question requires her to manually integrate data across:

- **EHR (Electronic Health Records)** - demographics, diagnoses, vitals
- **Claims data** - utilization patterns, costs
- **ADT (Admit/Discharge/Transfer) feeds** - recent discharges
- **Social determinants of health** - housing, transportation barriers
- **Care management notes** - previous interventions

### Before the Agent

Maria's typical Monday morning (4-6 hours):

1. ☕ Pull recent discharges from the ADT system (30 minutes)
2. 📊 Join with diagnosis codes and comorbidity scores (1 hour)
3. 🔍 Cross-reference against prior 30-day readmissions (45 minutes)
4. 📝 Check care management notes for social risk factors (1 hour)
5. 🧮 Calculate risk scores and prioritize by care coordinator capacity (1.5 hours)
6. 📈 Build an Excel report for the care coordination team (45 minutes)

**Result:** Report delivered Monday afternoon. Care coordinators start their week without guidance.

Maria reflects:
> *"I know exactly what they're going to ask, but I still have to rebuild the analysis every week. By the time I finish, it's Monday afternoon and coordinators have already started their day."*

### The Breakthrough: Clinical Analytics Agent

The HealthAnalytics AI team built a specialized **Clinical Analytics Agent** that can answer this question in **under 60 seconds** by orchestrating multiple data sources and applying clinical logic.

### After the Agent

Monday morning transformation (60 seconds):

1. 💬 VP asks the question in natural language
2. 🤖 Agent orchestrates 10 tool calls across 6 data sources
3. ✅ Actionable report delivered by 8am Monday morning

**Time saved:** 4-6 hours → 60 seconds
**Business impact:** Care coordinators can prioritize high-risk patients immediately

---

## 🎯 Workshop Objectives

In this hands-on lab, you'll learn how to:

1. **Build a production-ready clinical analytics agent** using Databricks AI Agents
2. **Create reusable analytics tools** (SQL functions + Python models) in Unity Catalog
3. **Evaluate agent performance** using custom healthcare-specific judges
4. **Deploy agents to Model Serving** for production use
5. **Maintain HIPAA compliance** by keeping all data within Databricks

---

## 🏗️ Architecture

![Architecture Diagram](./assets/architecture_diagram.html)

### Data Layer (Synthetic HIPAA-Compliant Data)

| Table | Records | Purpose | Key Fields |
|-------|---------|---------|------------|
| `patients` | 10,000 | Patient demographics | patient_id, mrn, age, gender, zip_code |
| `encounters` | 50,000 | Inpatient admissions & ED visits | encounter_id, admission_date, discharge_date, facility |
| `diagnoses` | 150,000 | Clinical diagnoses (ICD-10) | diagnosis_id, icd10_code, description, is_primary |
| `readmissions` | 5,000 | All-cause readmissions | readmission_id, days_between, is_30_day |
| `risk_scores` | 10,000 | Predictive risk scores | patient_id, risk_score, risk_factors |
| `sdoh` | 8,000 | Social determinants of health | housing_instability, transportation_barrier |
| `care_coordinators` | 15 | Care team capacity | coordinator_id, current_caseload, specialties |

### Tools Layer (9 Clinical Analytics Functions)

**SQL Functions (6):**
- `get_recent_discharges(days_back)` - Recent discharges
- `get_diagnoses_by_condition(condition)` - Filter by diagnosis (CHF, COPD, etc.)
- `get_patient_readmission_history(patient_id)` - Readmission history
- `get_risk_score(patient_id)` - Predictive risk score
- `get_sdoh_barriers(patient_id)` - Social barriers
- `get_available_care_coordinators(specialty)` - Coordinator capacity

**Python Functions (3):**
- `calculate_priority_score(patient_id)` - Composite risk scoring
- `assign_care_coordinator(patient_list, specialty)` - Caseload balancing
- `generate_outreach_report(patient_list)` - Markdown reports

### Agent Layer

- **Model:** Llama 3.1 70B Instruct (Databricks Foundation Model)
- **Framework:** Databricks AI Agents SDK
- **Deployment:** Model Serving with Serverless Compute
- **Evaluation:** MLflow with custom healthcare judges

---

## 🚀 Lab Structure

```
healthanalytics_agent_lab/
├── README.md                        # This file
├── CLAUDE.yaml                      # Project context
├── 00_setup/
│   ├── 00_config.py                 # User configuration
│   └── 01_dlt_synthetic_data.py     # DLT pipeline for synthetic data
├── 01_create_tools/
│   └── 01_create_tools.py           # SQL + Python function definitions
├── 02_agent_eval/
│   ├── agent.py                     # Agent definition and prompts
│   ├── eval_dataset.py              # Evaluation Q&A pairs
│   └── driver.py                    # Evaluation execution
├── 03_deployment/
│   └── deploy_agent.py              # Model registration and serving
└── assets/
    └── architecture_diagram.html    # Interactive diagram
```

---

## 📋 Prerequisites

### Databricks Requirements
- **Workspace:** AWS, Azure, or GCP
- **DBR:** 14.3 LTS or higher
- **Unity Catalog:** Enabled with catalog creation permissions
- **Model Serving:** Serverless or GPU-enabled clusters
- **Foundational Models API:** Access to Llama 3.1 70B

### Python Packages
```python
# Installed in DLT pipeline
faker
databricks-agents
mlflow
pydantic
```

### Knowledge
- Basic SQL and Python
- Familiarity with Databricks notebooks
- Understanding of healthcare terminology (helpful but not required)

---

## 🛠️ Setup Instructions

### Step 1: Clone or Import This Lab

```bash
# Option 1: Clone to Databricks Repos
git clone <your-repo-url>

# Option 2: Upload as Workspace folder
# Use Databricks UI to import the folder
```

### Step 2: Configure Your Catalog

Open `00_setup/00_config.py` and update:

```python
# REQUIRED: Replace with your Unity Catalog name
CATALOG = "your_catalog_name"  # e.g., "healthcare_analytics"

# The schema will be auto-created
SCHEMA = "healthanalytics_ai"
```

### Step 3: Generate Synthetic Data

Run `00_setup/01_dlt_synthetic_data.py` to create tables:

1. Open the notebook in Databricks
2. Click **Run All**
3. Wait for DLT pipeline to complete (~5 minutes)
4. Verify 7 tables created in `<CATALOG>.healthanalytics_ai`

**What gets created:**
- ✅ 10,000 synthetic patients (HIPAA-compliant, no real PHI)
- ✅ 127 patients discharged in last 7 days (planted demo data)
- ✅ 23 CHF patients, 19 COPD patients
- ✅ 32 patients with prior 30-day readmissions
- ✅ 18 high-risk patients (risk score > 0.7)

### Step 4: Create Analytics Tools

Run `01_create_tools/01_create_tools.py`:

1. Open the notebook
2. Execute all cells
3. Verify 9 functions registered in Unity Catalog

**What gets created:**
- ✅ 6 SQL functions for data retrieval
- ✅ 3 Python functions for risk scoring and assignment
- ✅ All functions callable by the agent

---

## 🎓 Lab Exercises

### Exercise 1: Build Your First Clinical Agent (30 min)

**File:** `02_agent_eval/agent.py`

**Tasks:**
1. Review the agent definition and system prompt
2. Understand how tools are wired to the agent
3. Test the agent with a simple query: *"How many patients were discharged in the last 7 days?"*
4. Observe the tool calls and response

**Key Learning:**
- How Databricks agents orchestrate tools
- System prompts for healthcare domains
- Tool calling patterns

### Exercise 2: Run the Demo Scenario (15 min)

**File:** `02_agent_eval/agent.py`

**The Monday Morning Question:**
```
Show me high-risk patients discharged in the last 7 days who need
outreach this week. Focus on CHF and COPD patients with prior readmissions.
```

**Expected Agent Behavior:**
1. ✅ Calls `get_recent_discharges(7)` → 127 patients
2. ✅ Calls `get_diagnoses_by_condition("CHF")` → 23 patients
3. ✅ Calls `get_diagnoses_by_condition("COPD")` → 19 patients
4. ✅ Calls `get_patient_readmission_history()` → 32 with history
5. ✅ Calls `get_risk_score()` → 18 high-risk patients
6. ✅ Calls `calculate_priority_score()` → Composite scores
7. ✅ Calls `get_available_care_coordinators("CHF/COPD")` → 2 available
8. ✅ Calls `assign_care_coordinator()` → Balanced assignments
9. ✅ Calls `generate_outreach_report()` → Final report

**Success Criteria:**
- Agent completes in <90 seconds
- Returns 18 high-risk patients
- Includes care coordinator assignments
- Highlights social barriers (transportation, housing)

### Exercise 3: Evaluate Agent Performance (30 min)

**Files:** `02_agent_eval/eval_dataset.py` and `02_agent_eval/driver.py`

**Tasks:**
1. Review the evaluation dataset (20 clinical Q&A pairs)
2. Run the evaluation with the **basic prompt**
3. Review results from custom judges:
   - Clinical Accuracy
   - Completeness
   - Actionability
   - HIPAA Compliance
4. Improve the system prompt
5. Re-run evaluation and compare results

**Key Learning:**
- How to create healthcare-specific evaluation datasets
- Custom judge implementation for clinical criteria
- Prompt engineering for clinical accuracy
- MLflow tracking for agent iterations

### Exercise 4: Deploy to Production (20 min)

**File:** `03_deployment/deploy_agent.py`

**Tasks:**
1. Register the agent to Unity Catalog as a model
2. Deploy to a Model Serving endpoint
3. Test the endpoint with a REST API call
4. Simulate a production query from a care management dashboard

**Key Learning:**
- Agent registration and versioning
- Model Serving for real-time inference
- Production endpoint configuration
- Monitoring and logging

---

## 🎬 Demo Script (For Presenters)

### Introduction (5 min)

**Slide 1: The Problem**
> "Meet Maria, a senior clinical analyst at a regional health system. Every Monday morning,
> she spends 4-6 hours manually building a readmission risk report. By the time she's done,
> it's Monday afternoon and care coordinators have already started their week without guidance."

**Slide 2: The Solution**
> "With Databricks AI Agents, Maria's team built a Clinical Analytics Agent that does the
> same analysis in 60 seconds. Let's see it in action."

### Live Demo (10 min)

**Step 1: Show the Data** (2 min)
```sql
-- Quick preview of the healthcare data
SELECT * FROM healthcare_analytics.healthanalytics_ai.encounters
WHERE discharge_date >= CURRENT_DATE - 7
LIMIT 10;

SELECT COUNT(*) as chf_patients
FROM healthcare_analytics.healthanalytics_ai.diagnoses
WHERE description LIKE '%Heart Failure%';
```

**Step 2: Show the Tools** (2 min)
```sql
-- Demonstrate a SQL function
SELECT * FROM healthcare_analytics.healthanalytics_ai.get_recent_discharges(7);

-- Demonstrate a Python function
SELECT calculate_priority_score('patient-001');
```

**Step 3: Run the Agent** (5 min)
```python
# Open 02_agent_eval/agent.py
# Run the Monday morning question

query = """
Show me high-risk patients discharged in the last 7 days who need
outreach this week. Focus on CHF and COPD patients with prior readmissions.
"""

response = agent.query(query)
print(response)
```

**Step 4: Show the Results** (1 min)
- Highlight: 18 patients identified
- Point out: Care coordinator assignments
- Emphasize: 60 seconds vs. 4-6 hours

### Discussion (10 min)

**Key Points to Cover:**
1. **HIPAA Compliance:** All data stays in Databricks, no external API calls
2. **Reusable Tools:** Functions can be used across multiple agents
3. **Clinical Accuracy:** Custom evaluation judges for healthcare
4. **Scalability:** From demo to production in minutes
5. **Extensibility:** Easy to add new data sources and tools

**Common Questions:**
- *"Can we use our own risk models?"* → Yes, swap the Python function
- *"How do we handle real PHI?"* → Unity Catalog + audit logs
- *"What about cost?"* → Databricks Foundation Models, predictable pricing
- *"Can we integrate with Epic/Cerner?"* → Yes, via DLT streaming

---

## 📊 Evaluation Results

### Custom Judges

| Judge | Criteria | Scoring |
|-------|----------|---------|
| **Clinical Accuracy** | Correct risk thresholds, ICD-10 coding, 30-day readmission logic | 1-5 scale |
| **Completeness** | All 6 data sources checked, no missed tools | 1-5 scale |
| **Actionability** | Specific patient list, coordinator assignments, timeline | 1-5 scale |
| **HIPAA Compliance** | No external calls, data stays in Databricks | Pass/Fail |

### Sample Results

**Basic Prompt:**
- Clinical Accuracy: 3.2/5
- Completeness: 3.8/5
- Actionability: 3.5/5
- HIPAA Compliance: Pass

**Improved Prompt (after refinement):**
- Clinical Accuracy: 4.7/5
- Completeness: 4.9/5
- Actionability: 4.8/5
- HIPAA Compliance: Pass

---

## 🔒 HIPAA Compliance & Security

### How This Demo Maintains Compliance

✅ **Synthetic Data Only:** All patient data is generated using Faker library
✅ **No External API Calls:** Agent uses only Databricks Foundation Models
✅ **Unity Catalog Governance:** All data access is audited and governed
✅ **No PHI Exposure:** Clearly labeled as synthetic data for demo purposes

### Production Considerations

For production deployments with real PHI:

1. **Enable Unity Catalog audit logs** for all data access
2. **Use Databricks Secrets** for any external system credentials
3. **Implement row-level security** for patient data access
4. **Configure encryption at rest and in transit** (Databricks default)
5. **Establish data retention policies** per HIPAA requirements
6. **Conduct regular access reviews** for care coordinators and analysts
7. **Document all data flows** for HIPAA compliance audits

---

## 🎓 Learning Outcomes

By the end of this workshop, you'll be able to:

- [x] Build production-ready AI agents for healthcare analytics
- [x] Create reusable tools (SQL + Python) in Unity Catalog
- [x] Evaluate agents using custom healthcare-specific metrics
- [x] Deploy agents to Model Serving for real-time inference
- [x] Maintain HIPAA compliance in all agent workflows
- [x] Apply agent patterns to other clinical use cases

---

## 🌟 Next Steps

### Additional Use Cases to Explore

1. **Emergency Department Optimization**
   - *"Which ED patients are at risk of admission and could benefit from observation?"*

2. **Population Health Management**
   - *"Identify diabetic patients overdue for HbA1c testing in ZIP codes with food deserts"*

3. **Financial Analytics**
   - *"Which high-cost patients would benefit from case management to reduce utilization?"*

4. **Clinical Quality Metrics**
   - *"Generate our monthly CMS quality measures report with drill-down by facility"*

### Productionizing Your Agent

1. **Integrate with Real Data Sources**
   - Connect DLT to Epic/Cerner HL7 feeds
   - Stream ADT data into Delta tables
   - Join with claims data warehouses

2. **Build a Care Management Dashboard**
   - Streamlit or Dash app calling the agent endpoint
   - Daily scheduled reports via Databricks Jobs
   - Slack integration for care coordinator alerts

3. **Implement Validated Risk Models**
   - Replace `calculate_priority_score()` with HOSPITAL score
   - Integrate LACE index for readmission risk
   - Add diagnosis-specific risk models (CHF, COPD, sepsis)

4. **Scale to Multiple Agents**
   - Create specialized agents per department
   - Build agent orchestration workflows
   - Implement agent-to-agent communication

---

## 🙏 Acknowledgments

This lab is inspired by real-world healthcare analytics challenges faced by regional health systems. Special thanks to:

- Clinical analysts who spend hours on manual reporting every week
- Care coordinators who need actionable insights at the start of their day
- Healthcare data teams building the future of clinical intelligence

**HealthAnalytics AI is a fictional team, but the problem they solved is very real.**

---

## 📚 Resources

### Databricks Documentation
- [AI Agents Developer Guide](https://docs.databricks.com/en/generative-ai/agent-framework/index.html)
- [Unity Catalog Functions](https://docs.databricks.com/en/sql/language-manual/sql-ref-functions-udf.html)
- [Delta Live Tables](https://docs.databricks.com/en/delta-live-tables/index.html)
- [Model Serving](https://docs.databricks.com/en/machine-learning/model-serving/index.html)

### Healthcare Analytics
- [CMS 30-Day Readmission Measures](https://www.cms.gov/medicare/quality/initiatives/hospital-quality-initiative/outcome-measures)
- [Social Determinants of Health (SDOH)](https://health.gov/healthypeople/priority-areas/social-determinants-health)
- [ICD-10 Code Reference](https://www.icd10data.com/)

### Workshop Support
- **GitHub Issues:** [Report bugs or request features]
- **Databricks Community:** [Ask questions and share learnings]

---

## 📄 License

This workshop is provided for educational purposes. All synthetic data is generated and contains no real patient information.

**No PHI. No PII. 100% Synthetic. 100% HIPAA-Compliant Demo.**

---

**Built with ❤️ by the Databricks Healthcare Solutions Team**

*Transforming clinical analytics from hours to seconds, one agent at a time.*
