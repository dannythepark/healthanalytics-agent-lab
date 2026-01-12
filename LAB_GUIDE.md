---
noteId: "7fef9490effc11f0a5d80ba1419066d8"
tags: []

---

# Healthcare Analytics AI Agent - Lab Guide

## Overview
Build an AI agent that helps care coordinators identify high-risk patients and prioritize follow-up actions. You'll use SQL/Python tools, Unity Catalog, and LLMs to create a production-ready healthcare analytics assistant.

---

## Part 1: Build Your Healthcare Agent

### 1.0 Setup - Generate Synthetic Healthcare Data
**Notebook:** `00_setup/01_dlt_synthetic_data.py`

Run this notebook to create synthetic healthcare tables:
- **`patients`** - 10,000 patient records with demographics and risk scores
- **`hospital_visits`** - 50,000 encounters with diagnoses and outcomes
- **`care_coordinators`** - 15 care team members with capacity and specialties

**⚠️ Important:** This is 100% synthetic data (no real PHI)

**Time:** ~3 minutes

---

### 1.1 Build Tools - SQL and Python Functions
**Notebook:** `01_create_tools/01_create_tools.py`

Write queries to access data critical for patient care coordination:

#### Core SQL Functions:
1. **`get_high_risk_patients(days_back)`**
   - Find recently discharged patients with high readmission risk
   - Use case: Daily outreach list generation

2. **`get_patients_by_condition(condition, days_back)`**
   - Search for patients with specific diagnoses (CHF, COPD, Diabetes)
   - Use case: Disease-specific care programs

3. **`find_available_coordinators(specialty)`**
   - Match patients to care coordinators based on specialty and capacity
   - Use case: Workload balancing and patient assignment

4. **`calculate_readmission_rate(days_back)`**
   - Analyze 30-day readmission rates by diagnosis
   - Use case: Quality metrics and reporting

#### Python Functions:
- **`prioritize_followup_list(patient_data)`**
  - Rank patients by urgency using risk score + barriers + history
  - Returns: Sorted list with recommended action timeline

These SQL functions are easy to call from within a notebook or an agent. We can also register Python functions for data manipulation outside of SQL.

**Explore Unity Catalog:**
- Navigate to Unity Catalog to see where your functions landed
- Path: `<your_catalog>.healthanalytics_ai.*`
- This is a common governance layer for Data, Functions, and Agents

**Time:** ~10 minutes

---

### 1.2 Integrate with an LLM [AI Playground]
**Combine Tools & LLM**

Use the **Databricks AI Playground** to bring together your SQL/Python tools and the Language Model (LLM).

**Configuration:**
- **Model:** `databricks-meta-llama-3-1-70b-instruct`
- **Tools:** `<your_catalog>.healthanalytics_ai.*`
- **Backup Tools:** `healthcare_lab.demo.*` (Use if you can't find your tools)

**System Prompt:**
```
You are a healthcare analytics assistant helping care coordinators identify
and prioritize high-risk patients for follow-up. Use the available tools to
answer questions about patient risk, readmission rates, and care coordination.

Always provide specific patient names, risk scores, and recommended actions.
```

**Time:** ~5 minutes

---

### 1.3 Test the Agent [AI Playground]

#### Test Questions:

**Question 1:** *"Which CHF patients discharged in the last 7 days are at highest risk for readmission?"*

**Expected Behavior:**
- Agent calls `get_patients_by_condition('CHF', 7)`
- Agent calls `get_high_risk_patients(7)`
- Combines results and ranks by risk score
- Returns: Patient names, risk scores, discharge dates

---

**Question 2:** *"I need to assign a care coordinator to high-risk COPD patients. Who has capacity?"*

**Expected Behavior:**
- Agent calls `get_patients_by_condition('COPD', 7)`
- Agent calls `find_available_coordinators('COPD')`
- Matches patients to available coordinators
- Returns: Coordinator names, current caseload, patient recommendations

---

**Question 3:** *"What's our 30-day readmission rate for heart failure patients?"*

**Expected Behavior:**
- Agent calls `calculate_readmission_rate(30)`
- Filters for CHF diagnosis
- Returns: Percentage, total encounters, comparison to benchmark

---

**Explore MLflow Traces:**
- Inspect agent runs in MLflow to understand how each tool is being called
- Review: Tool selection, parameters, results, reasoning chain

**Time:** ~15 minutes

---

## Part 2: Agent Evaluation & Improvement

### 2.0 Setup Evaluation Environment
**Notebook:** `02_agent_eval/driver.py`

This section focuses on systematic testing and improvement of your healthcare agent.

---

### 2.1 Define Agent and Retriever Tool

#### Inspect Agent.py
**File:** `02_agent_eval/agent.py`

We've defined a tool-calling Agent - explore the building blocks:
- **System Prompt:** How the agent understands its role
- **Tool Definitions:** Which functions are available
- **Response Format:** How results are structured

**Key Areas to Understand:**
- Where tools are being implemented (Unity Catalog functions)
- How to tweak tool descriptions for better agent performance
- When to add/remove tools based on use cases

---

#### Policy Knowledge Retrieval (Vector Search)

**Pre-staged Vector Search Index:**
- **Path:** `healthcare_lab.knowledge.clinical_guidelines_index`
- **Content:** Hospital policies, clinical protocols, CMS guidelines
- **Use Case:** Agent can look up readmission criteria, discharge protocols, care coordination standards

**Create Retriever Function:**
```python
def get_clinical_guidelines(query: str) -> str:
    """
    Search clinical guidelines and hospital policies.

    Args:
        query: Natural language question about policies or protocols

    Returns:
        Relevant excerpts from clinical documentation
    """
    results = vector_search_index.search(query, k=3)
    return format_results(results)
```

Wrap this Vector Search Index into a function your LLM can call to look up policy information.

**Time:** ~10 minutes

---

### 2.2 Define Evaluation Dataset
**File:** `02_agent_eval/eval_dataset.py`

#### Use Provided Dataset

We've created 20 evaluation questions covering common care coordination scenarios:

**Example Evaluation Cases:**

| Question | Expected Tools | Ground Truth |
|----------|---------------|--------------|
| "Show me high-risk CHF patients discharged yesterday" | `get_patients_by_condition`, `get_high_risk_patients` | Returns 3-5 patients with risk > 0.7 |
| "What's our readmission rate this month?" | `calculate_readmission_rate` | Returns percentage around 15-18% |
| "Who can take on new diabetes patients?" | `find_available_coordinators` | Returns coordinators with capacity |
| "According to our policy, when should we follow up with high-risk patients?" | `get_clinical_guidelines` | Returns "within 48 hours of discharge" |

**Dataset Structure:**
```python
eval_data = [
    {
        "request": "Which patients need urgent follow-up today?",
        "expected_tools": ["get_high_risk_patients", "prioritize_followup_list"],
        "expected_facts": ["risk_score > 0.7", "discharge within 48 hours"],
        "difficulty": "medium"
    },
    # ... 19 more examples
]
```

#### (Optional) Experiment with Synthetic Data Generation
Use MLflow to generate additional evaluation cases based on your specific use cases.

**Time:** ~10 minutes

---

### 2.3 Evaluate Agent
**Notebook:** `02_agent_eval/driver.py`

#### Run MLflow.evaluate()

```python
import mlflow

# Evaluate agent performance
results = mlflow.evaluate(
    model=agent,
    data=eval_dataset,
    model_type="databricks-agent",
    evaluators="default"
)
```

**What Gets Measured:**

1. **Tool Selection Accuracy**
   - Did the agent call the right tools?
   - Judge: Compare actual tools vs expected tools

2. **Response Quality**
   - Is the answer accurate and helpful?
   - Judge: LLM-based evaluation against ground truth

3. **Clinical Safety**
   - Does the response follow healthcare guidelines?
   - Judge: Custom evaluator checking for policy compliance

4. **Completeness**
   - Did the agent include all relevant patient details?
   - Judge: Checks for required fields (names, risk scores, dates)

**MLflow Evaluation UI:**
- View pass/fail rate for each question
- Drill into individual traces to see reasoning
- Compare scores across evaluation runs

**Time:** ~10 minutes

---

### 2.4 Refine and Re-Evaluate

#### Improve Prompt
**File:** `02_agent_eval/agent.py` (line 45)

A second, improved prompt has been provided. Let's enable it:

**Original Prompt:**
```python
system_prompt = """
You are a healthcare analytics assistant. Use tools to answer questions.
"""
```

**Improved Prompt (remove # to enable):**
```python
# system_prompt = """
# You are a healthcare analytics assistant helping care coordinators
# prioritize patient follow-up.
#
# CRITICAL GUIDELINES:
# - Always include patient names, MRNs, and risk scores in responses
# - For high-risk patients (>0.7), recommend contact within 48 hours
# - For patients with barriers (transportation, housing), flag for social work
# - When showing readmission rates, compare to 15% national benchmark
# - If asked about policies, use get_clinical_guidelines() tool
#
# RESPONSE FORMAT:
# 1. Summary: Brief answer to the question
# 2. Patient List: Table with names, risk scores, recommended actions
# 3. Next Steps: Specific action items for care coordinators
# """
```

**What Changed:**
- ✅ Clearer role definition
- ✅ Specific guidelines for risk thresholds
- ✅ Structured response format
- ✅ Explicit instructions to use retrieval tool

---

#### Re-run Evaluation

```python
# Start new MLflow run with improved prompt
with mlflow.start_run(run_name="agent_v2_improved_prompt"):
    results_v2 = mlflow.evaluate(
        model=agent_v2,  # Updated agent with new prompt
        data=eval_dataset,
        model_type="databricks-agent",
        evaluators="default"
    )
```

**Compare Results in MLflow UI:**

| Metric | v1 (Original) | v2 (Improved) | Change |
|--------|---------------|---------------|---------|
| Tool Selection Accuracy | 75% | 92% | +17% ✅ |
| Response Quality | 3.2/5 | 4.5/5 | +1.3 ✅ |
| Clinical Safety | 85% | 98% | +13% ✅ |
| Completeness | 60% | 88% | +28% ✅ |

**Observe Performance Gains:**
- More consistent tool selection
- Better-formatted responses
- Improved adherence to clinical guidelines

**Time:** ~10 minutes

---

### 2.5 Register Agent into Unity Catalog

With our Agent complete, we'll register it into Unity Catalog alongside our functions.

```python
from mlflow.models import ModelSignature

# Register agent
mlflow.register_model(
    model_uri=f"runs:/{run.info.run_id}/agent",
    name=f"{catalog}.{schema}.clinical_analytics_agent",
    tags={
        "use_case": "care_coordination",
        "model_type": "databricks-agent",
        "tools": "healthanalytics_ai.*"
    }
)
```

**Benefits:**
- Apply same governance as functions (permissions, lineage, versioning)
- Track agent versions alongside evaluation metrics
- Enable deployment to model serving

**Explore in Unity Catalog:**
- Navigate to `<your_catalog>.healthanalytics_ai.clinical_analytics_agent`
- View: Model versions, lineage, permissions, serving status

**Time:** ~5 minutes

---

### 2.6 Explore Deployed Model Serving Endpoint

#### Pre-Deployed Healthcare Agent

We've pre-deployed an endpoint on your workspace:
- **Endpoint:** `healthanalytics-agent-endpoint`
- **Model:** Same agent you just built and evaluated
- **Tools:** All Unity Catalog functions enabled

#### Test Deployed Endpoint:

**Via REST API:**
```python
import requests

response = requests.post(
    f"{workspace_url}/serving-endpoints/healthanalytics-agent-endpoint/invocations",
    headers={"Authorization": f"Bearer {token}"},
    json={
        "messages": [
            {
                "role": "user",
                "content": "Show me high-risk patients from yesterday"
            }
        ]
    }
)
```

**Via Databricks UI:**
1. Navigate to **Serving** → **Endpoints**
2. Select `healthanalytics-agent-endpoint`
3. Use the built-in query UI to test
4. Leave feedback using thumbs up/down buttons

**Observe:**
- Real-time inference latency (~2-3 seconds)
- Tool calls in trace logs
- Token usage and cost tracking

**Time:** ~10 minutes

---

## Next Steps

### Leave Lab Feedback
We'd love to know how we can improve! Please leave feedback in our survey:
**[Survey Link]**

### Extend Your Agent
**Add More Tools:**
- Lab results retrieval API
- Medication history lookup
- Appointment scheduling integration
- Claims data analysis

**Enhance Vector Search:**
- Add clinical literature (UpToDate, PubMed)
- Include payer-specific policies
- Index state/federal regulations

### Production Deployment
**Best Practices:**
1. **CI/CD Pipeline:**
   - Automated testing on every agent change
   - Evaluation gates before deployment
   - Rollback capabilities

2. **Monitoring:**
   - Track tool usage patterns
   - Alert on low-quality responses
   - Monitor latency and errors

3. **Model Versioning:**
   - A/B test prompt improvements
   - Champion/challenger deployment
   - Automatic rollback on quality degradation

4. **Human-in-the-Loop:**
   - Collect care coordinator feedback
   - Use feedback to improve eval dataset
   - Periodic review of agent decisions

---

## Troubleshooting

### Can't Find Your Tools?
Use backup tools: `healthcare_lab.demo.*`

### Agent Not Calling Tools?
1. Check tool descriptions are clear
2. Verify permissions in Unity Catalog
3. Test tools individually in SQL editor

### Evaluation Failing?
1. Verify eval dataset format
2. Check ground truth facts are achievable
3. Review traces for tool errors

### Endpoint Not Responding?
1. Check endpoint status (should be "Ready")
2. Verify authentication token
3. Review serving logs for errors

---

## Summary

**What You Built:**
✅ 4 SQL tools for healthcare analytics
✅ 1 Python function for patient prioritization
✅ 1 vector search retriever for clinical guidelines
✅ 1 production-ready AI agent
✅ Evaluation framework with 20 test cases
✅ Deployed model serving endpoint

**Key Learnings:**
- How to design tools that LLMs can use effectively
- Systematic agent evaluation with MLflow
- Prompt engineering for healthcare applications
- Governance with Unity Catalog
- Production deployment patterns

**Time Investment:** ~90 minutes total

---

## Additional Resources

- **Databricks Agent Docs:** [Link to docs]
- **Unity Catalog Guide:** [Link to guide]
- **MLflow Evaluate API:** [Link to API docs]
- **Healthcare AI Best Practices:** [Link to whitepaper]
