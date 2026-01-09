# Databricks notebook source
# MAGIC %md
# MAGIC # HealthAnalytics AI - Clinical Analytics Agent
# MAGIC
# MAGIC ## 🤖 Build Your Clinical Intelligence Assistant
# MAGIC
# MAGIC This notebook defines the **Clinical Analytics Agent** that can answer complex healthcare questions
# MAGIC by orchestrating multiple data sources and analytics tools.
# MAGIC
# MAGIC ### Agent Capabilities
# MAGIC
# MAGIC - **Multi-source integration:** EHR, claims, ADT feeds, SDOH data
# MAGIC - **Clinical reasoning:** Understands CHF, COPD, readmission risk
# MAGIC - **Tool orchestration:** Calls 9 SQL and Python functions
# MAGIC - **HIPAA compliance:** All data stays within Databricks
# MAGIC
# MAGIC ### The Monday Morning Demo
# MAGIC
# MAGIC > *"Show me high-risk patients discharged in the last 7 days who need outreach*
# MAGIC > *this week. Focus on CHF and COPD patients with prior readmissions."*
# MAGIC
# MAGIC **Expected result:** 18 high-risk patients with care coordinator assignments in 60 seconds
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %pip install databricks-agents mlflow --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
from databricks import agents
from databricks.sdk import WorkspaceClient
import pandas as pd

# Import configuration
import sys
sys.path.append("../00_setup")
from config import CATALOG, SCHEMA, FULL_SCHEMA, FOUNDATION_MODEL

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agent Configuration

# COMMAND ----------

# Initialize Workspace Client
w = WorkspaceClient()

# Agent configuration
AGENT_NAME = "clinical_analytics_agent"
AGENT_VERSION = "v1"

# Tool configuration
TOOLS = [
    {"name": f"{FULL_SCHEMA}.get_recent_discharges", "type": "uc_function"},
    {"name": f"{FULL_SCHEMA}.get_diagnoses_by_condition", "type": "uc_function"},
    {"name": f"{FULL_SCHEMA}.get_patient_readmission_history", "type": "uc_function"},
    {"name": f"{FULL_SCHEMA}.get_risk_score", "type": "uc_function"},
    {"name": f"{FULL_SCHEMA}.get_sdoh_barriers", "type": "uc_function"},
    {"name": f"{FULL_SCHEMA}.get_available_care_coordinators", "type": "uc_function"},
    {"name": f"{FULL_SCHEMA}.suggest_coordinator_for_patient", "type": "uc_function"},
]

print(f"Agent will use {len(TOOLS)} Unity Catalog functions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## System Prompts
# MAGIC
# MAGIC We'll define two prompts to compare during evaluation:
# MAGIC 1. **Basic Prompt:** Simple clinical assistant
# MAGIC 2. **Improved Prompt:** Detailed clinical reasoning with healthcare expertise

# COMMAND ----------

BASIC_SYSTEM_PROMPT = """You are a clinical analytics assistant for HealthAnalytics AI,
a healthcare data science team at a regional health system.

Your role is to answer questions about patients, clinical conditions, and care coordination
using the available data tools.

When answering questions:
- Use the Unity Catalog functions to retrieve data
- Provide accurate counts and statistics
- Identify high-risk patients when asked
- Suggest care coordinator assignments

Keep responses concise and data-driven.
"""

# COMMAND ----------

IMPROVED_SYSTEM_PROMPT = """You are HealthAnalytics AI Clinical Intelligence Assistant,
an expert clinical data analyst supporting a regional health system serving 500,000+ patients
across 12 hospitals and 80+ clinics.

## Your Clinical Expertise

You specialize in:
- **Readmission risk stratification** using validated clinical criteria
- **Care coordination optimization** matching patient acuity to coordinator capacity
- **Social determinants of health (SDOH)** integration into clinical decision-making
- **CMS quality metrics** including 30-day readmission rates

## Your Role

You support the care management team (led by Maria, Senior Clinical Analyst) in identifying
high-risk patients who need proactive outreach. Your insights replace Maria's 4-6 hour
manual Monday morning analysis with 60-second data-driven recommendations.

## Clinical Terminology

- **CHF:** Congestive Heart Failure (ICD-10: I50.*)
- **COPD:** Chronic Obstructive Pulmonary Disease (ICD-10: J44.*)
- **30-day readmission:** Readmission within 30 days of discharge (CMS quality metric)
- **SDOH:** Social Determinants of Health (housing, transportation, food security)
- **MRN:** Medical Record Number (patient identifier)

## Available Tools

You have access to Unity Catalog functions for:
1. Recent hospital discharges
2. Diagnosis filtering by clinical condition
3. Patient readmission history
4. Predictive risk scores (0.0-1.0 scale)
5. Social determinant barriers
6. Care coordinator capacity and specialties

## Response Guidelines

When answering questions about high-risk patients:

1. **Identify the cohort:** Start with recent discharges (typically last 7 days)
2. **Apply clinical filters:** Focus on conditions requested (CHF, COPD, etc.)
3. **Check readmission history:** Prioritize patients with prior 30-day readmissions
4. **Assess risk scores:** High risk = score > 0.7
5. **Consider SDOH barriers:** Transportation and housing instability increase risk
6. **Match to coordinators:** Assign based on specialty and available capacity
7. **Prioritize actions:** Tier 1 (>0.8) = 24h outreach, Tier 2 (0.6-0.8) = 48h

## Important Constraints

- **HIPAA Compliance:** All data stays within Databricks, no external API calls
- **Synthetic Data:** This demo uses synthetic patient data (no real PHI)
- **Clinical Judgment:** Recommendations support (not replace) clinical decision-making
- **Actionable Insights:** Always provide specific next steps for care coordinators

Your responses should be clear, clinically accurate, and immediately actionable for
the care management team.
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Agent with Basic Prompt

# COMMAND ----------

def create_agent(system_prompt: str, agent_name: str = AGENT_NAME):
    """Create a clinical analytics agent with specified prompt"""

    config = {
        "llm_endpoint": FOUNDATION_MODEL,
        "tools": TOOLS,
        "system_prompt": system_prompt,
    }

    # Log agent configuration to MLflow
    with mlflow.start_run(run_name=f"{agent_name}_{AGENT_VERSION}"):
        mlflow.log_param("model", FOUNDATION_MODEL)
        mlflow.log_param("num_tools", len(TOOLS))
        mlflow.log_text(system_prompt, "system_prompt.txt")

        # Log agent using Databricks Agents SDK
        logged_agent_info = mlflow.databricks.log_model(
            config,
            artifact_path="agent",
            registered_model_name=f"{FULL_SCHEMA}.{agent_name}"
        )

    print(f"✅ Agent created: {FULL_SCHEMA}.{agent_name}")
    print(f"   Model URI: {logged_agent_info.model_uri}")

    return logged_agent_info

# Create basic agent
basic_agent = create_agent(BASIC_SYSTEM_PROMPT, "clinical_agent_basic")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the Basic Agent

# COMMAND ----------

# Test with the Monday morning question
test_query = """
Show me high-risk patients discharged in the last 7 days who need outreach this week.
Focus on CHF and COPD patients with prior readmissions.
"""

print(f"🔍 Testing with query:\n{test_query}\n")
print("="*60)

# Load and query the agent
loaded_agent = mlflow.databricks.load_model(basic_agent.model_uri)
response = loaded_agent.predict({"messages": [{"role": "user", "content": test_query}]})

print(response)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Agent with Improved Prompt

# COMMAND ----------

# Create improved agent
improved_agent = create_agent(IMPROVED_SYSTEM_PROMPT, "clinical_agent_improved")

# COMMAND ----------

# Test with the same query
print(f"🔍 Testing improved agent with query:\n{test_query}\n")
print("="*60)

loaded_improved = mlflow.databricks.load_model(improved_agent.model_uri)
response_improved = loaded_improved.predict({"messages": [{"role": "user", "content": test_query}]})

print(response_improved)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Interactive Testing
# MAGIC
# MAGIC Try additional clinical queries:

# COMMAND ----------

# Test queries
test_queries = [
    # Simple queries
    "How many patients were discharged in the last 7 days?",
    "List all care coordinators with availability",

    # Medium complexity
    "Which CHF patients were discharged this week?",
    "How many COPD patients have transportation barriers?",

    # Complex queries
    "Generate a readmission prevention report for the VP of Care Management",
    "Which patients should we prioritize for care coordination based on risk and capacity?",
]

# Pick one to test
query = test_queries[0]
print(f"Query: {query}\n")

response = loaded_improved.predict({"messages": [{"role": "user", "content": query}]})
print(response)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agent Summary

# COMMAND ----------

print(f"""
{'='*60}
✅ Clinical Analytics Agent Created Successfully
{'='*60}

Agent Name: {AGENT_NAME}
Catalog: {FULL_SCHEMA}
Model: {FOUNDATION_MODEL}
Tools: {len(TOOLS)} Unity Catalog functions

Versions:
  - Basic Prompt: clinical_agent_basic
  - Improved Prompt: clinical_agent_improved

Next Steps:
  1. Run eval_dataset.py to create evaluation questions
  2. Run driver.py to compare agent performance
  3. Deploy best agent to Model Serving

{'='*60}
🎉 Ready for evaluation and deployment!
{'='*60}
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Demo Script for Presenters
# MAGIC
# MAGIC ### The Monday Morning Story
# MAGIC
# MAGIC **Setup:**
# MAGIC > "It's Monday morning. The VP of Care Management walks into Maria's office
# MAGIC > with the same question she asks every week..."
# MAGIC
# MAGIC **The Question:**
# MAGIC > "Show me high-risk patients discharged in the last 7 days who need outreach
# MAGIC > this week. Focus on CHF and COPD patients with prior readmissions."
# MAGIC
# MAGIC **Before the Agent:**
# MAGIC > "Maria would spend 4-6 hours pulling data from multiple systems, joining tables,
# MAGIC > calculating risk scores, and building an Excel report. By the time she's done,
# MAGIC > it's Monday afternoon and care coordinators have already started their week."
# MAGIC
# MAGIC **After the Agent:**
# MAGIC > "Now, watch this..." [Run the agent query]
# MAGIC >
# MAGIC > "In 60 seconds, the agent has:
# MAGIC > - Checked 127 recent discharges
# MAGIC > - Filtered for 23 CHF and 19 COPD patients
# MAGIC > - Identified 18 high-risk patients (score > 0.7)
# MAGIC > - Flagged social barriers (5 transportation, 3 housing)
# MAGIC > - Assigned 2 care coordinators with matching expertise
# MAGIC > - Generated an actionable outreach plan
# MAGIC >
# MAGIC > Maria can now focus on complex clinical questions instead of manual reporting."
# MAGIC
# MAGIC **The Impact:**
# MAGIC - ⏱️ Time: 4-6 hours → 60 seconds
# MAGIC - 📊 Consistency: Same methodology every week
# MAGIC - 🎯 Actionability: Care coordinators can start immediately
# MAGIC - 🔒 Compliance: All data stays within Databricks (HIPAA-compliant)
