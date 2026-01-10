# Databricks notebook source
# MAGIC %md
# MAGIC # HealthAnalytics AI - Evaluation Dataset
# MAGIC
# MAGIC ## 📝 Clinical Analytics Q&A Pairs
# MAGIC
# MAGIC This notebook creates an evaluation dataset with **20 clinical questions** across three complexity levels:
# MAGIC
# MAGIC - **Simple (6 questions):** Single-source factual retrieval
# MAGIC - **Medium (8 questions):** Multi-source filtering and joining
# MAGIC - **Complex (6 questions):** Multi-step reasoning with recommendations
# MAGIC
# MAGIC Each question includes:
# MAGIC - **request:** The clinical question
# MAGIC - **expected_facts:** Key facts that must appear in the answer
# MAGIC - **expected_tools:** Tools the agent should use
# MAGIC
# MAGIC ---

# COMMAND ----------

import pandas as pd
import json

# COMMAND ----------

# MAGIC %md
# MAGIC ## Simple Questions (6)
# MAGIC
# MAGIC Single data source, factual retrieval.

# COMMAND ----------

simple_questions = [
    {
        "request": "How many patients were discharged in the last 7 days?",
        "expected_facts": ["127", "discharged", "last 7 days"],
        "expected_tools": ["get_recent_discharges"],
        "complexity": "simple"
    },
    {
        "request": "List all care coordinators with available capacity",
        "expected_facts": ["care coordinator", "available", "capacity"],
        "expected_tools": ["get_available_care_coordinators"],
        "complexity": "simple"
    },
    {
        "request": "How many CHF patients are in the system?",
        "expected_facts": ["CHF", "Congestive Heart Failure", "patients"],
        "expected_tools": ["get_diagnoses_by_condition"],
        "complexity": "simple"
    },
    {
        "request": "Show me all COPD patients in the database",
        "expected_facts": ["COPD", "Chronic Obstructive Pulmonary Disease"],
        "expected_tools": ["get_diagnoses_by_condition"],
        "complexity": "simple"
    },
    {
        "request": "What is the risk score for patient-000001?",
        "expected_facts": ["risk score", "patient-000001"],
        "expected_tools": ["get_risk_score"],
        "complexity": "simple"
    },
    {
        "request": "Show me recent discharges from the last 3 days",
        "expected_facts": ["discharged", "3 days", "recent"],
        "expected_tools": ["get_recent_discharges"],
        "complexity": "simple"
    },
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Medium Questions (8)
# MAGIC
# MAGIC Multiple data sources, filtering and joining.

# COMMAND ----------

medium_questions = [
    {
        "request": "Which CHF patients were discharged in the last 7 days?",
        "expected_facts": ["CHF", "discharged", "7 days", "23"],
        "expected_tools": ["get_recent_discharges", "get_diagnoses_by_condition"],
        "complexity": "medium"
    },
    {
        "request": "How many COPD patients have transportation barriers?",
        "expected_facts": ["COPD", "transportation", "barrier"],
        "expected_tools": ["get_diagnoses_by_condition", "get_sdoh_barriers"],
        "complexity": "medium"
    },
    {
        "request": "Which care coordinators specialize in CHF and have availability?",
        "expected_facts": ["CHF", "care coordinator", "available", "2"],
        "expected_tools": ["get_available_care_coordinators"],
        "complexity": "medium"
    },
    {
        "request": "Show me patients with risk scores above 0.7",
        "expected_facts": ["risk score", "0.7", "high risk", "18"],
        "expected_tools": ["get_risk_score"],
        "complexity": "medium"
    },
    {
        "request": "How many patients have both CHF and prior readmissions?",
        "expected_facts": ["CHF", "readmission", "prior"],
        "expected_tools": ["get_diagnoses_by_condition", "get_patient_readmission_history"],
        "complexity": "medium"
    },
    {
        "request": "What are the social barriers for high-risk CHF patients?",
        "expected_facts": ["CHF", "social", "barriers", "SDOH", "transportation", "housing"],
        "expected_tools": ["get_diagnoses_by_condition", "get_risk_score", "get_sdoh_barriers"],
        "complexity": "medium"
    },
    {
        "request": "Which patients discharged last week have housing instability?",
        "expected_facts": ["discharged", "housing", "instability", "3"],
        "expected_tools": ["get_recent_discharges", "get_sdoh_barriers"],
        "complexity": "medium"
    },
    {
        "request": "How many patients with 30-day readmissions are in the recent discharge cohort?",
        "expected_facts": ["30-day", "readmission", "discharged", "32"],
        "expected_tools": ["get_recent_discharges", "get_patient_readmission_history"],
        "complexity": "medium"
    },
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Complex Questions (6)
# MAGIC
# MAGIC Multi-step reasoning, clinical logic, actionable recommendations.

# COMMAND ----------

complex_questions = [
    {
        "request": """Show me high-risk patients discharged in the last 7 days who need
        outreach this week. Focus on CHF and COPD patients with prior readmissions.""",
        "expected_facts": [
            "127", "discharged", "7 days",
            "23", "CHF", "19", "COPD",
            "32", "readmission",
            "18", "high-risk", "0.7",
            "transportation", "housing",
            "2", "care coordinator",
            "outreach", "priority"
        ],
        "expected_tools": [
            "get_recent_discharges",
            "get_diagnoses_by_condition",
            "get_patient_readmission_history",
            "get_risk_score",
            "get_sdoh_barriers",
            "get_available_care_coordinators",
            "calculate_priority_score",
            "generate_outreach_report"
        ],
        "complexity": "complex"
    },
    {
        "request": """Which patients should we prioritize for care coordination this week
        based on risk scores and coordinator capacity?""",
        "expected_facts": [
            "risk score", "high-risk", "priority",
            "care coordinator", "capacity",
            "assignment", "specialty"
        ],
        "expected_tools": [
            "get_risk_score",
            "get_available_care_coordinators",
            "suggest_coordinator_for_patient",
            "calculate_priority_score"
        ],
        "complexity": "complex"
    },
    {
        "request": """Generate a Monday morning readmission prevention report for the
        VP of Care Management including patient prioritization and coordinator assignments.""",
        "expected_facts": [
            "readmission", "prevention",
            "high-risk", "priority",
            "CHF", "COPD",
            "care coordinator", "assignment",
            "outreach", "recommendations"
        ],
        "expected_tools": [
            "get_recent_discharges",
            "get_diagnoses_by_condition",
            "get_risk_score",
            "get_patient_readmission_history",
            "get_sdoh_barriers",
            "get_available_care_coordinators"
        ],
        "complexity": "complex"
    },
    {
        "request": """Identify CHF patients with multiple risk factors (prior readmission,
        social barriers, high risk score) and assign them to appropriate coordinators.""",
        "expected_facts": [
            "CHF", "risk factors",
            "readmission", "social barriers",
            "risk score", "high-risk",
            "care coordinator", "assignment", "CHF"
        ],
        "expected_tools": [
            "get_diagnoses_by_condition",
            "get_patient_readmission_history",
            "get_sdoh_barriers",
            "get_risk_score",
            "get_available_care_coordinators",
            "suggest_coordinator_for_patient"
        ],
        "complexity": "complex"
    },
    {
        "request": """What interventions should we prioritize for COPD patients with both
        clinical risk (readmissions) and social risk factors (transportation, housing)?""",
        "expected_facts": [
            "COPD", "interventions",
            "readmission", "clinical risk",
            "social risk", "transportation", "housing",
            "priority", "outreach"
        ],
        "expected_tools": [
            "get_diagnoses_by_condition",
            "get_patient_readmission_history",
            "get_sdoh_barriers",
            "get_risk_score"
        ],
        "complexity": "complex"
    },
    {
        "request": """Create an outreach plan for high-risk patients considering care
        coordinator capacity, specialty matching, and patient acuity levels.""",
        "expected_facts": [
            "high-risk", "outreach plan",
            "care coordinator", "capacity", "specialty",
            "acuity", "assignment",
            "priority", "timeline"
        ],
        "expected_tools": [
            "get_risk_score",
            "get_recent_discharges",
            "get_available_care_coordinators",
            "suggest_coordinator_for_patient"
        ],
        "complexity": "complex"
    },
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Combine All Questions

# COMMAND ----------

# Combine all questions
all_questions = simple_questions + medium_questions + complex_questions

print(f"✅ Created {len(all_questions)} evaluation questions:")
print(f"   - Simple: {len(simple_questions)}")
print(f"   - Medium: {len(medium_questions)}")
print(f"   - Complex: {len(complex_questions)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Convert to MLflow Evaluation Format

# COMMAND ----------

# Convert to pandas DataFrame for MLflow
eval_df = pd.DataFrame([
    {
        "request": q["request"],
        "expected_facts": json.dumps(q["expected_facts"]),
        "expected_tools": json.dumps(q["expected_tools"]),
        "complexity": q["complexity"],
        "request_id": f"q_{idx+1:02d}"
    }
    for idx, q in enumerate(all_questions)
])

print("\n📊 Evaluation Dataset Preview:")
display(eval_df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Evaluation Dataset

# COMMAND ----------

# Save to Delta table for reuse
# MAGIC %run ../00_setup/00_config

# COMMAND ----------

# Save evaluation dataset
eval_table_name = f"{FULL_SCHEMA}.agent_eval_dataset"

spark.createDataFrame(eval_df).write.mode("overwrite").saveAsTable(eval_table_name)

print(f"✅ Saved evaluation dataset to: {eval_table_name}")
print(f"   Total questions: {len(eval_df)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sample Questions by Complexity

# COMMAND ----------

print("\n" + "="*60)
print("📝 Sample Questions by Complexity")
print("="*60 + "\n")

for complexity in ["simple", "medium", "complex"]:
    sample = eval_df[eval_df["complexity"] == complexity].iloc[0]
    print(f"\n{complexity.upper()} QUESTION:")
    print(f"  {sample['request']}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dataset Summary

# COMMAND ----------

print(f"""
{'='*60}
✅ Evaluation Dataset Created Successfully
{'='*60}

Total Questions: {len(eval_df)}
  - Simple: {len(eval_df[eval_df['complexity'] == 'simple'])}
  - Medium: {len(eval_df[eval_df['complexity'] == 'medium'])}
  - Complex: {len(eval_df[eval_df['complexity'] == 'complex'])}

Saved to: {eval_table_name}

Next Step: Run driver.py to evaluate the agent!

{'='*60}
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Expected Agent Behavior
# MAGIC
# MAGIC ### For the Main Demo Question:
# MAGIC
# MAGIC **Question:**
# MAGIC > "Show me high-risk patients discharged in the last 7 days who need outreach this week.
# MAGIC > Focus on CHF and COPD patients with prior readmissions."
# MAGIC
# MAGIC **Expected Tool Sequence:**
# MAGIC 1. `get_recent_discharges(7)` → 127 patients
# MAGIC 2. `get_diagnoses_by_condition('CHF')` → 23 patients
# MAGIC 3. `get_diagnoses_by_condition('COPD')` → 19 patients
# MAGIC 4. `get_patient_readmission_history()` for each → 32 with prior readmissions
# MAGIC 5. `get_risk_score()` for each → 18 high-risk (>0.7)
# MAGIC 6. `get_sdoh_barriers()` for high-risk → 5 transportation, 3 housing
# MAGIC 7. `get_available_care_coordinators('CHF')` → 2 coordinators
# MAGIC 8. `suggest_coordinator_for_patient()` → assignments
# MAGIC
# MAGIC **Expected Response:**
# MAGIC - Identifies 18 high-risk patients
# MAGIC - Lists CHF and COPD patients separately
# MAGIC - Highlights prior readmissions
# MAGIC - Flags social barriers (transportation, housing)
# MAGIC - Provides care coordinator assignments
# MAGIC - Includes priority tiers and outreach timeline
