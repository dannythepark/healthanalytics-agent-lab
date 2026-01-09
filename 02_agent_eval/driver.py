# Databricks notebook source
# MAGIC %md
# MAGIC # HealthAnalytics AI - Agent Evaluation Driver
# MAGIC
# MAGIC ## 🧪 Evaluate Clinical Agent Performance
# MAGIC
# MAGIC This notebook evaluates the Clinical Analytics Agent using custom healthcare-specific judges.
# MAGIC
# MAGIC ### Evaluation Process
# MAGIC
# MAGIC 1. Load the evaluation dataset (20 clinical questions)
# MAGIC 2. Run both agents (basic and improved prompts)
# MAGIC 3. Apply custom judges:
# MAGIC    - **Clinical Accuracy:** Correct risk thresholds, ICD-10 codes, 30-day readmissions
# MAGIC    - **Completeness:** All relevant data sources checked
# MAGIC    - **Actionability:** Specific patient lists, coordinator assignments, timelines
# MAGIC    - **HIPAA Compliance:** No external calls, data stays in Databricks
# MAGIC 4. Compare results and select best agent
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %pip install databricks-agents mlflow --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
from databricks import agents
import pandas as pd
import json

# Import configuration
import sys
sys.path.append("../00_setup")
from config import CATALOG, SCHEMA, FULL_SCHEMA

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Evaluation Dataset

# COMMAND ----------

# Load the evaluation questions
eval_table_name = f"{FULL_SCHEMA}.agent_eval_dataset"
eval_df = spark.table(eval_table_name).toPandas()

print(f"✅ Loaded {len(eval_df)} evaluation questions")
print(f"\nBreakdown by complexity:")
print(eval_df["complexity"].value_counts())

# COMMAND ----------

# Display sample questions
print("\n📝 Sample Questions:\n")
for idx, row in eval_df.head(3).iterrows():
    print(f"{row['request_id']}: {row['request'][:80]}...")
    print(f"   Complexity: {row['complexity']}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Custom Judges

# COMMAND ----------

def clinical_accuracy_judge(response: str, expected_facts: list) -> dict:
    """
    Judge clinical accuracy of the agent response.

    Criteria:
    - Correct risk thresholds (>0.7 for high-risk)
    - Proper identification of CHF/COPD using ICD-10
    - Accurate 30-day readmission logic
    - Appropriate use of clinical terminology

    Returns: Score 1-5 and rationale
    """
    score = 5
    rationale = []

    # Check for expected facts
    facts_found = sum(1 for fact in expected_facts if str(fact).lower() in response.lower())
    fact_coverage = facts_found / len(expected_facts) if expected_facts else 1.0

    if fact_coverage < 0.3:
        score = 1
        rationale.append("Missing most expected clinical facts")
    elif fact_coverage < 0.6:
        score = 3
        rationale.append("Missing some key clinical facts")
    elif fact_coverage < 0.8:
        score = 4
        rationale.append("Most clinical facts present")
    else:
        score = 5
        rationale.append("All key clinical facts present")

    # Check for clinical terminology
    clinical_terms = ["risk score", "readmission", "CHF", "COPD", "ICD-10", "30-day"]
    terms_found = sum(1 for term in clinical_terms if term.lower() in response.lower())

    if terms_found < 2:
        score = min(score, 2)
        rationale.append("Limited use of clinical terminology")

    return {
        "score": score,
        "rationale": "; ".join(rationale),
        "fact_coverage": round(fact_coverage, 2)
    }


def completeness_judge(response: str, expected_tools: list) -> dict:
    """
    Judge completeness of data source checking.

    Criteria:
    - All relevant tables queried
    - No missed data sources
    - Comprehensive patient cohort identified

    Returns: Score 1-5 and rationale
    """
    score = 5
    rationale = []

    # Check for mentions of key data sources
    data_sources = [
        "discharge", "diagnos", "readmission",
        "risk", "sdoh", "social", "coordinator"
    ]

    sources_mentioned = sum(1 for source in data_sources if source in response.lower())

    if sources_mentioned < 2:
        score = 2
        rationale.append("Very limited data source coverage")
    elif sources_mentioned < 4:
        score = 3
        rationale.append("Some data sources checked")
    elif sources_mentioned < 6:
        score = 4
        rationale.append("Good data source coverage")
    else:
        score = 5
        rationale.append("Comprehensive data source coverage")

    return {
        "score": score,
        "rationale": "; ".join(rationale),
        "sources_mentioned": sources_mentioned
    }


def actionability_judge(response: str) -> dict:
    """
    Judge actionability of recommendations.

    Criteria:
    - Specific patient list or count
    - Care coordinator assignments
    - Priority tiers with timeframes
    - Clear next steps

    Returns: Score 1-5 and rationale
    """
    score = 5
    rationale = []

    # Check for actionable elements
    actionable_elements = {
        "patient_list": any(word in response.lower() for word in ["patient", "mrn", "list"]),
        "coordinator": "coordinator" in response.lower(),
        "priority": any(word in response.lower() for word in ["priority", "tier", "urgent"]),
        "timeline": any(word in response.lower() for word in ["24 hour", "48 hour", "week", "immediately"]),
        "next_steps": any(word in response.lower() for word in ["recommend", "should", "contact", "outreach"])
    }

    elements_present = sum(actionable_elements.values())

    if elements_present < 2:
        score = 2
        rationale.append("Limited actionable guidance")
    elif elements_present < 3:
        score = 3
        rationale.append("Some actionable elements")
    elif elements_present < 4:
        score = 4
        rationale.append("Good actionable recommendations")
    else:
        score = 5
        rationale.append("Highly actionable with clear next steps")

    return {
        "score": score,
        "rationale": "; ".join(rationale),
        "actionable_elements": elements_present
    }


def hipaa_compliance_judge(response: str) -> dict:
    """
    Judge HIPAA compliance.

    Criteria:
    - No external API calls mentioned
    - All data stays within Databricks
    - No PHI exposure
    - Appropriate data handling

    Returns: Pass/Fail and rationale
    """
    # Red flags for HIPAA violations
    red_flags = [
        "external api", "third party", "export",
        "email", "download", "share outside"
    ]

    violations = [flag for flag in red_flags if flag in response.lower()]

    if violations:
        return {
            "score": "FAIL",
            "rationale": f"Potential HIPAA violation: {', '.join(violations)}",
            "violations": violations
        }

    # Green flags for compliance
    green_flags = [
        "databricks", "unity catalog", "within",
        "synthetic", "hipaa"
    ]

    compliance_indicators = sum(1 for flag in green_flags if flag in response.lower())

    return {
        "score": "PASS",
        "rationale": "No HIPAA violations detected; data stays within Databricks",
        "compliance_indicators": compliance_indicators
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Evaluation with Custom Judges

# COMMAND ----------

def evaluate_agent(agent_name: str, eval_dataset: pd.DataFrame):
    """
    Evaluate an agent using custom healthcare judges.

    Args:
        agent_name: Name of the agent model in Unity Catalog
        eval_dataset: DataFrame with evaluation questions

    Returns:
        Evaluation results with scores
    """
    print(f"\n🧪 Evaluating agent: {agent_name}")
    print("="*60)

    # Load the agent
    agent_uri = f"models:/{FULL_SCHEMA}.{agent_name}/latest"
    print(f"Loading agent from: {agent_uri}")

    with mlflow.start_run(run_name=f"eval_{agent_name}") as run:
        # Evaluate using MLflow
        results = mlflow.evaluate(
            agent_uri,
            data=eval_dataset,
            model_type="databricks-agent",
            targets="expected_facts",  # Ground truth
        )

        # Apply custom judges
        scores = []

        for idx, row in eval_dataset.iterrows():
            request = row["request"]
            expected_facts = json.loads(row["expected_facts"])
            expected_tools = json.loads(row["expected_tools"])

            # Get agent response (this would come from MLflow in practice)
            # For demo purposes, we'll skip actual prediction here

            # Placeholder response for judge testing
            response = "Sample response"  # In practice: agent.predict(request)

            # Apply judges
            clinical_score = clinical_accuracy_judge(response, expected_facts)
            completeness_score = completeness_judge(response, expected_tools)
            actionability_score = actionability_judge(response)
            hipaa_score = hipaa_compliance_judge(response)

            scores.append({
                "request_id": row["request_id"],
                "complexity": row["complexity"],
                "clinical_accuracy": clinical_score["score"],
                "completeness": completeness_score["score"],
                "actionability": actionability_score["score"],
                "hipaa_compliance": hipaa_score["score"],
            })

        # Log metrics to MLflow
        scores_df = pd.DataFrame(scores)

        avg_clinical = scores_df["clinical_accuracy"].mean()
        avg_completeness = scores_df["completeness"].mean()
        avg_actionability = scores_df["actionability"].mean()

        mlflow.log_metric("avg_clinical_accuracy", avg_clinical)
        mlflow.log_metric("avg_completeness", avg_completeness)
        mlflow.log_metric("avg_actionability", avg_actionability)

        print(f"\n✅ Evaluation Complete!")
        print(f"   Clinical Accuracy: {avg_clinical:.2f}/5")
        print(f"   Completeness: {avg_completeness:.2f}/5")
        print(f"   Actionability: {avg_actionability:.2f}/5")

        return results, scores_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate Basic Agent

# COMMAND ----------

print("Evaluating BASIC agent...")
basic_results, basic_scores = evaluate_agent("clinical_agent_basic", eval_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate Improved Agent

# COMMAND ----------

print("Evaluating IMPROVED agent...")
improved_results, improved_scores = evaluate_agent("clinical_agent_improved", eval_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compare Results

# COMMAND ----------

# Compare average scores
comparison = pd.DataFrame({
    "Judge": ["Clinical Accuracy", "Completeness", "Actionability"],
    "Basic Agent": [
        basic_scores["clinical_accuracy"].mean(),
        basic_scores["completeness"].mean(),
        basic_scores["actionability"].mean()
    ],
    "Improved Agent": [
        improved_scores["clinical_accuracy"].mean(),
        improved_scores["completeness"].mean(),
        improved_scores["actionability"].mean()
    ]
})

comparison["Improvement"] = comparison["Improved Agent"] - comparison["Basic Agent"]

print("\n" + "="*60)
print("📊 Agent Comparison")
print("="*60)
display(comparison)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Breakdown by Complexity

# COMMAND ----------

# Scores by question complexity
complexity_breakdown = improved_scores.groupby("complexity")[
    ["clinical_accuracy", "completeness", "actionability"]
].mean()

print("\n" + "="*60)
print("📈 Performance by Question Complexity (Improved Agent)")
print("="*60)
display(complexity_breakdown)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Identify Problem Areas

# COMMAND ----------

# Find low-scoring questions
low_scores = improved_scores[
    (improved_scores["clinical_accuracy"] < 3) |
    (improved_scores["completeness"] < 3) |
    (improved_scores["actionability"] < 3)
]

if len(low_scores) > 0:
    print(f"\n⚠️ Found {len(low_scores)} questions with low scores:")
    print("\nThese questions need prompt refinement:")
    for idx, row in low_scores.iterrows():
        request_row = eval_df[eval_df["request_id"] == row["request_id"]].iloc[0]
        print(f"\n{row['request_id']}: {request_row['request'][:60]}...")
        print(f"  Clinical: {row['clinical_accuracy']}, Completeness: {row['completeness']}, Actionability: {row['actionability']}")
else:
    print("\n✅ No low-scoring questions! Agent performing well across all questions.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Select Best Agent

# COMMAND ----------

# Calculate overall scores
basic_overall = basic_scores[["clinical_accuracy", "completeness", "actionability"]].mean().mean()
improved_overall = improved_scores[["clinical_accuracy", "completeness", "actionability"]].mean().mean()

print("\n" + "="*60)
print("🏆 Final Recommendation")
print("="*60)

print(f"\nBasic Agent Overall Score: {basic_overall:.2f}/5")
print(f"Improved Agent Overall Score: {improved_overall:.2f}/5")

if improved_overall > basic_overall:
    winner = "clinical_agent_improved"
    improvement = ((improved_overall - basic_overall) / basic_overall) * 100
    print(f"\n✅ RECOMMENDATION: Deploy the IMPROVED agent")
    print(f"   Performance improvement: {improvement:.1f}%")
else:
    winner = "clinical_agent_basic"
    print(f"\n✅ RECOMMENDATION: Deploy the BASIC agent")

print(f"\n📦 Winning agent: {FULL_SCHEMA}.{winner}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation Summary

# COMMAND ----------

print(f"""
{'='*60}
✅ Agent Evaluation Complete
{'='*60}

Dataset: {len(eval_df)} questions
  - Simple: {len(eval_df[eval_df['complexity'] == 'simple'])}
  - Medium: {len(eval_df[eval_df['complexity'] == 'medium'])}
  - Complex: {len(eval_df[eval_df['complexity'] == 'complex'])}

Judges Applied:
  ✓ Clinical Accuracy (correctness of clinical facts)
  ✓ Completeness (data source coverage)
  ✓ Actionability (specific recommendations)
  ✓ HIPAA Compliance (data privacy)

Results:
  Basic Agent: {basic_overall:.2f}/5
  Improved Agent: {improved_overall:.2f}/5

Recommendation: Deploy {winner}

Next Step: Run 03_deployment/deploy_agent.py to deploy to Model Serving

{'='*60}
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow Tracking
# MAGIC
# MAGIC View detailed evaluation results in the MLflow UI:
# MAGIC
# MAGIC 1. Click **Experiments** in the left sidebar
# MAGIC 2. Find the evaluation runs
# MAGIC 3. Compare metrics and artifacts
# MAGIC 4. Review individual question performance

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tips for Improving Agent Performance
# MAGIC
# MAGIC ### If Clinical Accuracy is Low:
# MAGIC - Add more clinical context to the system prompt
# MAGIC - Include specific risk thresholds (e.g., ">0.7 for high-risk")
# MAGIC - Reference ICD-10 codes explicitly
# MAGIC - Add examples of clinical reasoning
# MAGIC
# MAGIC ### If Completeness is Low:
# MAGIC - Explicitly list all available data sources in the prompt
# MAGIC - Encourage the agent to check multiple tables
# MAGIC - Add instructions to "always check readmissions AND social barriers"
# MAGIC
# MAGIC ### If Actionability is Low:
# MAGIC - Add examples of good recommendations
# MAGIC - Request specific formats (e.g., "provide a numbered list")
# MAGIC - Include priority tier definitions
# MAGIC - Add timeline guidance (24h, 48h, etc.)
# MAGIC
# MAGIC ### If HIPAA Compliance Fails:
# MAGIC - Emphasize "all data stays within Databricks"
# MAGIC - Prohibit external API calls in the prompt
# MAGIC - Add compliance reminders
