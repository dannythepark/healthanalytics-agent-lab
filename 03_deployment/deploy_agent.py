# Databricks notebook source
# MAGIC %md
# MAGIC # HealthAnalytics AI - Deploy Clinical Agent
# MAGIC
# MAGIC ## 🚀 Deploy to Production Model Serving
# MAGIC
# MAGIC This notebook deploys the Clinical Analytics Agent to a **Model Serving endpoint** for production use.
# MAGIC
# MAGIC ### Deployment Steps
# MAGIC
# MAGIC 1. **Select winning agent** from evaluation results
# MAGIC 2. **Register agent to Unity Catalog** as a versioned model
# MAGIC 3. **Deploy to Model Serving** with serverless compute
# MAGIC 4. **Test the endpoint** with sample queries
# MAGIC 5. **Enable monitoring** for usage and performance
# MAGIC
# MAGIC ### Production Considerations
# MAGIC
# MAGIC - **Versioning:** All agent versions tracked in Unity Catalog
# MAGIC - **Governance:** Permissions managed through Unity Catalog
# MAGIC - **Scalability:** Serverless endpoints auto-scale
# MAGIC - **Monitoring:** Query logs and performance metrics
# MAGIC - **Cost:** Pay only for actual usage
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %pip install databricks-agents mlflow --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
from databricks import agents
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput, EndpointCoreConfigInput
import time
import requests

# Import configuration
import sys
sys.path.append("../00_setup")
from config import CATALOG, SCHEMA, FULL_SCHEMA, SERVING_ENDPOINT_NAME

# COMMAND ----------

# Initialize Workspace Client
w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Select Agent for Deployment

# COMMAND ----------

# Based on evaluation results, deploy the improved agent
AGENT_TO_DEPLOY = "clinical_agent_improved"
MODEL_NAME = f"{FULL_SCHEMA}.{AGENT_TO_DEPLOY}"

print(f"🎯 Deploying agent: {MODEL_NAME}")
print(f"🚀 To endpoint: {SERVING_ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Register Agent to Unity Catalog
# MAGIC
# MAGIC The agent is already registered from agent.py, but we'll verify and update if needed.

# COMMAND ----------

# Get the latest version of the agent
client = mlflow.MlflowClient()

try:
    latest_version = client.get_latest_versions(MODEL_NAME, stages=["None"])[0]
    print(f"✅ Found registered model: {MODEL_NAME}")
    print(f"   Latest version: {latest_version.version}")
    print(f"   Status: {latest_version.status}")
except Exception as e:
    print(f"❌ Error finding model: {e}")
    print("   Please run 02_agent_eval/agent.py first to create the agent")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Deploy to Model Serving

# COMMAND ----------

# Check if endpoint already exists
existing_endpoints = [ep.name for ep in w.serving_endpoints.list()]

if SERVING_ENDPOINT_NAME in existing_endpoints:
    print(f"⚠️ Endpoint '{SERVING_ENDPOINT_NAME}' already exists")
    print("   Updating endpoint with new model version...")

    # Update existing endpoint
    w.serving_endpoints.update_config(
        name=SERVING_ENDPOINT_NAME,
        served_entities=[
            ServedEntityInput(
                entity_name=MODEL_NAME,
                scale_to_zero_enabled=True,
                workload_size="Small",
                entity_version=latest_version.version
            )
        ]
    )
    print(f"✅ Updated endpoint: {SERVING_ENDPOINT_NAME}")

else:
    print(f"Creating new endpoint: {SERVING_ENDPOINT_NAME}")

    # Create new endpoint
    w.serving_endpoints.create(
        name=SERVING_ENDPOINT_NAME,
        config=EndpointCoreConfigInput(
            served_entities=[
                ServedEntityInput(
                    entity_name=MODEL_NAME,
                    scale_to_zero_enabled=True,
                    workload_size="Small",
                    entity_version=latest_version.version
                )
            ]
        )
    )
    print(f"✅ Created endpoint: {SERVING_ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Wait for Endpoint to be Ready

# COMMAND ----------

print(f"\n⏳ Waiting for endpoint to be ready...")

max_wait = 600  # 10 minutes
start_time = time.time()

while time.time() - start_time < max_wait:
    endpoint = w.serving_endpoints.get(SERVING_ENDPOINT_NAME)
    state = endpoint.state.config_update if endpoint.state else None

    if state == "IN_PROGRESS":
        print(f"   Status: {state} - waiting...")
        time.sleep(30)
    elif state == "UPDATE_COMPLETE" or endpoint.state.ready == "READY":
        print(f"✅ Endpoint is ready!")
        break
    else:
        print(f"   Status: {state}")
        time.sleep(30)

# Get final endpoint details
endpoint = w.serving_endpoints.get(SERVING_ENDPOINT_NAME)
print(f"\n📊 Endpoint Details:")
print(f"   Name: {endpoint.name}")
print(f"   State: {endpoint.state.ready}")
print(f"   URL: https://{w.config.host}/serving-endpoints/{SERVING_ENDPOINT_NAME}/invocations")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Test the Deployed Endpoint

# COMMAND ----------

# Get authentication token
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# Endpoint URL
endpoint_url = f"https://{w.config.host}/serving-endpoints/{SERVING_ENDPOINT_NAME}/invocations"

# Test query - the Monday morning question
test_query = {
    "messages": [
        {
            "role": "user",
            "content": """Show me high-risk patients discharged in the last 7 days who need
            outreach this week. Focus on CHF and COPD patients with prior readmissions."""
        }
    ]
}

print("🧪 Testing endpoint with Monday morning question...")
print("="*60)

# Make request
response = requests.post(
    endpoint_url,
    json=test_query,
    headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    },
    timeout=120
)

if response.status_code == 200:
    result = response.json()
    print("✅ Endpoint test successful!\n")
    print("Response:")
    print(result.get("choices", [{}])[0].get("message", {}).get("content", "No response"))
else:
    print(f"❌ Endpoint test failed: {response.status_code}")
    print(response.text)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Additional Test Queries

# COMMAND ----------

# Simple test queries
simple_queries = [
    "How many patients were discharged in the last 7 days?",
    "Which care coordinators have available capacity?",
    "Show me all CHF patients"
]

print("\n🧪 Running additional test queries...\n")

for query in simple_queries:
    print(f"Query: {query}")

    response = requests.post(
        endpoint_url,
        json={"messages": [{"role": "user", "content": query}]},
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        },
        timeout=60
    )

    if response.status_code == 200:
        result = response.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        print(f"✅ Response: {content[:100]}...\n")
    else:
        print(f"❌ Failed: {response.status_code}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Enable Query Logging and Monitoring

# COMMAND ----------

# MLflow automatically logs queries to the endpoint
# You can query these logs for monitoring

print("""
📊 Monitoring Options:

1. **MLflow Tracking:**
   - All queries are logged to MLflow
   - View in the MLflow UI under the serving endpoint

2. **Databricks Lakehouse Monitoring:**
   - Set up data quality monitors
   - Track response times and token usage
   - Monitor for data drift

3. **Query Logs:**
   - Access raw query logs via Unity Catalog
   - Analyze usage patterns
   - Identify common questions

4. **Cost Tracking:**
   - Monitor token usage per query
   - Track endpoint compute costs
   - Optimize workload size if needed

See Databricks documentation for detailed monitoring setup.
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Production Integration Examples

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example 1: Python Application Integration

# COMMAND ----------

print("""
# HealthAnalytics Dashboard - Agent Integration

import requests
import os

ENDPOINT_URL = "https://your-workspace.cloud.databricks.com/serving-endpoints/{}/invocations"
TOKEN = os.environ.get("DATABRICKS_TOKEN")

def query_clinical_agent(question: str) -> str:
    '''Query the HealthAnalytics Clinical Agent'''

    payload = {{
        "messages": [
            {{"role": "user", "content": question}}
        ]
    }}

    response = requests.post(
        ENDPOINT_URL,
        json=payload,
        headers={{
            "Authorization": f"Bearer {{TOKEN}}",
            "Content-Type": "application/json"
        }}
    )

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Agent query failed: {{response.text}}")


# Example usage in a Streamlit dashboard
if __name__ == "__main__":
    question = st.text_input("Ask the Clinical Analytics Agent:")

    if st.button("Get Answer"):
        with st.spinner("Analyzing patient data..."):
            answer = query_clinical_agent(question)
            st.write(answer)
""".format(SERVING_ENDPOINT_NAME))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example 2: Scheduled Databricks Job

# COMMAND ----------

print("""
# Monday Morning Automated Report

from databricks import agents
import mlflow

# Load the agent
agent = mlflow.databricks.load_model(
    f"models:/{MODEL_NAME}/latest"
)

# Generate Monday morning report
question = '''
Show me high-risk patients discharged in the last 7 days who need outreach
this week. Focus on CHF and COPD patients with prior readmissions.
'''

response = agent.predict({{"messages": [{{"role": "user", "content": question}}]}})

# Send to care management team via email or Slack
send_to_care_team(response)

# Save to Delta table for record-keeping
spark.createDataFrame([
    {{"date": current_date(), "report": response}}
]).write.mode("append").saveAsTable("care_management.weekly_reports")
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deployment Summary

# COMMAND ----------

endpoint = w.serving_endpoints.get(SERVING_ENDPOINT_NAME)

print(f"""
{'='*60}
✅ Clinical Analytics Agent Deployed Successfully
{'='*60}

Agent: {MODEL_NAME}
Version: {latest_version.version}
Endpoint: {SERVING_ENDPOINT_NAME}
Status: {endpoint.state.ready}

Endpoint URL:
https://{w.config.host}/serving-endpoints/{SERVING_ENDPOINT_NAME}/invocations

{'='*60}
🎉 Production Deployment Complete!
{'='*60}

Next Steps:

1. **Integrate with Applications:**
   - Add agent to care management dashboard
   - Create Slack bot for clinicians
   - Build scheduled reporting jobs

2. **Monitor Performance:**
   - Track query response times
   - Monitor token usage and costs
   - Review query logs for common patterns

3. **Iterate and Improve:**
   - Collect user feedback
   - Add new tools as needed
   - Refine prompts based on usage

4. **Scale to Other Use Cases:**
   - Population health agent
   - Financial analytics agent
   - Clinical quality metrics agent

{'='*60}
📚 Documentation: docs.databricks.com/generative-ai/agent-framework
{'='*60}
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cost Optimization Tips

# COMMAND ----------

print("""
💰 Cost Optimization Strategies:

1. **Right-size the endpoint:**
   - Start with "Small" workload size
   - Monitor usage and scale up only if needed
   - Enable scale-to-zero for dev/test environments

2. **Optimize prompts:**
   - Shorter system prompts = fewer input tokens
   - Encourage concise responses
   - Use structured outputs when possible

3. **Cache common queries:**
   - Store frequent query results in Delta tables
   - Check cache before calling agent
   - Refresh cache on a schedule

4. **Use smaller models for simple queries:**
   - Route simple questions to a lightweight agent
   - Reserve large model for complex reasoning
   - Implement query complexity routing

5. **Monitor token usage:**
   - Track tokens per query in MLflow
   - Identify expensive queries
   - Optimize or cache them

Current setup: Llama 3.1 70B on serverless endpoint
Estimated cost: $X per 1M tokens (check Databricks pricing)
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rollback Plan

# COMMAND ----------

print(f"""
🔄 Rollback Instructions:

If you need to rollback to a previous agent version:

1. **List available versions:**
   ```python
   client = mlflow.MlflowClient()
   versions = client.search_model_versions(f"name='{MODEL_NAME}'")
   for v in versions:
       print(f"Version {{v.version}}: {{v.description}}")
   ```

2. **Update endpoint to previous version:**
   ```python
   w.serving_endpoints.update_config(
       name="{SERVING_ENDPOINT_NAME}",
       served_entities=[
           ServedEntityInput(
               entity_name="{MODEL_NAME}",
               entity_version="<previous_version>",  # e.g., "2"
               scale_to_zero_enabled=True,
               workload_size="Small"
           )
       ]
   )
   ```

3. **Verify rollback:**
   - Test endpoint with known queries
   - Check evaluation metrics
   - Monitor for any issues

Always test rollback procedures in a dev environment first!
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎉 Congratulations!
# MAGIC
# MAGIC You've successfully completed the **HealthAnalytics AI Clinical Agent Workshop**!
# MAGIC
# MAGIC ### What You Built
# MAGIC
# MAGIC - ✅ **Synthetic healthcare data** (7 tables, 233K+ records)
# MAGIC - ✅ **9 analytics tools** (SQL + Python functions)
# MAGIC - ✅ **Clinical AI agent** with healthcare expertise
# MAGIC - ✅ **Evaluation framework** with custom judges
# MAGIC - ✅ **Production deployment** on Model Serving
# MAGIC
# MAGIC ### The Impact
# MAGIC
# MAGIC - ⏱️ **Time saved:** 4-6 hours → 60 seconds
# MAGIC - 🎯 **Consistency:** Same methodology every time
# MAGIC - 🚀 **Actionability:** Immediate care coordinator assignments
# MAGIC - 🔒 **Compliance:** HIPAA-compliant, all data in Databricks
# MAGIC
# MAGIC ### Share Your Success
# MAGIC
# MAGIC This is Maria's new Monday morning:
# MAGIC > *"I can now focus on complex clinical research instead of manual reporting.*
# MAGIC > *The agent handles the repetitive analysis, and I validate the recommendations.*
# MAGIC > *Our care team gets actionable insights by 8am every Monday."*
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Built with ❤️ by the Databricks Healthcare Solutions Team**
