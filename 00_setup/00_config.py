# Databricks notebook source
# MAGIC %md
# MAGIC # HealthAnalytics AI - Configuration
# MAGIC
# MAGIC ## 🏥 Welcome to the Clinical Analytics Agent Lab
# MAGIC
# MAGIC This configuration file defines the Unity Catalog location for the HealthAnalytics AI demo.
# MAGIC
# MAGIC ### ✏️ **REQUIRED: Update Your Catalog Name**
# MAGIC
# MAGIC Before running any other notebooks, update the `CATALOG` variable below with your Unity Catalog name.
# MAGIC
# MAGIC **Example:**
# MAGIC ```python
# MAGIC CATALOG = "my_healthcare_catalog"
# MAGIC ```
# MAGIC
# MAGIC ### What Gets Created
# MAGIC
# MAGIC - **Catalog:** `<your_catalog>` (must already exist)
# MAGIC - **Schema:** `healthanalytics_ai` (auto-created)
# MAGIC - **Tables:** 7 synthetic healthcare tables (patients, encounters, diagnoses, etc.)
# MAGIC - **Functions:** 9 clinical analytics functions (SQL + Python)
# MAGIC - **Model:** Clinical Analytics Agent (registered and deployed)
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration Variables

# COMMAND ----------

# ========================================
# REQUIRED: Update this with your catalog name
# ========================================
CATALOG = "your_catalog_name"  # TODO: Replace with your Unity Catalog name

# Schema name (will be auto-created)
SCHEMA = "healthanalytics_ai"

# Full path for tables and functions
FULL_SCHEMA = f"{CATALOG}.{SCHEMA}"

# Model configuration
MODEL_NAME = "clinical_analytics_agent"
SERVING_ENDPOINT_NAME = "healthanalytics-agent-endpoint"

# Foundation model for the agent
FOUNDATION_MODEL = "databricks-meta-llama-3-1-70b-instruct"

# ========================================
# Data Configuration
# ========================================

# Synthetic data sizes (for DLT pipeline)
DATA_CONFIG = {
    "patients": 10_000,
    "encounters": 50_000,
    "diagnoses": 150_000,
    "readmissions": 5_000,
    "risk_scores": 10_000,
    "sdoh": 8_000,
    "care_coordinators": 15,
}

# Demo scenario configuration
DEMO_CONFIG = {
    "recent_discharges_days": 7,
    "recent_discharge_count": 127,
    "chf_patient_count": 23,
    "copd_patient_count": 19,
    "prior_readmission_count": 32,
    "high_risk_threshold": 0.7,
    "high_risk_count": 18,
    "transportation_barrier_count": 5,
    "housing_instability_count": 3,
    "available_coordinators": 2,
}

# Clinical codes (ICD-10)
CLINICAL_CODES = {
    "CHF": "I50",  # Congestive Heart Failure
    "COPD": "J44",  # Chronic Obstructive Pulmonary Disease
    "Diabetes": "E11",  # Type 2 Diabetes Mellitus
    "Hypertension": "I10",  # Essential Hypertension
    "Sepsis": "A41",  # Sepsis
}

# ========================================
# Validation
# ========================================

def validate_config():
    """Validate configuration before running notebooks"""

    if CATALOG == "your_catalog_name":
        raise ValueError(
            "❌ ERROR: Please update the CATALOG variable in 00_config.py "
            "with your Unity Catalog name before proceeding.\n\n"
            "Example: CATALOG = 'my_healthcare_catalog'"
        )

    print("✅ Configuration validated successfully!")
    print(f"   📁 Catalog: {CATALOG}")
    print(f"   📂 Schema: {SCHEMA}")
    print(f"   🎯 Full path: {FULL_SCHEMA}")
    print(f"   🤖 Model: {FOUNDATION_MODEL}")
    return True

# Run validation when this file is executed
if __name__ == "__main__":
    validate_config()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper Functions

# COMMAND ----------

def create_schema_if_not_exists():
    """Create the schema if it doesn't already exist"""
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {FULL_SCHEMA}")
    print(f"✅ Schema {FULL_SCHEMA} ready")

def get_table_path(table_name: str) -> str:
    """Get the full path for a table"""
    return f"{FULL_SCHEMA}.{table_name}"

def get_function_path(function_name: str) -> str:
    """Get the full path for a function"""
    return f"{FULL_SCHEMA}.{function_name}"

def list_tables():
    """List all tables in the schema"""
    tables = spark.sql(f"SHOW TABLES IN {FULL_SCHEMA}").collect()
    print(f"📊 Tables in {FULL_SCHEMA}:")
    for table in tables:
        print(f"   - {table.tableName}")
    return tables

def list_functions():
    """List all functions in the schema"""
    functions = spark.sql(f"SHOW USER FUNCTIONS IN {FULL_SCHEMA}").collect()
    print(f"🔧 Functions in {FULL_SCHEMA}:")
    for func in functions:
        print(f"   - {func.function}")
    return functions

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Configuration

# COMMAND ----------

# Validate configuration
validate_config()

# Create schema
create_schema_if_not_exists()

print("\n" + "="*60)
print("🎉 Configuration complete! You can now run the other notebooks.")
print("="*60)
print("\n📝 Next steps:")
print("   1. Run 00_setup/01_dlt_synthetic_data.py to create tables")
print("   2. Run 01_create_tools/01_create_tools.py to create functions")
print("   3. Run 02_agent_eval/agent.py to test the agent")
print("="*60)
