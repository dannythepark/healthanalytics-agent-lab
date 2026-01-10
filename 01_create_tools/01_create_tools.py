# Databricks notebook source
# MAGIC %md
# MAGIC # HealthAnalytics AI - Create Analytics Tools
# MAGIC
# MAGIC ## 🔧 Unity Catalog Functions for Clinical Analytics
# MAGIC
# MAGIC This notebook creates **9 reusable analytics functions** that the Clinical Analytics Agent will use:
# MAGIC
# MAGIC ### SQL Functions (6)
# MAGIC 1. `get_recent_discharges(days_back)` - Recent discharges
# MAGIC 2. `get_diagnoses_by_condition(condition_name)` - Filter by clinical condition
# MAGIC 3. `get_patient_readmission_history(patient_id)` - Readmission history
# MAGIC 4. `get_risk_score(patient_id)` - Predictive risk score
# MAGIC 5. `get_sdoh_barriers(patient_id)` - Social determinants data
# MAGIC 6. `get_available_care_coordinators(specialty)` - Coordinator capacity
# MAGIC
# MAGIC ### Python Functions (3)
# MAGIC 7. `calculate_priority_score(patient_id)` - Composite risk scoring
# MAGIC 8. `assign_care_coordinator(patient_list, specialty)` - Caseload balancing
# MAGIC 9. `generate_outreach_report(patient_list)` - Markdown reports
# MAGIC
# MAGIC These functions are registered in **Unity Catalog** and can be:
# MAGIC - Called by AI agents
# MAGIC - Used in SQL queries
# MAGIC - Tested in AI Playground
# MAGIC - Versioned and governed
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %run ../00_setup/00_config

# COMMAND ----------

print(f"Creating functions in: {FULL_SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## SQL Function 1: get_recent_discharges()
# MAGIC
# MAGIC Find patients discharged in the last N days.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {FULL_SCHEMA}.get_recent_discharges(days_back INT)
RETURNS TABLE(
  encounter_id STRING,
  patient_id STRING,
  mrn STRING,
  patient_name STRING,
  discharge_date TIMESTAMP,
  facility STRING,
  encounter_type STRING,
  length_of_stay INT,
  discharge_disposition STRING
)
COMMENT 'Returns patients discharged in the last N days with key demographics'
RETURN
  SELECT
    e.encounter_id,
    e.patient_id,
    p.mrn,
    CONCAT(p.first_name, ' ', p.last_name) as patient_name,
    e.discharge_date,
    e.facility,
    e.encounter_type,
    e.length_of_stay,
    e.discharge_disposition
  FROM {FULL_SCHEMA}.encounters e
  JOIN {FULL_SCHEMA}.patients p ON e.patient_id = p.patient_id
  WHERE e.discharge_date >= CURRENT_DATE() - days_back
  ORDER BY e.discharge_date DESC
""")

# COMMAND ----------

# Test the function
print("Testing get_recent_discharges(7)...")
display(spark.sql(f"SELECT * FROM {FULL_SCHEMA}.get_recent_discharges(7) LIMIT 10"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## SQL Function 2: get_diagnoses_by_condition()
# MAGIC
# MAGIC Filter encounters by clinical condition (CHF, COPD, Diabetes, etc.).

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {FULL_SCHEMA}.get_diagnoses_by_condition(condition_name STRING)
RETURNS TABLE(
  encounter_id STRING,
  patient_id STRING,
  mrn STRING,
  patient_name STRING,
  icd10_code STRING,
  description STRING,
  is_primary BOOLEAN,
  diagnosis_date TIMESTAMP
)
COMMENT 'Returns encounters with specific diagnosis (CHF, COPD, Diabetes, etc.)'
RETURN
  WITH condition_mapping AS (
    SELECT 'CHF' as condition, 'I50%' as icd_pattern
    UNION ALL SELECT 'COPD', 'J44%'
    UNION ALL SELECT 'Diabetes', 'E11%'
    UNION ALL SELECT 'Hypertension', 'I10%'
    UNION ALL SELECT 'Sepsis', 'A41%'
  )
  SELECT DISTINCT
    d.encounter_id,
    e.patient_id,
    p.mrn,
    CONCAT(p.first_name, ' ', p.last_name) as patient_name,
    d.icd10_code,
    d.description,
    d.is_primary,
    d.diagnosis_date
  FROM {FULL_SCHEMA}.diagnoses d
  JOIN {FULL_SCHEMA}.encounters e ON d.encounter_id = e.encounter_id
  JOIN {FULL_SCHEMA}.patients p ON e.patient_id = p.patient_id
  JOIN condition_mapping cm ON UPPER(cm.condition) = UPPER(condition_name)
  WHERE d.icd10_code LIKE cm.icd_pattern
""")

# COMMAND ----------

# Test the function
print("Testing get_diagnoses_by_condition('CHF')...")
display(spark.sql(f"SELECT * FROM {FULL_SCHEMA}.get_diagnoses_by_condition('CHF') LIMIT 10"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## SQL Function 3: get_patient_readmission_history()
# MAGIC
# MAGIC Check if a patient has prior readmissions (with 30-day flagging).

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {FULL_SCHEMA}.get_patient_readmission_history(patient_id_param STRING)
RETURNS TABLE(
  readmission_id STRING,
  patient_id STRING,
  original_discharge_date TIMESTAMP,
  readmit_admission_date TIMESTAMP,
  days_between INT,
  is_30_day BOOLEAN
)
COMMENT 'Returns readmission history for a patient with 30-day CMS quality metric'
RETURN
  SELECT
    readmission_id,
    patient_id,
    original_discharge_date,
    readmit_admission_date,
    days_between,
    is_30_day
  FROM {FULL_SCHEMA}.readmissions
  WHERE patient_id = patient_id_param
  ORDER BY original_discharge_date DESC
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## SQL Function 4: get_risk_score()
# MAGIC
# MAGIC Get patient's calculated readmission risk score.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {FULL_SCHEMA}.get_risk_score(patient_id_param STRING)
RETURNS TABLE(
  patient_id STRING,
  risk_score DOUBLE,
  risk_category STRING,
  risk_factors STRING,
  last_calculated TIMESTAMP
)
COMMENT 'Returns predictive readmission risk score (0.0-1.0 scale) and contributing factors'
RETURN
  SELECT
    patient_id,
    risk_score,
    risk_category,
    risk_factors,
    last_calculated
  FROM {FULL_SCHEMA}.risk_scores
  WHERE patient_id = patient_id_param
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## SQL Function 5: get_sdoh_barriers()
# MAGIC
# MAGIC Identify social determinants of health barriers for a patient.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {FULL_SCHEMA}.get_sdoh_barriers(patient_id_param STRING)
RETURNS TABLE(
  patient_id STRING,
  housing_instability BOOLEAN,
  transportation_barrier BOOLEAN,
  food_insecurity BOOLEAN,
  social_isolation BOOLEAN,
  financial_strain BOOLEAN,
  last_assessed TIMESTAMP
)
COMMENT 'Returns social determinants of health (SDOH) barriers affecting patient care'
RETURN
  SELECT
    patient_id,
    housing_instability,
    transportation_barrier,
    food_insecurity,
    social_isolation,
    financial_strain,
    last_assessed
  FROM {FULL_SCHEMA}.sdoh
  WHERE patient_id = patient_id_param
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## SQL Function 6: get_available_care_coordinators()
# MAGIC
# MAGIC Find care coordinators with available capacity and specialty match.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {FULL_SCHEMA}.get_available_care_coordinators(specialty_param STRING)
RETURNS TABLE(
  coordinator_id STRING,
  name STRING,
  title STRING,
  current_caseload INT,
  max_caseload INT,
  available_capacity INT,
  specialties STRING,
  years_experience INT
)
COMMENT 'Returns care coordinators with available capacity and optional specialty match'
RETURN
  SELECT
    coordinator_id,
    name,
    title,
    current_caseload,
    max_caseload,
    available_capacity,
    specialties,
    years_experience
  FROM {FULL_SCHEMA}.care_coordinators
  WHERE active = TRUE
    AND available_capacity > 0
    AND (specialty_param IS NULL OR specialties LIKE CONCAT('%', specialty_param, '%'))
  ORDER BY available_capacity DESC, years_experience DESC
""")

# COMMAND ----------

# Test the function
print("Testing get_available_care_coordinators('CHF')...")
display(spark.sql(f"SELECT * FROM {FULL_SCHEMA}.get_available_care_coordinators('CHF')"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Python Function 7: calculate_priority_score()
# MAGIC
# MAGIC Combine clinical + social factors into a composite priority score.

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, ArrayType
import json

@udf(returnType=StructType([
    StructField("patient_id", StringType()),
    StructField("priority_score", DoubleType()),
    StructField("risk_factors", ArrayType(StringType())),
    StructField("recommendations", StringType())
]))
def calculate_priority_score_udf(patient_id: str) -> dict:
    """
    Calculate composite priority score combining clinical and social risk factors.

    Scoring Algorithm:
    - Base risk score (0-1): 60% weight
    - Prior 30-day readmission: +0.2
    - Transportation barrier: +0.1
    - Housing instability: +0.15
    - CHF/COPD diagnosis: +0.1

    Returns dict with priority_score, risk_factors list, and recommendations.
    """
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()

    try:
        # Get risk score
        risk_df = spark.sql(f"""
            SELECT risk_score, risk_factors
            FROM {FULL_SCHEMA}.risk_scores
            WHERE patient_id = '{patient_id}'
        """)

        if risk_df.count() == 0:
            return {
                "patient_id": patient_id,
                "priority_score": 0.0,
                "risk_factors": ["No risk score available"],
                "recommendations": "Complete risk assessment"
            }

        risk_row = risk_df.first()
        base_score = float(risk_row.risk_score) * 0.6  # 60% weight

        # Get readmission history
        readmit_df = spark.sql(f"""
            SELECT COUNT(*) as readmit_count
            FROM {FULL_SCHEMA}.readmissions
            WHERE patient_id = '{patient_id}' AND is_30_day = TRUE
        """)
        has_readmission = readmit_df.first().readmit_count > 0

        # Get SDOH barriers
        sdoh_df = spark.sql(f"""
            SELECT housing_instability, transportation_barrier
            FROM {FULL_SCHEMA}.sdoh
            WHERE patient_id = '{patient_id}'
        """)

        has_transport = False
        has_housing = False
        if sdoh_df.count() > 0:
            sdoh_row = sdoh_df.first()
            has_transport = bool(sdoh_row.transportation_barrier)
            has_housing = bool(sdoh_row.housing_instability)

        # Get diagnoses
        diag_df = spark.sql(f"""
            SELECT COUNT(*) as chf_copd_count
            FROM {FULL_SCHEMA}.diagnoses d
            JOIN {FULL_SCHEMA}.encounters e ON d.encounter_id = e.encounter_id
            WHERE e.patient_id = '{patient_id}'
              AND (d.icd10_code LIKE 'I50%' OR d.icd10_code LIKE 'J44%')
        """)
        has_chf_copd = diag_df.first().chf_copd_count > 0

        # Calculate composite score
        priority_score = base_score
        risk_factors = []

        if has_readmission:
            priority_score += 0.2
            risk_factors.append("Prior 30-day readmission")

        if has_transport:
            priority_score += 0.1
            risk_factors.append("Transportation barrier")

        if has_housing:
            priority_score += 0.15
            risk_factors.append("Housing instability")

        if has_chf_copd:
            priority_score += 0.1
            risk_factors.append("CHF/COPD diagnosis")

        # Cap at 1.0
        priority_score = min(priority_score, 1.0)

        # Generate recommendations
        if priority_score > 0.8:
            recommendations = "URGENT: Contact within 24 hours. Coordinate with social services."
        elif priority_score > 0.6:
            recommendations = "HIGH PRIORITY: Contact within 48 hours. Address barriers."
        else:
            recommendations = "MODERATE: Contact within 72 hours. Standard follow-up."

        return {
            "patient_id": patient_id,
            "priority_score": round(priority_score, 3),
            "risk_factors": risk_factors,
            "recommendations": recommendations
        }

    except Exception as e:
        return {
            "patient_id": patient_id,
            "priority_score": 0.0,
            "risk_factors": [f"Error: {str(e)}"],
            "recommendations": "Manual review required"
        }

# Register as Unity Catalog function
spark.udf.register(f"{FULL_SCHEMA}.calculate_priority_score", calculate_priority_score_udf)

print(f"✅ Registered Python UDF: {FULL_SCHEMA}.calculate_priority_score")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Python Function 8: assign_care_coordinator()
# MAGIC
# MAGIC Assign patients to care coordinators balancing caseloads and matching specialties.

# COMMAND ----------

# MAGIC %md
# MAGIC **Note:** Complex Python functions with list inputs work best as standalone Python files.
# MAGIC For the demo, we'll create a simpler SQL-based assignment function.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION ${FULL_SCHEMA}.suggest_coordinator_for_patient(
# MAGIC   patient_id STRING,
# MAGIC   specialty STRING
# MAGIC )
# MAGIC RETURNS TABLE(
# MAGIC   patient_id STRING,
# MAGIC   coordinator_id STRING,
# MAGIC   coordinator_name STRING,
# MAGIC   available_capacity INT,
# MAGIC   match_quality STRING
# MAGIC )
# MAGIC COMMENT 'Suggests best care coordinator for a patient based on specialty and capacity'
# MAGIC RETURN
# MAGIC   SELECT
# MAGIC     patient_id as patient_id,
# MAGIC     coordinator_id,
# MAGIC     name as coordinator_name,
# MAGIC     available_capacity,
# MAGIC     CASE
# MAGIC       WHEN specialties LIKE CONCAT('%', specialty, '%') THEN 'Perfect Match'
# MAGIC       WHEN specialties LIKE '%Geriatrics%' THEN 'General Match'
# MAGIC       ELSE 'Available'
# MAGIC     END as match_quality
# MAGIC   FROM ${FULL_SCHEMA}.care_coordinators
# MAGIC   WHERE active = TRUE AND available_capacity > 0
# MAGIC   ORDER BY
# MAGIC     CASE WHEN specialties LIKE CONCAT('%', specialty, '%') THEN 1 ELSE 2 END,
# MAGIC     available_capacity DESC
# MAGIC   LIMIT 1;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Python Function 9: generate_outreach_report()
# MAGIC
# MAGIC Create actionable markdown report for care management team.

# COMMAND ----------

def generate_outreach_report(patient_ids: list) -> str:
    """
    Generate formatted markdown report for care management.

    Args:
        patient_ids: List of patient IDs to include in report

    Returns:
        Formatted markdown report string
    """
    from pyspark.sql import SparkSession
    from datetime import datetime

    spark = SparkSession.builder.getOrCreate()

    # Build patient ID list for SQL IN clause
    patient_id_list = ", ".join([f"'{pid}'" for pid in patient_ids])

    # Get comprehensive patient data
    query = f"""
    SELECT
      p.patient_id,
      p.mrn,
      CONCAT(p.first_name, ' ', p.last_name) as patient_name,
      p.age,
      rs.risk_score,
      rs.risk_category,
      rs.risk_factors,
      s.housing_instability,
      s.transportation_barrier,
      e.discharge_date,
      e.facility
    FROM {FULL_SCHEMA}.patients p
    LEFT JOIN {FULL_SCHEMA}.risk_scores rs ON p.patient_id = rs.patient_id
    LEFT JOIN {FULL_SCHEMA}.sdoh s ON p.patient_id = s.patient_id
    LEFT JOIN {FULL_SCHEMA}.encounters e ON p.patient_id = e.patient_id
    WHERE p.patient_id IN ({patient_id_list})
    ORDER BY rs.risk_score DESC
    """

    df = spark.sql(query).toPandas()

    # Generate markdown report
    report = f"""# 🏥 HealthAnalytics AI - Care Coordination Outreach Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Patients Identified:** {len(df)}

---

## 📊 Executive Summary

- **High-Risk Patients:** {len(df[df['risk_score'] > 0.7])}
- **Transportation Barriers:** {len(df[df['transportation_barrier'] == True])}
- **Housing Instability:** {len(df[df['housing_instability'] == True])}

---

## 👥 Patient List (Priority Order)

"""

    for idx, row in df.iterrows():
        tier = "🔴 TIER 1" if row['risk_score'] > 0.8 else "🟡 TIER 2" if row['risk_score'] > 0.6 else "🟢 TIER 3"

        barriers = []
        if row.get('transportation_barrier'):
            barriers.append("🚗 Transportation")
        if row.get('housing_instability'):
            barriers.append("🏠 Housing")

        barriers_text = ", ".join(barriers) if barriers else "None identified"

        report += f"""
### {idx+1}. {row['patient_name']} (MRN: {row['mrn']}) - {tier}

- **Age:** {row['age']}
- **Risk Score:** {row['risk_score']:.2f} ({row['risk_category']})
- **Last Discharge:** {row['discharge_date']} from {row['facility']}
- **Social Barriers:** {barriers_text}
- **Recommended Action:** {"Contact within 24h" if row['risk_score'] > 0.8 else "Contact within 48h"}

"""

    report += """
---

## 📝 Next Steps

1. Assign care coordinators to Tier 1 patients immediately
2. Schedule outreach calls within recommended timeframes
3. Address transportation and housing barriers with social services
4. Document all interventions in the care management system

**Report Generated by HealthAnalytics AI Clinical Agent**
"""

    return report

# Register as a callable Unity Catalog function
@udf(returnType=StringType())
def generate_outreach_report_udf(patient_ids: list) -> str:
    return generate_outreach_report(patient_ids)

spark.udf.register(f"{FULL_SCHEMA}.generate_outreach_report", generate_outreach_report_udf)

print(f"✅ Registered Python UDF: {FULL_SCHEMA}.generate_outreach_report")
print("✅ Created Python function: generate_outreach_report()")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary: All Functions Created

# COMMAND ----------

# List all created functions
functions = spark.sql(f"SHOW USER FUNCTIONS IN {FULL_SCHEMA}").collect()

print(f"\n{'='*60}")
print(f"✅ Successfully created {len(functions)} functions in {FULL_SCHEMA}")
print(f"{'='*60}\n")

print("SQL Functions:")
for func in functions:
    if "calculate_priority_score" not in func.function:
        print(f"  ✓ {func.function}")

print("\nPython Functions:")
print(f"  ✓ {FULL_SCHEMA}.calculate_priority_score (UDF)")
print(f"  ✓ generate_outreach_report() (Python function)")

print(f"\n{'='*60}")
print("🎉 All tools ready! Next step: Run 02_agent_eval/agent.py")
print(f"{'='*60}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test All Functions Together
# MAGIC
# MAGIC Let's simulate the full demo query workflow.

# COMMAND ----------

print("🧪 Testing the full Monday morning workflow...\n")

# Step 1: Recent discharges
print("1️⃣ Get recent discharges (last 7 days)...")
recent = spark.sql(f"SELECT COUNT(*) as count FROM {FULL_SCHEMA}.get_recent_discharges(7)").first()
print(f"   ✅ Found {recent.count} recent discharges\n")

# Step 2: CHF patients
print("2️⃣ Filter for CHF patients...")
chf = spark.sql(f"SELECT COUNT(DISTINCT patient_id) as count FROM {FULL_SCHEMA}.get_diagnoses_by_condition('CHF')").first()
print(f"   ✅ Found {chf.count} CHF patients\n")

# Step 3: COPD patients
print("3️⃣ Filter for COPD patients...")
copd = spark.sql(f"SELECT COUNT(DISTINCT patient_id) as count FROM {FULL_SCHEMA}.get_diagnoses_by_condition('COPD')").first()
print(f"   ✅ Found {copd.count} COPD patients\n")

# Step 4: Available coordinators
print("4️⃣ Find available care coordinators...")
coords = spark.sql(f"SELECT COUNT(*) as count FROM {FULL_SCHEMA}.get_available_care_coordinators('CHF')").first()
print(f"   ✅ Found {coords.count} coordinators with capacity\n")

print("="*60)
print("🎉 All functions working! Ready for agent integration.")
print("="*60)
