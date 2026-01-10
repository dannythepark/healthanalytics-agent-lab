# Databricks notebook source
# MAGIC %md
# MAGIC # HealthAnalytics AI - Synthetic Data Generation
# MAGIC
# MAGIC ## 🏥 HIPAA-Compliant Synthetic Healthcare Data
# MAGIC
# MAGIC This notebook generates **synthetic healthcare data** using standard Spark DataFrames.
# MAGIC
# MAGIC ### ⚠️ Important: This is 100% Synthetic Data
# MAGIC
# MAGIC - **No real patient information (PHI)**
# MAGIC - **No real medical records**
# MAGIC - **Generated using Faker library**
# MAGIC - **For demonstration and training purposes only**
# MAGIC
# MAGIC ### 📊 What Gets Created
# MAGIC
# MAGIC | Table | Records | Purpose |
# MAGIC |-------|---------|---------|
# MAGIC | `patients` | 10,000 | Patient demographics |
# MAGIC | `encounters` | 50,000 | Hospital admissions and ED visits |
# MAGIC | `diagnoses` | 150,000 | Clinical diagnoses (ICD-10 codes) |
# MAGIC | `readmissions` | 5,000 | All-cause readmissions |
# MAGIC | `risk_scores` | 10,000 | Predictive readmission risk scores |
# MAGIC | `sdoh` | 8,000 | Social determinants of health |
# MAGIC | `care_coordinators` | 15 | Care team capacity and specialties |
# MAGIC
# MAGIC ### 🎯 Demo Scenario Planted in Data
# MAGIC
# MAGIC - ✅ **127 recent discharges** in the last 7 days
# MAGIC - ✅ **23 CHF patients** (ICD-10: I50.*)
# MAGIC - ✅ **19 COPD patients** (ICD-10: J44.*)
# MAGIC - ✅ **32 patients** with prior 30-day readmissions
# MAGIC - ✅ **18 high-risk patients** (risk score > 0.7)
# MAGIC - ✅ **5 patients** with transportation barriers
# MAGIC - ✅ **3 patients** with housing instability
# MAGIC - ✅ **2 care coordinators** with CHF/COPD specialty and capacity
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %pip install faker --quiet
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from pyspark.sql import Row
from pyspark.sql.functions import *
from pyspark.sql.types import *
from faker import Faker
import random
from datetime import datetime, timedelta
import json
import builtins  # For Python's builtin min() to avoid PySpark conflict

# Initialize Faker
fake = Faker()
Faker.seed(42)
random.seed(42)

print("✅ Libraries imported successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC
# MAGIC **⚠️ Shared configuration loaded from 00_config**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table 1: Patients (10,000 records)

# COMMAND ----------

print(f"Generating {DATA_CONFIG['patients']:,} patients...")

# Generate patients using Row objects (more compatible with Serverless)
patients_rows = []
for i in range(DATA_CONFIG['patients']):
    gender = random.choice(["M", "F", "Other"])

    if gender == "M":
        first_name = fake.first_name_male()
    elif gender == "F":
        first_name = fake.first_name_female()
    else:
        first_name = fake.first_name()

    patients_rows.append(Row(
        patient_id=f"patient-{str(i+1).zfill(6)}",
        mrn=f"MRN{random.randint(100000, 999999)}",
        first_name=first_name,
        last_name=fake.last_name(),
        age=random.randint(18, 95),
        gender=gender,
        zip_code=fake.zipcode(),
        city=fake.city(),
        state=fake.state_abbr(),
        race_ethnicity=random.choices(
            ["White", "Black or African American", "Hispanic or Latino", "Asian", "Other"],
            weights=[60, 13, 18, 6, 3]
        )[0],
        created_at=fake.date_time_between(start_date="-5y", end_date="now")
    ))

patients_df = spark.createDataFrame(patients_rows)
patients_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{FULL_SCHEMA}.patients")

print(f"✅ Created {FULL_SCHEMA}.patients with {patients_df.count():,} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table 2: Encounters (50,000 records with 127 recent discharges)

# COMMAND ----------

print(f"Generating {DATA_CONFIG['encounters']:,} encounters...")

# Read patients to get IDs (using Spark operations, not collect)
patient_ids_df = spark.table(f"{FULL_SCHEMA}.patients").select("patient_id")
patient_ids = [row.patient_id for row in patient_ids_df.collect()]

facilities = [
    "Central Hospital", "North Medical Center", "South Regional Hospital",
    "East Community Hospital", "West Medical Center", "Children's Hospital"
]

encounters_rows = []
encounter_counter = 1

# STEP 1: Plant recent discharges
recent_count = DEMO_CONFIG["recent_discharge_count"]
recent_days = DEMO_CONFIG["recent_discharges_days"]
print(f"  → Planting {recent_count} recent discharges (last {recent_days} days)...")
demo_patients = random.sample(patient_ids, recent_count)

for patient_id in demo_patients:
    discharge_date = fake.date_time_between(start_date=f"-{recent_days}d", end_date="now")
    length_of_stay = random.randint(1, 14)
    admission_date = discharge_date - timedelta(days=length_of_stay)

    encounters_rows.append(Row(
        encounter_id=f"encounter-{str(encounter_counter).zfill(8)}",
        patient_id=patient_id,
        admission_date=admission_date,
        discharge_date=discharge_date,
        encounter_type=random.choice(["Inpatient", "Emergency"]),
        facility=random.choice(facilities),
        length_of_stay=length_of_stay,
        discharge_disposition=random.choice(["Home", "Home Health", "SNF", "Rehab"]),
        is_demo_recent_discharge=True
    ))
    encounter_counter += 1

# STEP 2: Generate remaining historical encounters
historical_count = DATA_CONFIG['encounters'] - recent_count
print(f"  → Generating {historical_count:,} historical encounters...")
for _ in range(historical_count):
    patient_id = random.choice(patient_ids)
    admission_date = fake.date_time_between(start_date="-3y", end_date=f"-{recent_days+1}d")
    length_of_stay = random.randint(1, 30)
    discharge_date = admission_date + timedelta(days=length_of_stay)

    encounters_rows.append(Row(
        encounter_id=f"encounter-{str(encounter_counter).zfill(8)}",
        patient_id=patient_id,
        admission_date=admission_date,
        discharge_date=discharge_date,
        encounter_type=random.choice(["Inpatient", "Emergency", "Observation"]),
        facility=random.choice(facilities),
        length_of_stay=length_of_stay,
        discharge_disposition=random.choice(["Home", "Home Health", "SNF", "Rehab", "AMA"]),
        is_demo_recent_discharge=False
    ))
    encounter_counter += 1

encounters_df = spark.createDataFrame(encounters_rows)
encounters_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{FULL_SCHEMA}.encounters")

print(f"✅ Created {FULL_SCHEMA}.encounters with 50,000 records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table 3: Diagnoses (150,000 records with CHF/COPD)

# COMMAND ----------

print(f"Generating {DATA_CONFIG['diagnoses']:,} diagnoses...")

all_encounter_ids = [row.encounter_id for row in spark.table(f"{FULL_SCHEMA}.encounters").select("encounter_id").collect()]
recent_encounter_ids = [
    row.encounter_id
    for row in spark.table(f"{FULL_SCHEMA}.encounters").filter(col("is_demo_recent_discharge") == True).collect()
]

common_diagnoses = [
    ("I50.9", "Congestive Heart Failure, unspecified"),
    ("I50.23", "Acute on chronic systolic heart failure"),
    ("J44.1", "Chronic obstructive pulmonary disease with acute exacerbation"),
    ("J44.0", "COPD with acute lower respiratory infection"),
    ("E11.9", "Type 2 diabetes mellitus without complications"),
    ("I10", "Essential (primary) hypertension"),
    ("I48.91", "Atrial fibrillation, unspecified"),
    ("N18.3", "Chronic kidney disease, stage 3"),
    ("J18.9", "Pneumonia, unspecified organism"),
    ("A41.9", "Sepsis, unspecified organism"),
]

diagnoses_rows = []
diagnosis_counter = 1

# STEP 1: Plant CHF patients
chf_count = DEMO_CONFIG["chf_patient_count"]
print(f"  → Planting {chf_count} CHF patients...")
chf_encounters = random.sample(recent_encounter_ids, builtins.min(chf_count, len(recent_encounter_ids)))

for encounter_id in chf_encounters:
    icd10, description = random.choice([d for d in common_diagnoses if d[0].startswith("I50")])
    diagnoses_rows.append(Row(
        diagnosis_id=f"diagnosis-{str(diagnosis_counter).zfill(9)}",
        encounter_id=encounter_id,
        icd10_code=icd10,
        description=description,
        is_primary=True,
        diagnosis_date=fake.date_time_between(start_date=f"-{DEMO_CONFIG['recent_discharges_days']}d", end_date="now"),
        is_demo_chf=True,
        is_demo_copd=False
    ))
    diagnosis_counter += 1

# STEP 2: Plant COPD patients
copd_count = DEMO_CONFIG["copd_patient_count"]
print(f"  → Planting {copd_count} COPD patients...")
remaining_recent = [e for e in recent_encounter_ids if e not in chf_encounters]
copd_encounters = random.sample(remaining_recent, builtins.min(copd_count, len(remaining_recent)))

for encounter_id in copd_encounters:
    icd10, description = random.choice([d for d in common_diagnoses if d[0].startswith("J44")])
    diagnoses_rows.append(Row(
        diagnosis_id=f"diagnosis-{str(diagnosis_counter).zfill(9)}",
        encounter_id=encounter_id,
        icd10_code=icd10,
        description=description,
        is_primary=True,
        diagnosis_date=fake.date_time_between(start_date=f"-{DEMO_CONFIG['recent_discharges_days']}d", end_date="now"),
        is_demo_chf=False,
        is_demo_copd=True
    ))
    diagnosis_counter += 1

# STEP 3: Generate remaining diagnoses
print("  → Generating remaining diagnoses...")
total_diagnoses = DATA_CONFIG['diagnoses']
batch_size = 10000
while len(diagnoses_rows) < total_diagnoses:
    for _ in range(builtins.min(batch_size, total_diagnoses - len(diagnoses_rows))):
        encounter_id = random.choice(all_encounter_ids)
        icd10, description = random.choice(common_diagnoses)

        diagnoses_rows.append(Row(
            diagnosis_id=f"diagnosis-{str(diagnosis_counter).zfill(9)}",
            encounter_id=encounter_id,
            icd10_code=icd10,
            description=description,
            is_primary=random.choice([True, False]),
            diagnosis_date=fake.date_time_between(start_date="-3y", end_date="now"),
            is_demo_chf=False,
            is_demo_copd=False
        ))
        diagnosis_counter += 1

diagnoses_df = spark.createDataFrame(diagnoses_rows)
diagnoses_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{FULL_SCHEMA}.diagnoses")

print(f"✅ Created {FULL_SCHEMA}.diagnoses with {total_diagnoses:,} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table 4: Readmissions (5,000 records with 32 prior 30-day)

# COMMAND ----------

print(f"Generating {DATA_CONFIG['readmissions']:,} readmissions using optimized Spark SQL...")

# Use pure Spark SQL to generate readmissions - much faster!
# This creates all readmissions in one query instead of looping
spark.sql(f"""
    CREATE OR REPLACE TABLE {FULL_SCHEMA}.readmissions AS
    WITH ranked_encounters AS (
        SELECT
            patient_id,
            encounter_id,
            admission_date,
            discharge_date,
            is_demo_recent_discharge,
            ROW_NUMBER() OVER (PARTITION BY patient_id ORDER BY admission_date) as enc_num,
            COUNT(*) OVER (PARTITION BY patient_id) as total_encs
        FROM {FULL_SCHEMA}.encounters
    ),
    encounter_pairs AS (
        SELECT
            e1.patient_id,
            e1.encounter_id as original_encounter_id,
            e2.encounter_id as readmit_encounter_id,
            e1.discharge_date as original_discharge_date,
            e2.admission_date as readmit_admission_date,
            DATEDIFF(e2.admission_date, e1.discharge_date) as days_between,
            e1.enc_num,
            e2.enc_num as readmit_enc_num,
            e2.is_demo_recent_discharge
        FROM ranked_encounters e1
        JOIN ranked_encounters e2
            ON e1.patient_id = e2.patient_id
            AND e2.enc_num = e1.enc_num + 1
        WHERE e1.total_encs >= 2
            AND e2.admission_date > e1.discharge_date
            AND DATEDIFF(e2.admission_date, e1.discharge_date) <= 365
    )
    SELECT
        CONCAT('readmit-', LPAD(CAST(ROW_NUMBER() OVER (ORDER BY patient_id, days_between) AS STRING), 7, '0')) as readmission_id,
        patient_id,
        original_encounter_id,
        readmit_encounter_id,
        original_discharge_date,
        readmit_admission_date,
        days_between,
        CASE WHEN days_between <= 30 THEN TRUE ELSE FALSE END as is_30_day,
        FALSE as is_demo_readmission,
        is_demo_recent_discharge
    FROM encounter_pairs
    LIMIT {DATA_CONFIG['readmissions']}
""")

# Now update records to be demo readmissions with 30-day flag
readmit_count = DEMO_CONFIG["prior_readmission_count"]
print(f"  → Marking {readmit_count} as demo 30-day readmissions...")
spark.sql(f"""
    CREATE OR REPLACE TABLE {FULL_SCHEMA}.readmissions AS
    WITH marked_readmissions AS (
        SELECT
            *,
            ROW_NUMBER() OVER (ORDER BY is_demo_recent_discharge DESC, RAND()) as row_num,
            CAST(FLOOR(RAND() * 30) + 1 AS INT) as random_days
        FROM {FULL_SCHEMA}.readmissions
    )
    SELECT
        readmission_id,
        patient_id,
        original_encounter_id,
        readmit_encounter_id,
        original_discharge_date,
        CASE
            WHEN row_num <= {readmit_count} THEN DATE_ADD(original_discharge_date, random_days)
            ELSE readmit_admission_date
        END as readmit_admission_date,
        CASE
            WHEN row_num <= {readmit_count} THEN random_days
            ELSE days_between
        END as days_between,
        CASE WHEN row_num <= {readmit_count} THEN TRUE ELSE is_30_day END as is_30_day,
        CASE WHEN row_num <= {readmit_count} THEN TRUE ELSE FALSE END as is_demo_readmission
    FROM marked_readmissions
""")

readmission_count = spark.table(f"{FULL_SCHEMA}.readmissions").count()
print(f"✅ Created {FULL_SCHEMA}.readmissions with {readmission_count:,} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table 5: Risk Scores (10,000 records with 18 high-risk)

# COMMAND ----------

print(f"Generating {DATA_CONFIG['risk_scores']:,} risk scores...")

all_patient_ids = [row.patient_id for row in spark.table(f"{FULL_SCHEMA}.patients").select("patient_id").collect()]

# Get CHF/COPD patients
chf_copd_patient_ids = [
    row.patient_id
    for row in spark.sql(f"""
        SELECT DISTINCT e.patient_id
        FROM {FULL_SCHEMA}.diagnoses d
        JOIN {FULL_SCHEMA}.encounters e ON d.encounter_id = e.encounter_id
        WHERE d.is_demo_chf = TRUE OR d.is_demo_copd = TRUE
    """).collect()
]

risk_scores_rows = []

# Plant high-risk patients
hr_count = DEMO_CONFIG["high_risk_count"]
hr_threshold = DEMO_CONFIG["high_risk_threshold"]
print(f"  → Planting {hr_count} high-risk patients (score > {hr_threshold})...")
high_risk_patients = random.sample(chf_copd_patient_ids, builtins.min(hr_count, len(chf_copd_patient_ids)))

for patient_id in high_risk_patients:
    risk_score = builtins.round(random.uniform(hr_threshold + 0.01, 0.95), 3)
    risk_factors = random.sample([
        "Prior 30-day readmission",
        "CHF/COPD diagnosis",
        "Multiple comorbidities",
        "Age > 65",
        "Polypharmacy",
        "Frequent ED utilization"
    ], k=random.randint(3, 5))

    risk_scores_rows.append(Row(
        patient_id=patient_id,
        risk_score=risk_score,
        risk_category="High",
        risk_factors=json.dumps(risk_factors),
        last_calculated=fake.date_time_between(start_date="-7d", end_date="now"),
        model_version="v2.3.1",
        is_demo_high_risk=True
    ))

# Add remaining risk scores
print("  → Generating remaining risk scores...")
used_patients = set(high_risk_patients)
remaining_patients = [p for p in all_patient_ids if p not in used_patients]
total_risk_scores = DATA_CONFIG['risk_scores']

for patient_id in remaining_patients[:total_risk_scores - len(high_risk_patients)]:
    risk_score = builtins.round(random.triangular(0.1, 0.8, 0.3), 3)
    category = "Low" if risk_score < 0.3 else "Moderate" if risk_score < hr_threshold else "High"

    risk_scores_rows.append(Row(
        patient_id=patient_id,
        risk_score=risk_score,
        risk_category=category,
        risk_factors=json.dumps([]),
        last_calculated=fake.date_time_between(start_date="-30d", end_date="now"),
        model_version="v2.3.1",
        is_demo_high_risk=False
    ))

risk_scores_df = spark.createDataFrame(risk_scores_rows)
risk_scores_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{FULL_SCHEMA}.risk_scores")

print(f"✅ Created {FULL_SCHEMA}.risk_scores with {len(risk_scores_rows):,} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table 6: Social Determinants of Health (8,000 records)

# COMMAND ----------

print(f"Generating {DATA_CONFIG['sdoh']:,} SDOH records...")

high_risk_patient_ids = [
    row.patient_id
    for row in spark.table(f"{FULL_SCHEMA}.risk_scores").filter(col("is_demo_high_risk") == True).collect()
]

sdoh_rows = []

# Plant transportation barriers
trans_count = DEMO_CONFIG["transportation_barrier_count"]
print(f"  → Planting {trans_count} patients with transportation barriers...")
transport_patients = random.sample(high_risk_patient_ids, builtins.min(trans_count, len(high_risk_patient_ids)))

for patient_id in transport_patients:
    sdoh_rows.append(Row(
        patient_id=patient_id,
        housing_instability=random.choice([True, False]),
        transportation_barrier=True,
        food_insecurity=random.choice([True, False]),
        social_isolation=random.choice([True, False]),
        financial_strain=random.choice([True, False]),
        utility_assistance_needed=random.choice([True, False]),
        last_assessed=fake.date_time_between(start_date="-30d", end_date="now")
    ))

# Plant housing instability
housing_count = DEMO_CONFIG["housing_instability_count"]
print(f"  → Planting {housing_count} patients with housing instability...")
remaining_high_risk = [p for p in high_risk_patient_ids if p not in transport_patients]
housing_patients = random.sample(remaining_high_risk, builtins.min(housing_count, len(remaining_high_risk)))

for patient_id in housing_patients:
    sdoh_rows.append(Row(
        patient_id=patient_id,
        housing_instability=True,
        transportation_barrier=random.choice([True, False]),
        food_insecurity=random.choice([True, False]),
        social_isolation=random.choice([True, False]),
        financial_strain=True,
        utility_assistance_needed=random.choice([True, False]),
        last_assessed=fake.date_time_between(start_date="-30d", end_date="now")
    ))

# Generate remaining SDOH records
print("  → Generating remaining SDOH records...")
used_patients = set(transport_patients + housing_patients)
remaining = [p for p in all_patient_ids if p not in used_patients]
total_sdoh = DATA_CONFIG['sdoh']

for patient_id in remaining[:total_sdoh - len(sdoh_rows)]:
    num_barriers = random.choices([0, 1, 2, 3], weights=[40, 30, 20, 10])[0]
    all_barriers = ["housing_instability", "transportation_barrier", "food_insecurity",
                    "social_isolation", "financial_strain", "utility_assistance_needed"]

    active_barriers = set(random.sample(all_barriers, k=num_barriers)) if num_barriers > 0 else set()

    sdoh_rows.append(Row(
        patient_id=patient_id,
        housing_instability=("housing_instability" in active_barriers),
        transportation_barrier=("transportation_barrier" in active_barriers),
        food_insecurity=("food_insecurity" in active_barriers),
        social_isolation=("social_isolation" in active_barriers),
        financial_strain=("financial_strain" in active_barriers),
        utility_assistance_needed=("utility_assistance_needed" in active_barriers),
        last_assessed=fake.date_time_between(start_date="-90d", end_date="now")
    ))

sdoh_df = spark.createDataFrame(sdoh_rows)
sdoh_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{FULL_SCHEMA}.sdoh")

print(f"✅ Created {FULL_SCHEMA}.sdoh with {len(sdoh_rows):,} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table 7: Care Coordinators (15 records)

# COMMAND ----------

print(f"Generating {DATA_CONFIG['care_coordinators']:,} care coordinators...")

coordinators_rows = [
    Row(coordinator_id="coord-001", name="Sarah Johnson, RN", title="Senior Care Coordinator",
        current_caseload=18, max_caseload=30, available_capacity=12,
        specialties=json.dumps(["CHF", "COPD", "Cardiology"]), years_experience=12,
        active=True, is_demo_coordinator=True),
    Row(coordinator_id="coord-002", name="Michael Chen, MSW", title="Care Coordinator",
        current_caseload=22, max_caseload=30, available_capacity=8,
        specialties=json.dumps(["COPD", "Pulmonary", "Geriatrics"]), years_experience=8,
        active=True, is_demo_coordinator=True),
    Row(coordinator_id="coord-003", name="Jessica Martinez, RN", title="Care Coordinator",
        current_caseload=28, max_caseload=30, available_capacity=2,
        specialties=json.dumps(["Diabetes", "Endocrinology"]), years_experience=6,
        active=True, is_demo_coordinator=False),
    Row(coordinator_id="coord-004", name="David Thompson, BSN", title="Care Coordinator",
        current_caseload=25, max_caseload=30, available_capacity=5,
        specialties=json.dumps(["Oncology", "Palliative Care"]), years_experience=10,
        active=True, is_demo_coordinator=False),
    Row(coordinator_id="coord-005", name="Emily Rodriguez, RN", title="Senior Care Coordinator",
        current_caseload=20, max_caseload=30, available_capacity=10,
        specialties=json.dumps(["Pediatrics"]), years_experience=15,
        active=True, is_demo_coordinator=False),
]

# Add more coordinators
total_coords = DATA_CONFIG['care_coordinators']
more_coords = [
    ("coord-006", "James Wilson, MSW", 30, ["Behavioral Health"], 7),
    ("coord-007", "Maria Garcia, RN", 24, ["Nephrology"], 9),
    ("coord-008", "Robert Lee, BSN", 26, ["Orthopedics"], 5),
    ("coord-009", "Linda Anderson, MSW", 15, ["Geriatrics"], 18),
    ("coord-010", "Christopher Brown, RN", 30, ["Trauma"], 11),
    ("coord-011", "Patricia Davis, BSN", 29, ["Maternal Health"], 8),
    ("coord-012", "Daniel Miller, MSW", 21, ["Homeless Outreach"], 6),
    ("coord-013", "Jennifer Taylor, RN", 27, ["Infectious Disease"], 13),
    ("coord-014", "Kevin White, BSN", 23, ["Neurology"], 7),
    ("coord-015", "Nancy Thomas, MSW", 19, ["Pain Management"], 9),
]

for coord_id, name, caseload, specialties, experience in more_coords[:total_coords - 5]:
    coordinators_rows.append(Row(
        coordinator_id=coord_id, name=name, title="Care Coordinator",
        current_caseload=caseload, max_caseload=30, available_capacity=30-caseload,
        specialties=json.dumps(specialties), years_experience=experience,
        active=True, is_demo_coordinator=False
    ))

coordinators_df = spark.createDataFrame(coordinators_rows)
coordinators_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{FULL_SCHEMA}.care_coordinators")

print(f"✅ Created {FULL_SCHEMA}.care_coordinators with {len(coordinators_rows):,} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary & Verification

# COMMAND ----------

print("\n" + "="*60)
print("🎉 SYNTHETIC DATA GENERATION COMPLETE!")
print("="*60)

# Verify all tables
tables = spark.sql(f"SHOW TABLES IN {FULL_SCHEMA}").collect()
print(f"\n📊 Created {len(tables)} tables in {FULL_SCHEMA}:")

for table in tables:
    table_name = table.tableName
    count = spark.table(f"{FULL_SCHEMA}.{table_name}").count()
    print(f"   ✅ {table_name}: {count:,} records")

print("\n" + "="*60)
print("🎯 Demo Data Verification:")
print("="*60)

# Verify demo data
recent_discharges = spark.sql(f"""
    SELECT COUNT(*) as count
    FROM {FULL_SCHEMA}.encounters
    WHERE discharge_date >= CURRENT_DATE() - 7
""").first().count

chf_patients = spark.sql(f"""
    SELECT COUNT(DISTINCT patient_id) as count
    FROM {FULL_SCHEMA}.diagnoses d
    JOIN {FULL_SCHEMA}.encounters e ON d.encounter_id = e.encounter_id
    WHERE d.is_demo_chf = TRUE
""").first().count

copd_patients = spark.sql(f"""
    SELECT COUNT(DISTINCT patient_id) as count
    FROM {FULL_SCHEMA}.diagnoses d
    JOIN {FULL_SCHEMA}.encounters e ON d.encounter_id = e.encounter_id
    WHERE d.is_demo_copd = TRUE
""").first().count

high_risk = spark.table(f"{FULL_SCHEMA}.risk_scores").filter(col("risk_score") > 0.7).count()

print(f"\n✅ Recent discharges (last 7 days): {recent_discharges}")
print(f"✅ CHF patients: {chf_patients}")
print(f"✅ COPD patients: {copd_patients}")
print(f"✅ High-risk patients (>0.7): {high_risk}")
print(f"✅ 30-day readmissions: {spark.table(f'{FULL_SCHEMA}.readmissions').filter(col('is_30_day') == True).count()}")
print(f"✅ Transportation barriers: {spark.table(f'{FULL_SCHEMA}.sdoh').filter(col('transportation_barrier') == True).count()}")
print(f"✅ Housing instability: {spark.table(f'{FULL_SCHEMA}.sdoh').filter(col('housing_instability') == True).count()}")
print(f"✅ CHF/COPD coordinators: {spark.table(f'{FULL_SCHEMA}.care_coordinators').filter(col('is_demo_coordinator') == True).count()}")

print("\n" + "="*60)
print("📝 Next Step: Run 01_create_tools/01_create_tools.py")
print("="*60)
