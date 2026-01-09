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
# MAGIC **⚠️ IMPORTANT: Update the CATALOG name below!**

# COMMAND ----------

# Configuration - UPDATE THIS!
CATALOG = "danny_park"  # ← Change to your catalog name
SCHEMA = "healthanalytics_ai"
FULL_SCHEMA = f"{CATALOG}.{SCHEMA}"

# Create schema if it doesn't exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {FULL_SCHEMA}")
print(f"✅ Using schema: {FULL_SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table 1: Patients (10,000 records)

# COMMAND ----------

print("Generating 10,000 patients...")

# Generate patients using Row objects (more compatible with Serverless)
patients_rows = []
for i in range(10_000):
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

print("Generating 50,000 encounters...")

# Read patients to get IDs (using Spark operations, not collect)
patient_ids_df = spark.table(f"{FULL_SCHEMA}.patients").select("patient_id")
patient_ids = [row.patient_id for row in patient_ids_df.collect()]

facilities = [
    "Central Hospital", "North Medical Center", "South Regional Hospital",
    "East Community Hospital", "West Medical Center", "Children's Hospital"
]

encounters_rows = []
encounter_counter = 1

# STEP 1: Plant 127 recent discharges (last 7 days)
print("  → Planting 127 recent discharges...")
demo_patients = random.sample(patient_ids, 127)

for patient_id in demo_patients:
    discharge_date = fake.date_time_between(start_date="-7d", end_date="now")
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
print("  → Generating 49,873 historical encounters...")
for _ in range(50_000 - 127):
    patient_id = random.choice(patient_ids)
    admission_date = fake.date_time_between(start_date="-3y", end_date="-8d")
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

print("Generating 150,000 diagnoses...")

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

# STEP 1: Plant 23 CHF patients
print("  → Planting 23 CHF patients...")
chf_encounters = random.sample(recent_encounter_ids, 23)

for encounter_id in chf_encounters:
    icd10, description = random.choice([d for d in common_diagnoses if d[0].startswith("I50")])
    diagnoses_rows.append(Row(
        diagnosis_id=f"diagnosis-{str(diagnosis_counter).zfill(9)}",
        encounter_id=encounter_id,
        icd10_code=icd10,
        description=description,
        is_primary=True,
        diagnosis_date=fake.date_time_between(start_date="-7d", end_date="now"),
        is_demo_chf=True,
        is_demo_copd=False
    ))
    diagnosis_counter += 1

# STEP 2: Plant 19 COPD patients
print("  → Planting 19 COPD patients...")
remaining_recent = [e for e in recent_encounter_ids if e not in chf_encounters]
copd_encounters = random.sample(remaining_recent, 19)

for encounter_id in copd_encounters:
    icd10, description = random.choice([d for d in common_diagnoses if d[0].startswith("J44")])
    diagnoses_rows.append(Row(
        diagnosis_id=f"diagnosis-{str(diagnosis_counter).zfill(9)}",
        encounter_id=encounter_id,
        icd10_code=icd10,
        description=description,
        is_primary=True,
        diagnosis_date=fake.date_time_between(start_date="-7d", end_date="now"),
        is_demo_chf=False,
        is_demo_copd=True
    ))
    diagnosis_counter += 1

# STEP 3: Generate remaining diagnoses (in batches for better performance)
print("  → Generating remaining diagnoses...")
batch_size = 10000
while len(diagnoses_rows) < 150_000:
    for _ in range(builtins.min(batch_size, 150_000 - len(diagnoses_rows))):
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

print(f"✅ Created {FULL_SCHEMA}.diagnoses with 150,000 records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table 4: Readmissions (5,000 records with 32 prior 30-day)

# COMMAND ----------

print("Generating 5,000 readmissions...")

# Use Spark SQL to find patients with multiple encounters
multi_encounter_patients = spark.sql(f"""
    SELECT patient_id, COUNT(*) as encounter_count
    FROM {FULL_SCHEMA}.encounters
    GROUP BY patient_id
    HAVING COUNT(*) >= 2
""").collect()

patient_with_multiple = [row.patient_id for row in multi_encounter_patients]

readmissions_rows = []
readmission_counter = 1

# Plant 32 patients with 30-day readmissions
print("  → Planting 32 patients with 30-day readmissions...")
demo_readmit_patients = random.sample(patient_with_multiple, builtins.min(32, len(patient_with_multiple)))

for patient_id in demo_readmit_patients:
    patient_encs = spark.table(f"{FULL_SCHEMA}.encounters").filter(
        col("patient_id") == patient_id
    ).orderBy("admission_date").collect()

    if len(patient_encs) >= 2:
        original = patient_encs[-2]
        readmit = patient_encs[-1]
        days_between = random.randint(1, 30)

        readmissions_rows.append(Row(
            readmission_id=f"readmit-{str(readmission_counter).zfill(7)}",
            patient_id=patient_id,
            original_encounter_id=original.encounter_id,
            readmit_encounter_id=readmit.encounter_id,
            original_discharge_date=original.discharge_date,
            readmit_admission_date=readmit.admission_date,
            days_between=days_between,
            is_30_day=True,
            is_demo_readmission=True
        ))
        readmission_counter += 1

# Generate remaining readmissions
print("  → Generating remaining readmissions...")
remaining_patients = [p for p in patient_with_multiple if p not in demo_readmit_patients]

for patient_id in random.sample(remaining_patients, builtins.min(5_000 - len(readmissions_rows), len(remaining_patients))):
    patient_encs = spark.table(f"{FULL_SCHEMA}.encounters").filter(
        col("patient_id") == patient_id
    ).orderBy("admission_date").collect()

    if len(patient_encs) >= 2:
        original = patient_encs[0]
        readmit = patient_encs[-1]

        if readmit.admission_date > original.discharge_date:
            days_between = (readmit.admission_date - original.discharge_date).days

            readmissions_rows.append(Row(
                readmission_id=f"readmit-{str(readmission_counter).zfill(7)}",
                patient_id=patient_id,
                original_encounter_id=original.encounter_id,
                readmit_encounter_id=readmit.encounter_id,
                original_discharge_date=original.discharge_date,
                readmit_admission_date=readmit.admission_date,
                days_between=days_between,
                is_30_day=(days_between <= 30),
                is_demo_readmission=False
            ))
            readmission_counter += 1

readmissions_df = spark.createDataFrame(readmissions_rows)
readmissions_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{FULL_SCHEMA}.readmissions")

print(f"✅ Created {FULL_SCHEMA}.readmissions with {len(readmissions_rows):,} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table 5: Risk Scores (10,000 records with 18 high-risk)

# COMMAND ----------

print("Generating 10,000 risk scores...")

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

# Plant 18 high-risk patients
print("  → Planting 18 high-risk patients (score > 0.7)...")
high_risk_patients = random.sample(chf_copd_patient_ids, builtins.min(18, len(chf_copd_patient_ids)))

for patient_id in high_risk_patients:
    risk_score = builtins.round(random.uniform(0.71, 0.95), 3)
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

for patient_id in remaining_patients[:10_000 - len(high_risk_patients)]:
    risk_score = builtins.round(random.triangular(0.1, 0.8, 0.3), 3)
    category = "Low" if risk_score < 0.3 else "Moderate" if risk_score < 0.7 else "High"

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

print("Generating 8,000 SDOH records...")

high_risk_patient_ids = [
    row.patient_id
    for row in spark.table(f"{FULL_SCHEMA}.risk_scores").filter(col("is_demo_high_risk") == True).collect()
]

sdoh_rows = []

# Plant 5 with transportation barriers
print("  → Planting 5 patients with transportation barriers...")
transport_patients = random.sample(high_risk_patient_ids, builtins.min(5, len(high_risk_patient_ids)))

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

# Plant 3 with housing instability
print("  → Planting 3 patients with housing instability...")
remaining_high_risk = [p for p in high_risk_patient_ids if p not in transport_patients]
housing_patients = random.sample(remaining_high_risk, builtins.min(3, len(remaining_high_risk)))

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

for patient_id in remaining[:8_000 - len(sdoh_rows)]:
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

print("Generating 15 care coordinators...")

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

# Add 10 more coordinators
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

for coord_id, name, caseload, specialties, experience in more_coords:
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
