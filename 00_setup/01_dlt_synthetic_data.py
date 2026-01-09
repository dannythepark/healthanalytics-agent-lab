# Databricks notebook source
# MAGIC %md
# MAGIC # HealthAnalytics AI - Synthetic Data Pipeline
# MAGIC
# MAGIC ## 🏥 HIPAA-Compliant Synthetic Healthcare Data
# MAGIC
# MAGIC This Delta Live Tables (DLT) pipeline generates **synthetic healthcare data** for the Clinical Analytics Agent demo.
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
# MAGIC This pipeline plants specific data to make the demo work perfectly:
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
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import dlt
from pyspark.sql.functions import *
from pyspark.sql.types import *
from faker import Faker
import random
from datetime import datetime, timedelta
import json

# Initialize Faker for synthetic data generation
fake = Faker()
Faker.seed(42)  # Reproducible results
random.seed(42)

# Import configuration
import sys
sys.path.append("/Workspace/Users/...")  # Update with your path
from config import CATALOG, SCHEMA, DEMO_CONFIG, CLINICAL_CODES

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table 1: Patients (10,000 records)
# MAGIC
# MAGIC Core patient registry with demographics.

# COMMAND ----------

@dlt.table(
    name="patients",
    comment="Synthetic patient demographics (HIPAA-compliant, no real PHI)",
    table_properties={"quality": "gold", "pipelines.autoOptimize.managed": "true"}
)
def patients():
    """Generate synthetic patient data"""

    patients_data = []

    for i in range(10_000):
        patient_id = f"patient-{str(i+1).zfill(6)}"
        mrn = f"MRN{random.randint(100000, 999999)}"

        # Demographics
        gender = random.choice(["M", "F", "Other"])
        age = random.randint(18, 95)

        # Realistic name based on gender
        if gender == "M":
            first_name = fake.first_name_male()
        elif gender == "F":
            first_name = fake.first_name_female()
        else:
            first_name = fake.first_name()

        last_name = fake.last_name()

        # Location
        zip_code = fake.zipcode()
        city = fake.city()
        state = fake.state_abbr()

        # Race/ethnicity (realistic distribution)
        race_ethnicity = random.choices(
            ["White", "Black or African American", "Hispanic or Latino", "Asian", "Other"],
            weights=[60, 13, 18, 6, 3]
        )[0]

        patients_data.append({
            "patient_id": patient_id,
            "mrn": mrn,
            "first_name": first_name,
            "last_name": last_name,
            "age": age,
            "gender": gender,
            "zip_code": zip_code,
            "city": city,
            "state": state,
            "race_ethnicity": race_ethnicity,
            "created_at": fake.date_time_between(start_date="-5y", end_date="now")
        })

    return spark.createDataFrame(patients_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table 2: Encounters (50,000 records)
# MAGIC
# MAGIC Hospital admissions and emergency department visits.
# MAGIC
# MAGIC **CRITICAL:** Plants 127 discharges in the last 7 days for the demo.

# COMMAND ----------

@dlt.table(
    name="encounters",
    comment="Inpatient admissions and ED visits with discharge dates",
    table_properties={"quality": "gold"}
)
@dlt.expect_or_drop("valid_discharge", "discharge_date >= admission_date")
def encounters():
    """Generate synthetic encounter data with planted recent discharges"""

    # Read patients table
    patients_df = dlt.read("patients")
    patient_ids = [row.patient_id for row in patients_df.select("patient_id").collect()]

    encounters_data = []
    encounter_counter = 1

    # Facilities in the health system
    facilities = [
        "Central Hospital", "North Medical Center", "South Regional Hospital",
        "East Community Hospital", "West Medical Center", "Children's Hospital",
        "Cardiac Specialty Center", "Cancer Center", "Rehabilitation Hospital",
        "Behavioral Health Center", "Urgent Care - Main", "Urgent Care - North"
    ]

    # Encounter types
    encounter_types = ["Inpatient", "Emergency", "Observation", "Outpatient Surgery"]

    # STEP 1: Plant 127 recent discharges (last 7 days) for demo
    demo_discharge_count = DEMO_CONFIG["recent_discharge_count"]
    demo_patients = random.sample(patient_ids, demo_discharge_count)

    for patient_id in demo_patients:
        encounter_id = f"encounter-{str(encounter_counter).zfill(8)}"
        encounter_counter += 1

        # Discharge in the last 7 days
        discharge_date = fake.date_time_between(start_date="-7d", end_date="now")
        length_of_stay = random.randint(1, 14)
        admission_date = discharge_date - timedelta(days=length_of_stay)

        encounters_data.append({
            "encounter_id": encounter_id,
            "patient_id": patient_id,
            "admission_date": admission_date,
            "discharge_date": discharge_date,
            "encounter_type": random.choice(["Inpatient", "Emergency"]),
            "facility": random.choice(facilities),
            "length_of_stay": length_of_stay,
            "discharge_disposition": random.choice(["Home", "Home Health", "SNF", "Rehab"]),
            "is_demo_recent_discharge": True  # Flag for verification
        })

    # STEP 2: Generate remaining historical encounters
    remaining_encounters = 50_000 - demo_discharge_count

    for _ in range(remaining_encounters):
        encounter_id = f"encounter-{str(encounter_counter).zfill(8)}"
        encounter_counter += 1

        patient_id = random.choice(patient_ids)

        # Historical dates (up to 3 years ago)
        admission_date = fake.date_time_between(start_date="-3y", end_date="-8d")
        length_of_stay = random.randint(1, 30)
        discharge_date = admission_date + timedelta(days=length_of_stay)

        encounters_data.append({
            "encounter_id": encounter_id,
            "patient_id": patient_id,
            "admission_date": admission_date,
            "discharge_date": discharge_date,
            "encounter_type": random.choice(encounter_types),
            "facility": random.choice(facilities),
            "length_of_stay": length_of_stay,
            "discharge_disposition": random.choice([
                "Home", "Home Health", "SNF", "Rehab", "AMA", "Deceased"
            ]),
            "is_demo_recent_discharge": False
        })

    return spark.createDataFrame(encounters_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table 3: Diagnoses (150,000 records)
# MAGIC
# MAGIC Clinical diagnoses using ICD-10 codes.
# MAGIC
# MAGIC **CRITICAL:** Plants CHF and COPD diagnoses for recent discharges.

# COMMAND ----------

@dlt.table(
    name="diagnoses",
    comment="Clinical diagnoses using ICD-10 coding",
    table_properties={"quality": "gold"}
)
def diagnoses():
    """Generate synthetic diagnosis data with planted CHF/COPD cases"""

    # Read encounters
    encounters_df = dlt.read("encounters")
    recent_encounters = [
        row.encounter_id
        for row in encounters_df.filter(col("is_demo_recent_discharge") == True).collect()
    ]
    all_encounters = [row.encounter_id for row in encounters_df.select("encounter_id").collect()]

    # Common diagnoses with ICD-10 codes
    common_diagnoses = [
        ("I50.9", "Congestive Heart Failure, unspecified"),
        ("I50.23", "Acute on chronic systolic heart failure"),
        ("J44.1", "Chronic obstructive pulmonary disease with acute exacerbation"),
        ("J44.0", "Chronic obstructive pulmonary disease with acute lower respiratory infection"),
        ("E11.9", "Type 2 diabetes mellitus without complications"),
        ("E11.65", "Type 2 diabetes mellitus with hyperglycemia"),
        ("I10", "Essential (primary) hypertension"),
        ("I48.91", "Atrial fibrillation, unspecified"),
        ("N18.3", "Chronic kidney disease, stage 3"),
        ("J18.9", "Pneumonia, unspecified organism"),
        ("A41.9", "Sepsis, unspecified organism"),
        ("I21.9", "Acute myocardial infarction, unspecified"),
        ("I63.9", "Cerebral infarction, unspecified"),
        ("K92.2", "Gastrointestinal hemorrhage, unspecified"),
        ("N39.0", "Urinary tract infection, site not specified"),
        ("F32.9", "Major depressive disorder, single episode, unspecified"),
        ("M19.90", "Osteoarthritis, unspecified"),
        ("E78.5", "Hyperlipidemia, unspecified"),
        ("G47.33", "Obstructive sleep apnea"),
        ("K21.9", "Gastro-esophageal reflux disease"),
    ]

    diagnoses_data = []
    diagnosis_counter = 1

    # STEP 1: Plant CHF diagnoses (23 patients from recent discharges)
    chf_encounters = random.sample(recent_encounters, DEMO_CONFIG["chf_patient_count"])

    for encounter_id in chf_encounters:
        # Primary CHF diagnosis
        icd10, description = random.choice([d for d in common_diagnoses if d[0].startswith("I50")])
        diagnoses_data.append({
            "diagnosis_id": f"diagnosis-{str(diagnosis_counter).zfill(9)}",
            "encounter_id": encounter_id,
            "icd10_code": icd10,
            "description": description,
            "is_primary": True,
            "diagnosis_date": fake.date_time_between(start_date="-7d", end_date="now"),
            "is_demo_chf": True
        })
        diagnosis_counter += 1

        # Add 2-4 secondary diagnoses
        for _ in range(random.randint(2, 4)):
            icd10, description = random.choice(common_diagnoses)
            diagnoses_data.append({
                "diagnosis_id": f"diagnosis-{str(diagnosis_counter).zfill(9)}",
                "encounter_id": encounter_id,
                "icd10_code": icd10,
                "description": description,
                "is_primary": False,
                "diagnosis_date": fake.date_time_between(start_date="-7d", end_date="now"),
                "is_demo_chf": False
            })
            diagnosis_counter += 1

    # STEP 2: Plant COPD diagnoses (19 patients from recent discharges, non-overlapping with CHF)
    remaining_recent = [e for e in recent_encounters if e not in chf_encounters]
    copd_encounters = random.sample(remaining_recent, DEMO_CONFIG["copd_patient_count"])

    for encounter_id in copd_encounters:
        # Primary COPD diagnosis
        icd10, description = random.choice([d for d in common_diagnoses if d[0].startswith("J44")])
        diagnoses_data.append({
            "diagnosis_id": f"diagnosis-{str(diagnosis_counter).zfill(9)}",
            "encounter_id": encounter_id,
            "icd10_code": icd10,
            "description": description,
            "is_primary": True,
            "diagnosis_date": fake.date_time_between(start_date="-7d", end_date="now"),
            "is_demo_copd": True
        })
        diagnosis_counter += 1

        # Add 2-4 secondary diagnoses
        for _ in range(random.randint(2, 4)):
            icd10, description = random.choice(common_diagnoses)
            diagnoses_data.append({
                "diagnosis_id": f"diagnosis-{str(diagnosis_counter).zfill(9)}",
                "encounter_id": encounter_id,
                "icd10_code": icd10,
                "description": description,
                "is_primary": False,
                "diagnosis_date": fake.date_time_between(start_date="-7d", end_date="now"),
                "is_demo_copd": False
            })
            diagnosis_counter += 1

    # STEP 3: Generate diagnoses for remaining encounters
    target_total = 150_000
    current_count = len(diagnoses_data)

    while current_count < target_total:
        encounter_id = random.choice(all_encounters)
        num_diagnoses = random.randint(1, 6)

        for i in range(num_diagnoses):
            if current_count >= target_total:
                break

            icd10, description = random.choice(common_diagnoses)
            diagnoses_data.append({
                "diagnosis_id": f"diagnosis-{str(diagnosis_counter).zfill(9)}",
                "encounter_id": encounter_id,
                "icd10_code": icd10,
                "description": description,
                "is_primary": (i == 0),
                "diagnosis_date": fake.date_time_between(start_date="-3y", end_date="now"),
                "is_demo_chf": False,
                "is_demo_copd": False
            })
            diagnosis_counter += 1
            current_count += 1

    return spark.createDataFrame(diagnoses_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table 4: Readmissions (5,000 records)
# MAGIC
# MAGIC All-cause readmissions with 30-day flagging.
# MAGIC
# MAGIC **CRITICAL:** Plants 32 patients with prior 30-day readmissions.

# COMMAND ----------

@dlt.table(
    name="readmissions",
    comment="All-cause readmissions with 30-day CMS quality metric",
    table_properties={"quality": "gold"}
)
@dlt.expect_or_drop("valid_readmission", "days_between >= 0")
def readmissions():
    """Generate synthetic readmission data"""

    # Read encounters and get patient-level data
    encounters_df = dlt.read("encounters").toPandas()
    encounters_df = encounters_df.sort_values(["patient_id", "admission_date"])

    readmissions_data = []
    readmission_counter = 1

    # Find actual readmissions (same patient, multiple encounters)
    patient_encounters = encounters_df.groupby("patient_id")["encounter_id"].apply(list).to_dict()

    patients_with_readmissions = [
        pid for pid, encs in patient_encounters.items() if len(encs) >= 2
    ]

    # STEP 1: Plant 32 patients with 30-day readmissions
    demo_readmit_patients = random.sample(
        patients_with_readmissions,
        min(DEMO_CONFIG["prior_readmission_count"], len(patients_with_readmissions))
    )

    for patient_id in demo_readmit_patients:
        patient_encs = encounters_df[encounters_df["patient_id"] == patient_id].to_dict("records")

        if len(patient_encs) >= 2:
            original = patient_encs[-2]
            readmit = patient_encs[-1]

            days_between = (readmit["admission_date"] - original["discharge_date"]).days

            # Force it to be within 30 days for demo
            if days_between > 30:
                days_between = random.randint(1, 30)

            readmissions_data.append({
                "readmission_id": f"readmit-{str(readmission_counter).zfill(7)}",
                "patient_id": patient_id,
                "original_encounter_id": original["encounter_id"],
                "readmit_encounter_id": readmit["encounter_id"],
                "original_discharge_date": original["discharge_date"],
                "readmit_admission_date": readmit["admission_date"],
                "days_between": days_between,
                "is_30_day": True,  # Forced for demo
                "is_demo_readmission": True
            })
            readmission_counter += 1

    # STEP 2: Generate remaining readmissions
    remaining_patients = [p for p in patients_with_readmissions if p not in demo_readmit_patients]

    for _ in range(5_000 - len(demo_readmit_patients)):
        if not remaining_patients:
            break

        patient_id = random.choice(remaining_patients)
        patient_encs = encounters_df[encounters_df["patient_id"] == patient_id].to_dict("records")

        if len(patient_encs) >= 2:
            original = random.choice(patient_encs[:-1])
            later_encs = [e for e in patient_encs if e["admission_date"] > original["discharge_date"]]

            if later_encs:
                readmit = random.choice(later_encs)
                days_between = (readmit["admission_date"] - original["discharge_date"]).days

                readmissions_data.append({
                    "readmission_id": f"readmit-{str(readmission_counter).zfill(7)}",
                    "patient_id": patient_id,
                    "original_encounter_id": original["encounter_id"],
                    "readmit_encounter_id": readmit["encounter_id"],
                    "original_discharge_date": original["discharge_date"],
                    "readmit_admission_date": readmit["admission_date"],
                    "days_between": days_between,
                    "is_30_day": (days_between <= 30),
                    "is_demo_readmission": False
                })
                readmission_counter += 1

    return spark.createDataFrame(readmissions_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table 5: Risk Scores (10,000 records)
# MAGIC
# MAGIC Predictive readmission risk scores (0.0-1.0).
# MAGIC
# MAGIC **CRITICAL:** Plants 18 high-risk patients (score > 0.7) from CHF/COPD cohort.

# COMMAND ----------

@dlt.table(
    name="risk_scores",
    comment="Predictive readmission risk scores (0.0-1.0 scale)",
    table_properties={"quality": "gold"}
)
@dlt.expect("valid_risk_score", "risk_score >= 0 AND risk_score <= 1")
def risk_scores():
    """Generate synthetic risk scores with planted high-risk patients"""

    # Read patients and encounters
    patients_df = dlt.read("patients")
    encounters_df = dlt.read("encounters")
    diagnoses_df = dlt.read("diagnoses")

    # Get CHF/COPD patients from recent discharges
    demo_encounters = encounters_df.filter(col("is_demo_recent_discharge") == True)
    demo_diagnoses = diagnoses_df.filter(
        (col("is_demo_chf") == True) | (col("is_demo_copd") == True)
    )

    demo_encounter_ids = [row.encounter_id for row in demo_diagnoses.select("encounter_id").collect()]
    demo_patient_ids = [
        row.patient_id
        for row in demo_encounters.filter(col("encounter_id").isin(demo_encounter_ids)).select("patient_id").collect()
    ]

    all_patient_ids = [row.patient_id for row in patients_df.select("patient_id").collect()]

    risk_scores_data = []

    # STEP 1: Plant 18 high-risk patients from CHF/COPD cohort
    high_risk_patients = random.sample(demo_patient_ids, min(DEMO_CONFIG["high_risk_count"], len(demo_patient_ids)))

    for patient_id in high_risk_patients:
        risk_score = round(random.uniform(0.71, 0.95), 3)

        # Risk factors for high-risk patients
        risk_factors = random.sample([
            "Prior 30-day readmission",
            "CHF/COPD diagnosis",
            "Multiple comorbidities",
            "Age > 65",
            "Polypharmacy (>10 medications)",
            "Low health literacy",
            "Frequent ED utilization",
            "Social barriers (transportation/housing)"
        ], k=random.randint(3, 5))

        risk_scores_data.append({
            "patient_id": patient_id,
            "risk_score": risk_score,
            "risk_category": "High",
            "risk_factors": json.dumps(risk_factors),
            "last_calculated": fake.date_time_between(start_date="-7d", end_date="now"),
            "model_version": "v2.3.1",
            "is_demo_high_risk": True
        })

    # STEP 2: Moderate risk for remaining CHF/COPD patients
    moderate_risk_patients = [p for p in demo_patient_ids if p not in high_risk_patients]

    for patient_id in moderate_risk_patients:
        risk_score = round(random.uniform(0.40, 0.70), 3)

        risk_factors = random.sample([
            "Chronic condition (CHF/COPD)",
            "Age > 50",
            "Hypertension",
            "Diabetes"
        ], k=random.randint(1, 3))

        risk_scores_data.append({
            "patient_id": patient_id,
            "risk_score": risk_score,
            "risk_category": "Moderate",
            "risk_factors": json.dumps(risk_factors),
            "last_calculated": fake.date_time_between(start_date="-7d", end_date="now"),
            "model_version": "v2.3.1",
            "is_demo_high_risk": False
        })

    # STEP 3: Generate remaining risk scores
    used_patients = set(demo_patient_ids)
    remaining_patients = [p for p in all_patient_ids if p not in used_patients]

    for _ in range(10_000 - len(risk_scores_data)):
        if not remaining_patients:
            break

        patient_id = random.choice(remaining_patients)
        remaining_patients.remove(patient_id)

        # Most patients are low-moderate risk
        risk_score = round(random.triangular(0.1, 0.8, 0.3), 3)

        if risk_score < 0.3:
            category = "Low"
        elif risk_score < 0.7:
            category = "Moderate"
        else:
            category = "High"

        risk_factors = random.sample([
            "Age",
            "Chronic conditions",
            "Recent hospitalization",
            "Medication adherence"
        ], k=random.randint(0, 2))

        risk_scores_data.append({
            "patient_id": patient_id,
            "risk_score": risk_score,
            "risk_category": category,
            "risk_factors": json.dumps(risk_factors),
            "last_calculated": fake.date_time_between(start_date="-30d", end_date="now"),
            "model_version": "v2.3.1",
            "is_demo_high_risk": False
        })

    return spark.createDataFrame(risk_scores_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table 6: Social Determinants of Health (8,000 records)
# MAGIC
# MAGIC Social barriers affecting health outcomes.
# MAGIC
# MAGIC **CRITICAL:** Plants transportation and housing barriers for high-risk patients.

# COMMAND ----------

@dlt.table(
    name="sdoh",
    comment="Social Determinants of Health (SDOH) barriers",
    table_properties={"quality": "gold"}
)
def sdoh():
    """Generate synthetic SDOH data with planted barriers"""

    # Read patients and risk scores
    patients_df = dlt.read("patients")
    risk_scores_df = dlt.read("risk_scores")

    # Get high-risk patients
    high_risk_patients = [
        row.patient_id
        for row in risk_scores_df.filter(col("is_demo_high_risk") == True).select("patient_id").collect()
    ]

    all_patients = [row.patient_id for row in patients_df.select("patient_id").collect()]

    sdoh_data = []

    # STEP 1: Plant transportation barriers (5 high-risk patients)
    transport_barrier_patients = random.sample(
        high_risk_patients,
        min(DEMO_CONFIG["transportation_barrier_count"], len(high_risk_patients))
    )

    for patient_id in transport_barrier_patients:
        sdoh_data.append({
            "patient_id": patient_id,
            "housing_instability": random.choice([True, False]),
            "transportation_barrier": True,  # Planted
            "food_insecurity": random.choice([True, False]),
            "social_isolation": random.choice([True, False]),
            "financial_strain": random.choice([True, False]),
            "utility_assistance_needed": random.choice([True, False]),
            "last_assessed": fake.date_time_between(start_date="-30d", end_date="now"),
            "is_demo_transport": True
        })

    # STEP 2: Plant housing instability (3 high-risk patients, non-overlapping)
    remaining_high_risk = [p for p in high_risk_patients if p not in transport_barrier_patients]
    housing_barrier_patients = random.sample(
        remaining_high_risk,
        min(DEMO_CONFIG["housing_instability_count"], len(remaining_high_risk))
    )

    for patient_id in housing_barrier_patients:
        sdoh_data.append({
            "patient_id": patient_id,
            "housing_instability": True,  # Planted
            "transportation_barrier": random.choice([True, False]),
            "food_insecurity": random.choice([True, False]),
            "social_isolation": random.choice([True, False]),
            "financial_strain": True,  # Often correlated
            "utility_assistance_needed": random.choice([True, False]),
            "last_assessed": fake.date_time_between(start_date="-30d", end_date="now"),
            "is_demo_housing": True
        })

    # STEP 3: Generate remaining SDOH records
    used_patients = set(transport_barrier_patients + housing_barrier_patients)
    remaining_patients = [p for p in all_patients if p not in used_patients]

    for _ in range(8_000 - len(sdoh_data)):
        if not remaining_patients:
            break

        patient_id = random.choice(remaining_patients)
        remaining_patients.remove(patient_id)

        # Most patients have 0-2 barriers
        num_barriers = random.choices([0, 1, 2, 3, 4], weights=[40, 30, 20, 7, 3])[0]

        barriers = {
            "housing_instability": False,
            "transportation_barrier": False,
            "food_insecurity": False,
            "social_isolation": False,
            "financial_strain": False,
            "utility_assistance_needed": False
        }

        if num_barriers > 0:
            barrier_keys = random.sample(list(barriers.keys()), k=num_barriers)
            for key in barrier_keys:
                barriers[key] = True

        sdoh_data.append({
            "patient_id": patient_id,
            **barriers,
            "last_assessed": fake.date_time_between(start_date="-90d", end_date="now"),
            "is_demo_transport": False,
            "is_demo_housing": False
        })

    return spark.createDataFrame(sdoh_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table 7: Care Coordinators (15 records)
# MAGIC
# MAGIC Care management team capacity and specialties.
# MAGIC
# MAGIC **CRITICAL:** Plants 2 coordinators with CHF/COPD specialty and available capacity.

# COMMAND ----------

@dlt.table(
    name="care_coordinators",
    comment="Care coordination team capacity and specialties",
    table_properties={"quality": "gold"}
)
def care_coordinators():
    """Generate care coordinator data with planted CHF/COPD specialists"""

    coordinators_data = []

    # STEP 1: Plant 2 coordinators with CHF/COPD specialty and capacity
    coordinators_data.append({
        "coordinator_id": "coord-001",
        "name": "Sarah Johnson, RN",
        "title": "Senior Care Coordinator",
        "current_caseload": 18,
        "max_caseload": 30,
        "available_capacity": 12,
        "specialties": json.dumps(["CHF", "COPD", "Cardiology"]),
        "years_experience": 12,
        "active": True,
        "is_demo_coordinator": True
    })

    coordinators_data.append({
        "coordinator_id": "coord-002",
        "name": "Michael Chen, MSW",
        "title": "Care Coordinator",
        "current_caseload": 22,
        "max_caseload": 30,
        "available_capacity": 8,
        "specialties": json.dumps(["COPD", "Pulmonary", "Geriatrics"]),
        "years_experience": 8,
        "active": True,
        "is_demo_coordinator": True
    })

    # STEP 2: Add remaining coordinators with various specialties
    other_coordinators = [
        ("coord-003", "Jessica Martinez, RN", "Care Coordinator", 28, 30, ["Diabetes", "Endocrinology"], 6),
        ("coord-004", "David Thompson, BSN", "Care Coordinator", 25, 30, ["Oncology", "Palliative Care"], 10),
        ("coord-005", "Emily Rodriguez, RN", "Senior Care Coordinator", 20, 30, ["Pediatrics", "Neonatology"], 15),
        ("coord-006", "James Wilson, MSW", "Care Coordinator", 30, 30, ["Behavioral Health", "Substance Abuse"], 7),
        ("coord-007", "Maria Garcia, RN", "Care Coordinator", 24, 30, ["Nephrology", "Dialysis"], 9),
        ("coord-008", "Robert Lee, BSN", "Care Coordinator", 26, 30, ["Orthopedics", "Rehabilitation"], 5),
        ("coord-009", "Linda Anderson, MSW", "Senior Care Coordinator", 15, 30, ["Geriatrics", "Dementia Care"], 18),
        ("coord-010", "Christopher Brown, RN", "Care Coordinator", 30, 30, ["Trauma", "Emergency"], 11),
        ("coord-011", "Patricia Davis, BSN", "Care Coordinator", 29, 30, ["Maternal Health", "OB/GYN"], 8),
        ("coord-012", "Daniel Miller, MSW", "Care Coordinator", 21, 30, ["Homeless Outreach", "SDOH"], 6),
        ("coord-013", "Jennifer Taylor, RN", "Care Coordinator", 27, 30, ["Infectious Disease", "HIV/AIDS"], 13),
    ]

    for coord_id, name, title, caseload, max_caseload, specialties, experience in other_coordinators:
        coordinators_data.append({
            "coordinator_id": coord_id,
            "name": name,
            "title": title,
            "current_caseload": caseload,
            "max_caseload": max_caseload,
            "available_capacity": max_caseload - caseload,
            "specialties": json.dumps(specialties),
            "years_experience": experience,
            "active": True,
            "is_demo_coordinator": False
        })

    return spark.createDataFrame(coordinators_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline Summary

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### ✅ Pipeline Complete!
# MAGIC
# MAGIC You've successfully created 7 tables with synthetic healthcare data:
# MAGIC
# MAGIC 1. ✅ **patients** (10,000 records)
# MAGIC 2. ✅ **encounters** (50,000 records)
# MAGIC 3. ✅ **diagnoses** (150,000 records)
# MAGIC 4. ✅ **readmissions** (5,000 records)
# MAGIC 5. ✅ **risk_scores** (10,000 records)
# MAGIC 6. ✅ **sdoh** (8,000 records)
# MAGIC 7. ✅ **care_coordinators** (15 records)
# MAGIC
# MAGIC ### 🎯 Demo Data Planted
# MAGIC
# MAGIC - 127 recent discharges in last 7 days
# MAGIC - 23 CHF patients, 19 COPD patients
# MAGIC - 32 patients with prior 30-day readmissions
# MAGIC - 18 high-risk patients (score > 0.7)
# MAGIC - 5 patients with transportation barriers
# MAGIC - 3 patients with housing instability
# MAGIC - 2 care coordinators with CHF/COPD specialty and capacity
# MAGIC
# MAGIC ### 📊 Next Step
# MAGIC
# MAGIC Run `01_create_tools/01_create_tools.py` to create the analytics functions!
