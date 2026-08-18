import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# ===== CONFIG =====

# Number of days to generate
n_days = 10  # adjust as needed

# Base date
start_date = datetime(2025, 1, 1)

# OR rooms (skip 13), plus cysto and MRI
rooms = [
    "OR1", "OR2",
    "OR3", "OR4",
    "OR5", "OR6",
    "OR7",
    "OR8", "OR9", "OR10", "OR11", "OR12",
    "OR14", "OR15", "OR16",
    "OR17",
    "OR18",
    "OR19", "OR20",
    "OR21", "OR22", "OR23",
    "OR24", "OR25",
    "OR26",
    "CystoA", "CystoB",
    "FarscanMRI"
]

# Service mapping per room with typical cases/day range
room_services = {
    "OR1":  {"service": "Cardiac", "min_cases": 1, "max_cases": 2},
    "OR2":  {"service": "Cardiac", "min_cases": 1, "max_cases": 2},
    "OR3":  {"service": "Abdominal Transplant", "min_cases": 1, "max_cases": 2},
    "OR4":  {"service": "Abdominal Transplant", "min_cases": 1, "max_cases": 2},
    "OR5":  {"service": "ENT", "min_cases": 3, "max_cases": 4},
    "OR6":  {"service": "ENT", "min_cases": 3, "max_cases": 4},
    "OR7":  {"service": "Upper Thoracic", "min_cases": 3, "max_cases": 3},
    "OR8":  {"service": "General (Robot)", "min_cases": 4, "max_cases": 4},
    "OR9":  {"service": "General (Robot)", "min_cases": 4, "max_cases": 4},
    "OR10": {"service": "General (Robot)", "min_cases": 4, "max_cases": 4},
    "OR11": {"service": "General (Robot)", "min_cases": 4, "max_cases": 4},
    "OR12": {"service": "General (Robot)", "min_cases": 4, "max_cases": 4},
    "OR14": {"service": "Ortho", "min_cases": 4, "max_cases": 5},
    "OR15": {"service": "Ortho", "min_cases": 4, "max_cases": 5},
    "OR16": {"service": "Ortho", "min_cases": 4, "max_cases": 5},
    "OR17": {"service": "General", "min_cases": 2, "max_cases": 3},
    "OR18": {"service": "Vascular", "min_cases": 2, "max_cases": 3},
    "OR19": {"service": "General (Robot)", "min_cases": 2, "max_cases": 3},
    "OR20": {"service": "General (Robot)", "min_cases": 2, "max_cases": 3},
    "OR21": {"service": "Neuro", "min_cases": 1, "max_cases": 3},
    "OR22": {"service": "Neuro", "min_cases": 1, "max_cases": 3},
    "OR23": {"service": "Neuro", "min_cases": 1, "max_cases": 3},
    "OR24": {"service": "Ortho", "min_cases": 4, "max_cases": 5},
    "OR25": {"service": "Ortho", "min_cases": 4, "max_cases": 5},
    "OR26": {"service": "IMRI", "min_cases": 1, "max_cases": 2},
    "CystoA": {"service": "Cysto/Uro", "min_cases": 5, "max_cases": 7},
    "CystoB": {"service": "Cysto/Uro", "min_cases": 5, "max_cases": 7},
    "FarscanMRI": {"service": "MRI Anesthesia", "min_cases": 1, "max_cases": 4},
}

# Case types per service (NO pediatrics, NO trauma)
service_case_types = {
    "Cardiac": [
        "CABG", "Valve Replacement", "Aortic Root Repair", "LVAD Placement"
    ],
    "Abdominal Transplant": [
        "Liver Transplant", "Kidney Transplant", "Pancreas Transplant"
    ],
    "ENT": [
        "Tonsillectomy", "Sinus Surgery", "Thyroidectomy", "Parotidectomy"
    ],
    "Upper Thoracic": [
        "Lobectomy", "Esophagectomy", "Mediastinal Mass Resection"
    ],
    "General (Robot)": [
        "Robotic Colectomy", "Robotic Hernia Repair", "Robotic Cholecystectomy"
    ],
    "General": [
        "Open Colectomy", "Appendectomy", "Open Hernia Repair", "Laparoscopic Cholecystectomy"
    ],
    "Ortho": [
        "Total Knee Replacement", "Total Hip Replacement", "ORIF Ankle", "Spinal Fusion"
    ],
    "Vascular": [
        "Carotid Endarterectomy", "EVAR", "Fem-Pop Bypass"
    ],
    "Neuro": [
        "Craniotomy", "Spine Decompression", "Tumor Resection"
    ],
    "IMRI": [
        "Intraoperative Brain MRI Case"
    ],
    "Cysto/Uro": [
        "TURBT", "TURP", "Cystoscopy with Stent", "Ureteroscopy"
    ],
    "MRI Anesthesia": [
        "MRI with GA - Adult"
    ]
}

# Duration distributions (mean minutes per case type)
case_duration_means = {
    # Cardiac
    "CABG": 240, "Valve Replacement": 210, "Aortic Root Repair": 270, "LVAD Placement": 300,
    # Abdominal Transplant
    "Liver Transplant": 360, "Kidney Transplant": 240, "Pancreas Transplant": 300,
    # ENT
    "Tonsillectomy": 60, "Sinus Surgery": 120, "Thyroidectomy": 150, "Parotidectomy": 180,
    # Upper Thoracic
    "Lobectomy": 180, "Esophagectomy": 300, "Mediastinal Mass Resection": 240,
    # General (Robot)
    "Robotic Colectomy": 180, "Robotic Hernia Repair": 120, "Robotic Cholecystectomy": 90,
    # General
    "Open Colectomy": 180, "Appendectomy": 75, "Open Hernia Repair": 90, "Laparoscopic Cholecystectomy": 65,
    # Ortho
    "Total Knee Replacement": 140, "Total Hip Replacement": 130, "ORIF Ankle": 90, "Spinal Fusion": 180,
    # Vascular
    "Carotid Endarterectomy": 120, "EVAR": 180, "Fem-Pop Bypass": 180,
    # Neuro
    "Craniotomy": 240, "Spine Decompression": 180, "Tumor Resection": 300,
    # IMRI
    "Intraoperative Brain MRI Case": 180,
    # Cysto/Uro
    "TURBT": 60, "TURP": 90, "Cystoscopy with Stent": 45, "Ureteroscopy": 75,
    # MRI Anesthesia
    "MRI with GA - Adult": 90
}

# Airway patterns per service
service_airway_patterns = {
    "Cardiac": ["DL - Direct Laryngoscopy", "Video Laryngoscopy", "GlideScope Size 3", "GlideScope Size 4"],
    "Abdominal Transplant": ["DL - Direct Laryngoscopy", "Video Laryngoscopy", "GlideScope Size 3"],
    "ENT": ["DL - Direct Laryngoscopy", "GlideScope Size 3", "GlideScope Size 4", "Fiberoptic"],
    "Upper Thoracic": ["DL - Direct Laryngoscopy", "Video Laryngoscopy", "GlideScope Size 4"],
    "General (Robot)": ["DL - Direct Laryngoscopy", "LMA", "Video Laryngoscopy"],
    "General": ["DL - Direct Laryngoscopy", "LMA"],
    "Ortho": ["DL - Direct Laryngoscopy", "LMA", "MAC (Monitored Anesthesia Care)"],
    "Vascular": ["DL - Direct Laryngoscopy", "Video Laryngoscopy"],
    "Neuro": ["DL - Direct Laryngoscopy", "Video Laryngoscopy", "Fiberoptic"],
    "IMRI": ["DL - Direct Laryngoscopy", "Video Laryngoscopy"],
    "Cysto/Uro": ["LMA", "MAC (Monitored Anesthesia Care)", "DL - Direct Laryngoscopy"],
    "MRI Anesthesia": ["LMA", "MAC (Monitored Anesthesia Care)"],
}

# Small cross-over probability
CROSSOVER_PROB = 0.05


def sample_case_type(service):
    if np.random.rand() > CROSSOVER_PROB:
        return np.random.choice(service_case_types[service])
    else:
        other_services = [s for s in service_case_types.keys() if s != service]
        other_service = np.random.choice(other_services)
        return np.random.choice(service_case_types[other_service])


def sample_duration(case_type):
    mean = case_duration_means.get(case_type, 120)
    dur = int(np.round(np.random.normal(mean, mean * 0.25)))
    return max(30, dur)


def sample_airway(service):
    return np.random.choice(service_airway_patterns.get(service, ["DL - Direct Laryngoscopy"]))


def generate_day_schedule(day_index):
    date = start_date + timedelta(days=day_index)
    cases = []

    for room in rooms:
        svc_info = room_services[room]
        service = svc_info["service"]
        n_cases = np.random.randint(svc_info["min_cases"], svc_info["max_cases"] + 1)

        base_hour = np.random.choice(range(7, 10))
        current_start = datetime.combine(date.date(), datetime.min.time()) + timedelta(hours=base_hour)

        for i in range(n_cases):
            case_type = sample_case_type(service)
            duration = sample_duration(case_type)

            turnover = np.random.randint(10, 30)
            if i > 0:
                current_start += timedelta(minutes=turnover)

            scheduled_start = datetime.combine(date.date(), datetime.min.time()) + timedelta(hours=base_hour + i)
            delay_minutes = int(np.round((current_start - scheduled_start).total_seconds() / 60))

            end_time = current_start + timedelta(minutes=duration)

            cases.append({
                "DayIndex": day_index + 1,
                "CaseDate": date.date(),
                "OR_Room": room,
                "Service": service,
                "CaseType": case_type,
                "StartDateTime": current_start,
                "EndDateTime": end_time,
                "DurationMinutes": duration,
                "DelayMinutes": delay_minutes,
                "AirwayManagement": sample_airway(service),
                "PatientID": np.random.randint(100000, 999999),
                "Age": int(np.clip(np.round(np.random.normal(55, 18)), 18, 95)),
                "AnesthesiologistID": np.random.choice(range(101, 111)),
                "Complications": np.random.choice(
                    ["", "Hypotension", "Bradycardia", "Nausea", "Allergic Reaction", "Airway Issue", "Post-op Pain"],
                    p=[0.70, 0.08, 0.06, 0.07, 0.03, 0.04, 0.02]
                )
            })

    return cases


# ===== GENERATE DATA =====

all_cases = []
for d in range(n_days):
    all_cases.extend(generate_day_schedule(d))

df = pd.DataFrame(all_cases)

print(df.head(20).to_string(index=False))
print(f"\nTotal rows: {len(df)}")
print(f"Avg cases per day: {len(df) / n_days:.1f}")
print(f"Avg duration: {df['DurationMinutes'].mean():.1f} min")
print(f"Avg delay: {df['DelayMinutes'].mean():.1f} min")

# Uncomment to save
# df.to_csv("synthetic_main_or_schedule.csv", index=False)
