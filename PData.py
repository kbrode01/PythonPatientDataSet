import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Number of synthetic days to generate
N_DAYS = 10

# Synthetic data only. This does not represent real Mayo data.
START_DATE = datetime(2026, 10, 5)

# Approximate turnover time between cases
MIN_TURNOVER_MINUTES = 15
MAX_TURNOVER_MINUTES = 35

# Actual case starts may vary from the scheduled start.
# Negative values allow an occasional slightly early start.
MIN_START_VARIANCE_MINUTES = -5
MAX_START_VARIANCE_MINUTES = 30


# ============================================================
# OPERATING LOCATIONS
# ============================================================

ROOMS = [
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


# ============================================================
# ROOM / SERVICE CONFIGURATION
# ============================================================

ROOM_SERVICES = {
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


# ============================================================
# PROCEDURE TYPES
# ============================================================

SERVICE_CASE_TYPES = {
    "Cardiac": [
        "CABG",
        "Valve Replacement",
        "Aortic Root Repair",
        "LVAD Placement"
    ],
    "Abdominal Transplant": [
        "Liver Transplant",
        "Kidney Transplant",
        "Pancreas Transplant"
    ],
    "ENT": [
        "Tonsillectomy",
        "Sinus Surgery",
        "Thyroidectomy",
        "Parotidectomy"
    ],
    "Upper Thoracic": [
        "Lobectomy",
        "Esophagectomy",
        "Mediastinal Mass Resection"
    ],
    "General (Robot)": [
        "Robotic Colectomy",
        "Robotic Hernia Repair",
        "Robotic Cholecystectomy"
    ],
    "General": [
        "Open Colectomy",
        "Appendectomy",
        "Open Hernia Repair",
        "Laparoscopic Cholecystectomy"
    ],
    "Ortho": [
        "Total Knee Replacement",
        "Total Hip Replacement",
        "ORIF Ankle",
        "Spinal Fusion"
    ],
    "Vascular": [
        "Carotid Endarterectomy",
        "EVAR",
        "Fem-Pop Bypass"
    ],
    "Neuro": [
        "Craniotomy",
        "Spine Decompression",
        "Tumor Resection"
    ],
    "IMRI": [
        "Intraoperative Brain MRI Case"
    ],
    "Cysto/Uro": [
        "TURBT",
        "TURP",
        "Cystoscopy with Stent",
        "Ureteroscopy"
    ],
    "MRI Anesthesia": [
        "MRI with GA - Adult"
    ],
}


# ============================================================
# PROCEDURE DURATION MEANS
# ============================================================

CASE_DURATION_MEANS = {
    # Cardiac
    "CABG": 240,
    "Valve Replacement": 210,
    "Aortic Root Repair": 270,
    "LVAD Placement": 300,

    # Abdominal Transplant
    "Liver Transplant": 360,
    "Kidney Transplant": 240,
    "Pancreas Transplant": 300,

    # ENT
    "Tonsillectomy": 60,
    "Sinus Surgery": 120,
    "Thyroidectomy": 150,
    "Parotidectomy": 180,

    # Upper Thoracic
    "Lobectomy": 180,
    "Esophagectomy": 300,
    "Mediastinal Mass Resection": 240,

    # General (Robot)
    "Robotic Colectomy": 180,
    "Robotic Hernia Repair": 120,
    "Robotic Cholecystectomy": 90,

    # General
    "Open Colectomy": 180,
    "Appendectomy": 75,
    "Open Hernia Repair": 90,
    "Laparoscopic Cholecystectomy": 65,

    # Ortho
    "Total Knee Replacement": 140,
    "Total Hip Replacement": 130,
    "ORIF Ankle": 90,
    "Spinal Fusion": 180,

    # Vascular
    "Carotid Endarterectomy": 120,
    "EVAR": 180,
    "Fem-Pop Bypass": 180,

    # Neuro
    "Craniotomy": 240,
    "Spine Decompression": 180,
    "Tumor Resection": 300,

    # IMRI
    "Intraoperative Brain MRI Case": 180,

    # Cysto/Uro
    "TURBT": 60,
    "TURP": 90,
    "Cystoscopy with Stent": 45,
    "Ureteroscopy": 75,

    # MRI Anesthesia
    "MRI with GA - Adult": 90,
}


# ============================================================
# ANESTHESIA TYPES
# ============================================================

SERVICE_ANESTHESIA_TYPES = {
    "Cardiac": ["General"],
    "Abdominal Transplant": ["General"],
    "ENT": ["General"],
    "Upper Thoracic": ["General"],
    "General (Robot)": ["General"],
    "General": ["General", "MAC"],
    "Ortho": ["General", "MAC"],
    "Vascular": ["General", "MAC"],
    "Neuro": ["General"],
    "IMRI": ["General"],
    "Cysto/Uro": ["General", "MAC"],
    "MRI Anesthesia": ["General", "MAC"],
}


# ============================================================
# SAMPLING FUNCTIONS
# ============================================================

def sample_case_type(service):
    """
    Select a procedure appropriate for the room's assigned service.

    Cross-service randomization is intentionally excluded.
    Add-on and overflow behavior should be modeled explicitly later.
    """
    return np.random.choice(SERVICE_CASE_TYPES[service])


def sample_scheduled_duration(case_type):
    """
    Generate the scheduled duration around the typical duration for
    the procedure.
    """
    mean = CASE_DURATION_MEANS.get(case_type, 120)

    duration = int(
        np.round(
            np.random.normal(
                loc=mean,
                scale=mean * 0.15
            )
        )
    )

    return max(30, duration)


def sample_actual_duration(scheduled_duration):
    """
    Generate an actual duration that differs somewhat from the
    scheduled duration.
    """
    duration = int(
        np.round(
            np.random.normal(
                loc=scheduled_duration,
                scale=max(10, scheduled_duration * 0.15)
            )
        )
    )

    return max(30, duration)


def sample_anesthesia_type(service):
    return np.random.choice(
        SERVICE_ANESTHESIA_TYPES.get(service, ["General"])
    )


def sample_start_variance():
    """
    Generate the difference between scheduled and actual start time.
    Positive values represent delays.
    """
    return int(
        np.random.randint(
            MIN_START_VARIANCE_MINUTES,
            MAX_START_VARIANCE_MINUTES + 1
        )
    )


def sample_turnover():
    return int(
        np.random.randint(
            MIN_TURNOVER_MINUTES,
            MAX_TURNOVER_MINUTES + 1
        )
    )


# ============================================================
# DAILY SCHEDULE GENERATION
# ============================================================

def generate_day_schedule(day_index):
    date = START_DATE + timedelta(days=day_index)
    cases = []

    daily_case_sequence = 1

    for room in ROOMS:
        svc_info = ROOM_SERVICES[room]

        service = svc_info["service"]

        n_cases = np.random.randint(
            svc_info["min_cases"],
            svc_info["max_cases"] + 1
        )

        # First scheduled case starts between 07:00 and 08:00.
        first_case_minutes = np.random.randint(0, 61)

        scheduled_start = (
            datetime.combine(date.date(), datetime.min.time())
            + timedelta(hours=7, minutes=int(first_case_minutes))
        )

        previous_actual_end = None

        for case_index in range(n_cases):
            procedure_type = sample_case_type(service)

            scheduled_duration = sample_scheduled_duration(
                procedure_type
            )

            # Cases after the first are scheduled after the preceding
            # scheduled case plus a synthetic turnover period.
            if case_index > 0:
                scheduled_start = (
                    previous_scheduled_end
                    + timedelta(minutes=sample_turnover())
                )

            start_variance = sample_start_variance()

            proposed_actual_start = (
                scheduled_start
                + timedelta(minutes=start_variance)
            )

            # Prevent physically impossible overlap in the same room.
            if previous_actual_end is not None:
                minimum_actual_start = (
                    previous_actual_end
                    + timedelta(minutes=sample_turnover())
                )

                actual_start = max(
                    proposed_actual_start,
                    minimum_actual_start
                )
            else:
                actual_start = proposed_actual_start

            actual_duration = sample_actual_duration(
                scheduled_duration
            )

            scheduled_end = (
                scheduled_start
                + timedelta(minutes=scheduled_duration)
            )

            actual_end = (
                actual_start
                + timedelta(minutes=actual_duration)
            )

            case_number = (
                f"PA-{date.strftime('%Y%m%d')}-"
                f"{daily_case_sequence:04d}"
            )

            cases.append({
                "CaseNumber": case_number,
                "Location": room,
                "Service": service,
                "ProcedureType": procedure_type,
                "ProcedureCode": "",
                "ProcedureCodeSystem": "",
                "ScheduledStart": scheduled_start,
                "ScheduledDurationMinutes": scheduled_duration,
                "ActualStart": actual_start,
                "ActualEnd": actual_end,
                "ActualDurationMinutes": actual_duration,
                "AnesthesiaType": sample_anesthesia_type(service),
                "Status": "Completed",
                "Notes": "",
                "IsSynthetic": True,
            })

            previous_scheduled_end = scheduled_end
            previous_actual_end = actual_end
            daily_case_sequence += 1

    return cases


# ============================================================
# DATASET GENERATION
# ============================================================

def generate_dataset():
    all_cases = []

    for day_index in range(N_DAYS):
        all_cases.extend(
            generate_day_schedule(day_index)
        )

    return pd.DataFrame(all_cases)


# ============================================================
# VALIDATION
# ============================================================

def validate_dataset(df):
    print("\n===== VALIDATION =====")

    print(f"Total cases: {len(df)}")

    daily_counts = (
        df.groupby(df["ScheduledStart"].dt.date)
        .size()
    )

    print(
        f"Average cases/day: "
        f"{daily_counts.mean():.1f}"
    )

    print(
        f"Minimum cases/day: "
        f"{daily_counts.min()}"
    )

    print(
        f"Maximum cases/day: "
        f"{daily_counts.max()}"
    )

    print(
        f"Average scheduled duration: "
        f"{df['ScheduledDurationMinutes'].mean():.1f} min"
    )

    print(
        f"Average actual duration: "
        f"{df['ActualDurationMinutes'].mean():.1f} min"
    )

    delay_minutes = (
        df["ActualStart"]
        - df["ScheduledStart"]
    ).dt.total_seconds() / 60

    print(
        f"Average start variance: "
        f"{delay_minutes.mean():.1f} min"
    )

    duplicate_case_numbers = (
        df["CaseNumber"].duplicated().sum()
    )

    print(
        f"Duplicate CaseNumbers: "
        f"{duplicate_case_numbers}"
    )

    # Verify that actual cases do not overlap within a room.
    overlap_count = 0

    sorted_df = df.sort_values(
        ["Location", "ActualStart"]
    )

    for _, room_df in sorted_df.groupby("Location"):
        previous_end = None

        for _, row in room_df.iterrows():
            if (
                previous_end is not None
                and row["ActualStart"] < previous_end
            ):
                overlap_count += 1

            previous_end = row["ActualEnd"]

    print(
        f"Same-room actual overlaps: "
        f"{overlap_count}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    df = generate_dataset()

    print("\n===== SAMPLE =====")
    print(
        df.head(20).to_string(index=False)
    )

    validate_dataset(df)

    output_file = "synthetic_surgical_cases.csv"

    df.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nSynthetic dataset written to: "
        f"{output_file}"
    )
