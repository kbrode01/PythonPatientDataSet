Pythonimport pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set seed for reproducibility
np.random.seed(42)

n_cases = 300  # Feel free to increase to 500–1000

# Business days in 2025 for realistic case dates
start_date = datetime(2025, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(365) if (start_date + timedelta(days=i)).weekday() < 5]
case_dates = np.random.choice(dates, size=n_cases, replace=True)
case_dates.sort()

# Common procedures with approximate probabilities
procedures = [
    'Appendectomy', 'Cholecystectomy', 'Total Knee Replacement',
    'Coronary Artery Bypass Graft', 'Hip Replacement', 'Hysterectomy',
    'Laparoscopic Cholecystectomy', 'Spinal Fusion', 'Cataract Surgery',
    'Colon Resection'
]
procedure_probs = [0.12, 0.10, 0.09, 0.08, 0.08, 0.07, 0.15, 0.07, 0.12, 0.12]
procedures_sample = np.random.choice(procedures, size=n_cases, p=np.array(procedure_probs)/sum(procedure_probs))

# OR rooms
or_rooms = ['OR1', 'OR2', 'OR3', 'OR4', 'OR5']
or_room_sample = np.random.choice(or_rooms, size=n_cases)

# Anesthesiologist IDs (fake)
anes_ids = np.random.choice(range(101, 111), size=n_cases)

# Patient ages ~ normal distribution
ages = np.round(np.random.normal(55, 18, n_cases)).astype(int).clip(18, 95)

# Complications (70% none)
complication_options = ['', 'Hypotension', 'Bradycardia', 'Nausea', 'Allergic Reaction', 'Airway Issue', 'Post-op Pain']
complication_probs = [0.70, 0.08, 0.06, 0.07, 0.03, 0.04, 0.02]
complications = np.random.choice(complication_options, size=n_cases, p=complication_probs)

# Durations based on procedure (mean + variation)
base_durations = {
    'Appendectomy': 75,
    'Cholecystectomy': 90,
    'Total Knee Replacement': 140,
    'Coronary Artery Bypass Graft': 240,
    'Hip Replacement': 130,
    'Hysterectomy': 110,
    'Laparoscopic Cholecystectomy': 65,
    'Spinal Fusion': 180,
    'Cataract Surgery': 35,
    'Colon Resection': 150
}
durations = []
for proc in procedures_sample:
    mean_dur = base_durations[proc]
    dur = np.round(np.random.normal(mean_dur, mean_dur * 0.25)).astype(int).clip(30, mean_dur * 2)
    durations.append(dur)
durations = np.array(durations)

# Start times (mostly 7–17, on 15/30/45 min marks)
start_hours = np.random.choice(range(7, 18), size=n_cases, p=[0.05,0.1,0.15,0.15,0.12,0.1,0.08,0.08,0.07,0.05,0.03])
start_minutes = np.random.choice([0, 15, 30, 45], size=n_cases, p=[0.4, 0.2, 0.3, 0.1])
start_times = [
    datetime.combine(d.date(), datetime.min.time()) + timedelta(hours=int(h), minutes=int(m))
    for d, h, m in zip(case_dates, start_hours, start_minutes)
]

# End times
end_times = [s + timedelta(minutes=int(d)) for s, d in zip(start_times, durations)]

# Delay from scheduled (scheduled = round hour)
scheduled_starts = [
    datetime.combine(d.date(), datetime.min.time()) + timedelta(hours=int(h))
    for d, h in zip(case_dates, start_hours)
]
delays = np.round([(actual - sched).total_seconds() / 60 for actual, sched in zip(start_times, scheduled_starts)]).astype(int)

# Build DataFrame
df = pd.DataFrame({
    'CaseID': range(1, n_cases + 1),
    'PatientID': np.random.randint(100000, 999999, n_cases),
    'Age': ages,
    'ProcedureName': procedures_sample,
    'OR_Room': or_room_sample,
    'AnesthesiologistID': anes_ids,
    'StartDateTime': start_times,
    'EndDateTime': end_times,
    'DurationMinutes': durations,
    'DelayMinutes': delays,
    'Complications': complications
})

# Save to CSV (uncomment to use)
# df.to_csv('synthetic_anesthesia_cases.csv', index=False)
# print("Saved to synthetic_anesthesia_cases.csv")

# Preview
print(df.head(10).to_string(index=False))
print(f"\nTotal rows: {len(df)}")
print(f"Avg duration: {df['DurationMinutes'].mean():.1f} min")
print(f"Avg delay: {df['DelayMinutes'].mean():.1f} min")