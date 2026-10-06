import numpy as np
import pandas as pd

np.random.seed(42)

n = 1000

df = pd.DataFrame({
    "StudentID": np.arange(1, n + 1),
    "CGPA": np.round(np.random.uniform(5.5, 9.8, n), 2),
    "Internships": np.random.randint(0, 4, n),
    "Projects": np.random.randint(0, 5, n),
    "AptitudeScore": np.random.randint(40, 101, n),
    "CommunicationScore": np.random.randint(40, 101, n),
    "TechnicalScore": np.random.randint(40, 101, n),
    "Attendance": np.round(np.random.uniform(60, 100, n), 1),
    "Gender": np.random.choice(["Male", "Female"], n),
    "Branch": np.random.choice(
        ["CSE", "ECE", "EEE", "MECH", "CIVIL"], n
    ),
    "Education": np.random.choice(
        ["B.Tech", "B.E."], n
    )
})

# Placement probability
score = (
    0.9 * df["CGPA"]
    + 0.15 * df["Internships"]
    + 0.10 * df["Projects"]
    + 0.015 * df["AptitudeScore"]
    + 0.015 * df["TechnicalScore"]
    + 0.01 * df["CommunicationScore"]
    + 0.01 * df["Attendance"]
    - 8
)

probability = 1 / (1 + np.exp(-score + 8))

df["PlacementStatus"] = (
    np.random.random(n) < probability
).astype(int)

# Salary only for placed students
df["Salary Package"] = np.where(
    df["PlacementStatus"] == 1,
    np.round(
        2.5
        + 0.7 * df["CGPA"]
        + 0.4 * df["Internships"]
        + 0.15 * df["Projects"]
        + np.random.normal(0, 1, n),
        2
    ),
    0
)

df["CGPA_Tier"] = pd.cut(
    df["CGPA"],
    bins=[0, 6.5, 7.5, 8.5, 10],
    labels=["Low", "Medium", "Good", "Excellent"]
).astype(str)

df["IsAnomaly"] = 0

output = "Small_placement_predict_1k_Dataset - Copy(1).csv"

df.to_csv(output, index=False)

print("Dataset created successfully!")
print("Shape:", df.shape)
print("Saved as:", output)
print(df.head())