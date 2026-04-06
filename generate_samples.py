"""Generate sample_outputs for all 10 students."""
import json
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from main import load_data, STUDENTS_DATA, analyze_student, recommend_dosts, compute_student_score

load_data()

os.makedirs("sample_outputs", exist_ok=True)

all_analyses = []
all_recommendations = []

for student in STUDENTS_DATA:
    sid = student["student_id"]
    analysis = analyze_student(student)
    recommendation = recommend_dosts(student, analysis)

    all_analyses.append(analysis)
    all_recommendations.append(recommendation)

    # Individual files
    with open(f"sample_outputs/analyze_{sid}.json", "w") as f:
        json.dump(analysis, f, indent=2)
    with open(f"sample_outputs/recommend_{sid}.json", "w") as f:
        json.dump(recommendation, f, indent=2)

    print(f"Generated: {sid} ({student['name']})")

# Leaderboard
entries = []
for student in STUDENTS_DATA:
    analysis = analyze_student(student)
    score = compute_student_score(student, analysis)
    entries.append({
        "rank": 0,
        "student_id": student["student_id"],
        "name": student["name"],
        "score": score,
        "overall_pct": analysis["overall_score_pct"],
        "strength": analysis["strongest_chapters"][0] if analysis["strongest_chapters"] else "N/A",
        "weakness": analysis["weakest_chapters"][0] if analysis["weakest_chapters"] else "N/A",
        "focus_area": analysis["weakest_chapters"][0] if analysis["weakest_chapters"] else "N/A",
        "trend": analysis["trend"],
    })
entries.sort(key=lambda x: -x["score"])
for i, e in enumerate(entries):
    e["rank"] = i + 1

with open("sample_outputs/leaderboard.json", "w") as f:
    json.dump({"leaderboard": entries}, f, indent=2)

print("\nLeaderboard generated.")
print("All sample outputs saved to sample_outputs/")
