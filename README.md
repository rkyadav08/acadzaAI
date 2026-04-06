# Acadza AI — Student Performance Analyzer & DOST Recommender

## Overview

A FastAPI-based recommender system for Acadza's EdTech platform. It ingests student performance data across JEE/NEET test sessions, analyzes patterns and weaknesses, and generates personalized step-by-step study plans using Acadza's DOST (Dynamic Optimized Study Task) framework — complete with specific question recommendations pulled from the question bank.

---

## Setup & Run

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

The server starts at `http://127.0.0.1:8000`. Interactive docs at `/docs`.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/analyze/{student_id}` | Full performance analysis across all sessions |
| POST | `/recommend/{student_id}` | Step-by-step DOST plan with question IDs |
| GET | `/question/{question_id}` | Question lookup with plaintext preview |
| GET | `/leaderboard` | All 10 students ranked by composite score |

---

## Approach to the Build Task

### How I Analyzed Student Data

The analysis pipeline processes each student's attempts chronologically, extracting several dimensions of performance. For each attempt, I parse the marks field (more on that below), compute percentage scores relative to estimated max marks, and track chapter-wise and subject-wise breakdowns. Key metrics I compute include overall score percentage, attempt rate (how many questions the student actually tries vs skips), completion rate (did they finish or abort mid-test), average time per question, and a trend indicator.

The trend indicator compares the first half of a student's attempts against the second half. If the second half average is more than 5 percentage points higher, the student is "improving"; more than 5 lower, "declining"; otherwise "stable". This is a simple heuristic but it captures the essential trajectory without overfitting to noise.

Chapter-wise breakdown aggregates all attempts touching a chapter (since attempts can cover multiple chapters) and computes average scores and timing. The weakest 3 chapters and strongest 2 are identified by sorting on average score percentage. I also track the slowest questions from each session, which helps identify specific pain points.

### How I Decided Which DOSTs to Recommend and in What Order

The recommendation logic follows a pedagogical progression that mirrors how a good tutor would structure a study session:

1. **Concept first** — for the weakest chapter, start with theory. You can't solve problems if the fundamentals are shaky. This maps to the `concept` DOST.

2. **Formula revision** — for the second-weakest chapter, a formula sheet helps bridge the gap between understanding concepts and applying them. Maps to `formula` DOST.

3. **Targeted practice** — a `practiceAssignment` on the weakest chapter, with difficulty scaled to the student's current level (easy if below 30%, medium if below 60%, hard otherwise). Questions are selected using the fixed cosine-similarity recommender, which matches the student's weakness profile against the question bank.

4. **Speed drills** — if the student's average time per question exceeds 150 seconds (2.5 minutes, which is slow for JEE), a `clickingPower` drill on a stronger topic builds speed without the frustration of hard content.

5. **Picking power** — if the attempt rate is below 80% (meaning the student skips too many questions), option elimination practice helps them attempt more questions with confidence.

6. **Multi-day revision** — if there are multiple weak chapters, a structured 3-day `revision` plan ensures systematic coverage rather than ad-hoc cramming.

7. **Mock test** — if completion rate is low or the trend is declining, a shorter-than-usual `practiceTest` builds exam stamina. The key insight: students who abort tests need confidence more than difficulty.

8. **Speed race** — for students above 40% overall, a competitive `speedRace` against a bot adds motivation and tests performance under pressure.

Not every student gets all 8 steps. The conditions (low completion rate, slow timing, low attempt rate, etc.) gate which steps are included, so each plan is tailored.

### Question Selection via the Recommender

For selecting specific questions, I use the fixed cosine-similarity recommender as a module. The student's chapter-wise performance is converted into a weakness score vector (inverted: low performance = high weakness). This vector is matched against the question bank's topic-difficulty features using cosine similarity, and the top-N questions are returned. The recommender naturally prioritizes questions from the student's weak topics at appropriate difficulty levels.

---

## Handling the Messy Marks Field

The marks field comes in at least five formats across the dataset:

- **Plain integer**: `49`, `72` — used directly as the raw score.
- **Plus-minus format**: `"+52 -8"` — I subtract: 52 - 8 = 44. This represents correct marks minus negative marking.
- **Fraction format**: `"39/100"` — I extract the numerator (39) as raw score and the denominator (100) as max marks.
- **Fraction with percentage**: `"49/120 (40.8%)"` — I extract the numerator and denominator, ignoring the parenthetical percentage (it's redundant).
- **Plain string number**: `"22"` — parsed as float.

For computing percentages, I need max marks. When the fraction format provides a denominator, I use that. Otherwise, I estimate max marks as `total_questions × 4` (standard JEE Mains scoring of +4 per correct). This is an assumption — actual JEE Advanced has different scoring for different question types — but it's a reasonable default given the data doesn't specify the marking scheme explicitly.

One edge case: the plus-minus format doesn't tell us max marks, so the estimation fallback kicks in. This means percentage comparisons across different mark formats aren't perfectly calibrated, but they're consistent enough for relative ranking within a student's own sessions.

---

## Debug Process — The Buggy Recommender

### What the Bug Was

The bug was in the `recommend()` function, specifically lines 47-48. The code does this:

```python
cohort_baseline = student_matrix.mean(axis=0)
student_profile = student_matrix[student_idx] - cohort_baseline  # Line 45: CORRECT

profile_norm = np.linalg.norm(cohort_baseline)                   # Line 47: BUG - uses wrong vector
student_profile = cohort_baseline / (profile_norm + 1e-10)        # Line 48: BUG - overwrites with wrong value
```

Line 45 correctly computes a student-specific profile by subtracting the cohort average from the student's vector. But then line 47 computes the norm of `cohort_baseline` (not `student_profile`), and line 48 **overwrites** `student_profile` with the normalized `cohort_baseline`. The per-student adjustment from line 45 is completely discarded.

The result: every student ends up with the exact same profile vector (the normalized cohort mean), so all three students get identical recommendations with 10/10 overlap.

### How I Found It

The first thing I did was run the buggy code and look at the output. The 10/10 overlap between all pairs was the immediate red flag — three students with completely different weakness profiles (Physics-heavy, Chemistry-heavy, Math-heavy) should never get identical recommendations.

I then read the `recommend()` function line by line. The variable naming made the bug sneaky: `student_profile` is computed correctly, then two lines later it's overwritten. The variable name `profile_norm` sounds like it should be the norm of `student_profile`, but it's actually `np.linalg.norm(cohort_baseline)`. And the assignment `student_profile = cohort_baseline / ...` looks like a normalization step, but it's replacing the student-specific vector with the cohort average.

This is the kind of bug that's designed to fool AI tools — the code reads naturally, uses sensible variable names, and doesn't throw any errors. The logic "compute baseline, subtract it, normalize the result" is correct in intent; the implementation just normalizes the wrong vector.

### The Fix

```python
profile_norm = np.linalg.norm(student_profile)        # Use student_profile's own norm
student_profile = student_profile / (profile_norm + 1e-10)  # Normalize the actual profile
```

After fixing, Arjun gets mechanics questions, Priya gets organic chemistry, Rahul gets algebra — exactly matching their weakness profiles, with 0/10 overlap between pairs.

---

## Leaderboard Scoring Formula

The composite score weights multiple dimensions:

- **40%** overall score percentage — raw academic performance
- **20%** completion rate — finishing tests matters for exam readiness  
- **15%** attempt rate — not skipping questions shows confidence
- **15%** trend bonus — improving students get 15 points, stable get 8, declining get 0
- **10%** consistency — low variance across sessions (calculated as inverse of standard deviation, scaled)

This formula rewards both performance and good habits. A student who scores moderately but completes every test and is improving will rank above a high-scorer who aborts tests and is declining.

---

## What I'd Improve Given More Time

- **Per-question accuracy tracking**: The current data gives aggregate marks per session. If we had per-question correct/incorrect data, we could build much more precise topic-level weakness profiles and identify specific misconception patterns.

- **Spaced repetition integration**: The revision DOST could incorporate spaced repetition scheduling — if a student got thermodynamics wrong 3 days ago, it should resurface today, not next week.

- **Adaptive difficulty within DOSTs**: Instead of setting difficulty once per step, the system could adjust mid-session based on how the student is performing on the first few questions.

- **Time-of-day and fatigue modeling**: Students perform differently at different times. With timestamp data, we could recommend optimal study windows.

- **Better marks normalization**: With access to the actual marking scheme per exam pattern (Mains vs Advanced, different weightings per question type), the percentage calculations would be more accurate.

---

## Project Structure

```
acadza/
├── main.py                  # FastAPI app with all endpoints
├── recommender.py           # Fixed cosine-similarity recommender module
├── student_performance.json # 10 students, 5-8 sessions each
├── question_bank.json       # 200 questions
├── dost_config.json         # DOST type configurations
├── requirements.txt
├── generate_samples.py      # Script to produce sample_outputs/
├── sample_outputs/          # Pre-generated outputs for all 10 students
│   ├── analyze_STU_001.json ... analyze_STU_010.json
│   ├── recommend_STU_001.json ... recommend_STU_010.json
│   └── leaderboard.json
├── debug/
│   ├── recommender_buggy.py   # Original buggy file
│   └── recommender_fixed.py   # Corrected file with explanation
└── README.md
```
