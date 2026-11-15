# Project Explanation – Smart Hostel System

> Written for beginners. No jargon without explanation.

---

## What Problem Are We Solving?

A university has several hostels (student dormitories). Every semester, hundreds of students
apply for rooms. The hostel office faces two challenges:

1. **Planning challenge:** How many students will apply to each hostel? (So they can plan food,
   maintenance, warden staffing, etc.)

2. **Fairness challenge:** Among all applicants, who gets which room? How do we make sure
   disabled students, scholarship holders, and first-years are prioritised? How do we avoid
   putting a night-owl and an early-bird in the same room?

This project builds a computer system to answer both questions automatically.

---

## Part A: Prediction (Supervised Learning)

### What is Supervised Learning?
Supervised learning means teaching a computer from examples.  
We show it many past records: "In 2019, hostel H1 received 140 applications."  
The computer learns the pattern and can predict future values.

### A1: Demand Forecasting

**The question:** How many students will apply to hostel X in semester Y of year Z?

**How we solve it:**
1. We have 15 years of historical data (2010-2025): how many students applied to each hostel
   each semester, plus relevant context (cohort size, fees, etc.).
2. We engineer "lag features" — e.g., "how many applied last semester?" This is the single
   strongest predictor because demand is sticky.
3. We train 4 regression models and pick the best using cross-validation on the training data
   only (never peeking at the test years 2024-2025).
4. The best model predicts next year's demand per hostel and flags each as
   OVERSUBSCRIBED / BALANCED / UNDER-USED.

**Why time-based splitting matters:**
If we split randomly (e.g., train on 2023 data, test on 2015 data), the model would look
at data "from the future" during training, which is cheating. We always train on older data
and test on newer data.

### A2: Satisfaction Prediction

**The question:** How satisfied will student S be if placed in room R of hostel H?

**How we solve it:**
1. We have historical survey data: students rated their room 1-10 after staying.
2. We extract features like "budget gap" (can they afford it?), "AC match" (did they get what
   they wanted?), "distance" (is it close?), "first-choice match" (did they get their top pick?).
3. We train regression models and pick the best by cross-validation.
4. The model scores every possible (student, hostel) pair — producing a matrix of
   ~1400 students × 6 hostels = ~8400 predicted scores.

**Why does this matter?** These predicted scores feed directly into the optimizer in Part B.

---

## Part B: Optimization (Room Allocation)

### What is Optimization?
Optimization means finding the best solution among many possible ones, subject to rules.

### The Assignment Problem

We need to assign 1400 students to ~950 beds. We want to maximise total satisfaction.
But we have rules:

**Hard constraints (non-negotiable):**
- Male students → boys' hostels only
- Female students → girls' hostels only  
- Budget: student can't be placed in a hostel costing more than their budget + ₹5000
- Disabled students → ground floor only

**Soft goals (do our best):**
- High satisfaction (especially for priority students)
- First-years, disabled, scholarship holders, far-away students get extra priority
- Avoid putting night-owls and early-risers in the same room

### The Hungarian Algorithm

The Hungarian algorithm (invented in 1955!) solves the "assignment problem" optimally.

**Intuition:**
- Think of it as matching workers to jobs at minimum cost.
- We have workers (students) and jobs (beds).
- The cost of assigning student i to bed j = -(priority_weight_i × predicted_satisfaction_i,j)
  (negative because we want to maximise, but the algorithm minimises)
- Hard constraint violations get cost = infinity (so they're never chosen)
- The algorithm finds the globally optimal assignment.

**Why is this better than Greedy?**
Greedy just gives each student (in priority order) their best available option.
But this can be suboptimal: giving student A their 1st choice might force students B and C
to get bad options. The Hungarian algorithm considers the entire picture simultaneously.

### Roommate Local Search

After Hungarian assignment, we run a "swap" pass:
- Find rooms with mismatched sleepers (early + night in same room)
- Try swapping two students between rooms
- Accept if it reduces total mismatches
- Repeat until no improvement is possible

This is a "local search" — it doesn't guarantee the global optimum for roommates, but
it typically finds a good solution quickly.

---

## Evaluation

### Why use "True Satisfaction" for evaluation?

After allocating students, we evaluate using the **hidden ground-truth satisfaction function**,
not the model's predictions. This is critical for honest evaluation.

If we evaluated using the model's own predictions:
- A model that predicted everyone = 8 would seem great
- But real satisfaction might be 5!

By using the hidden oracle function, we see what satisfaction actually would be.

### Metrics Explained

| Metric | What it means | Good value |
|---|---|---|
| Allocation rate | % of applicants who got a room | Close to 100% |
| Mean satisfaction | Average true satisfaction (1-10) | ≥ 7 |
| Gini coefficient | Inequality of satisfaction scores | Close to 0 (equal) |
| Priority coverage | % of priority students housed | ≥ 95% |
| First-choice match | % who got their top hostel pick | Higher = better |
| Constraint violations | Number of rule violations | Must be 0 |
| Happy % | % with satisfaction ≥ 7 | Higher = better |

---

## The Key Insight (for Viva)

> **Prediction feeds Optimization, and Evaluation judges the combination.**

- Without prediction, we'd allocate randomly (no satisfaction information)
- With perfect prediction (oracle), allocation would be near-ideal
- The gap between Oracle and Hungarian shows the "cost" of imperfect prediction
- Better survey data → better preference model → better allocation → more satisfied students

This is a complete ML pipeline:  Data → Features → Model → Predictions → Optimizer → Evaluation
