# VIVA QUESTIONS & ANSWERS – Smart Hostel System

> 25 likely exam/viva questions with clear, beginner-friendly answers.

---

## Part A: Supervised Learning / Prediction

**Q1. What is the difference between classification and regression? Which one does this project use?**

**A:** Classification predicts a *category* (e.g., pass/fail, spam/not-spam). Regression predicts a *continuous number* (e.g., 142 applications, 7.3 satisfaction rating). This project uses **regression** for both sub-problems: predicting the count of applications (demand model) and predicting satisfaction scores (preference model).

---

**Q2. Why did you use a time-based train/test split instead of a random split for the demand model?**

**A:** The demand data is a *time series* — future values depend on past values (via lag features). If we split randomly, the training data could contain observations from 2023 while the test data contains 2015 observations. The model would then be tested on data from its "past," which is too easy (it already "knows" those patterns). A time-based split (train=2010–2023, test=2024–2025) simulates the real scenario of predicting the future from the past.

---

**Q3. What are lag features? Why are they the most important features?**

**A:** A lag feature is the value of the target variable from a previous time step. For example, `lag1_applications` = how many students applied to this hostel last semester.  
They are often the most important features because demand is *autocorrelated* — if a hostel was popular last semester, it tends to be popular this semester too. Students base their decisions partly on what they heard from seniors.

---

**Q4. What is Ridge Regression and why is it used?**

**A:** Ridge Regression is linear regression with an L2 penalty on the coefficient magnitudes:  
`loss = RSS + α × sum(β²)`  
This prevents any single feature from getting an extremely large weight (overfitting). The `α` parameter controls the strength of regularisation. When features are correlated (e.g., lag1 and lag2 are correlated), Ridge handles this better than ordinary least squares.

---

**Q5. How does a Random Forest work?**

**A:** A Random Forest builds many decision trees, each trained on a different random subset of the training data and features (this is called *bagging*). Each tree makes its own prediction, and the forest averages all predictions. This *ensemble* approach reduces overfitting compared to a single tree. It also handles non-linear relationships well without any feature engineering for interactions.

---

**Q6. What is cross-validation? Why do we use TimeSeriesSplit?**

**A:** Cross-validation estimates how well a model will perform on unseen data by repeatedly training on one subset and testing on another. `TimeSeriesSplit` is a version that respects time order: in each fold, the test set is always *after* the training set. This prevents information leakage from future to past.

---

**Q7. How did you select the best demand model? Did you use the test set for selection?**

**A:** We selected the best model by **cross-validation MAE on the training set only** (years 2010–2023). The test set (2024–2025) was used only for final evaluation — never for model selection. This prevents a subtle form of overfitting called "test set contamination."

---

**Q8. What is MAPE and why is it useful?**

**A:** MAPE (Mean Absolute Percentage Error) = mean of |predicted − actual| / |actual| × 100%.  
It expresses error as a *percentage*, making it interpretable regardless of scale. A MAPE of 8% means on average we're off by 8% of the true value. (Limitation: MAPE is undefined when actual = 0, which is why we filter out zero values.)

---

**Q9. What is permutation importance? How does it differ from model coefficients?**

**A:** Permutation importance measures how much the model's accuracy drops when we *randomly shuffle* one feature (breaking its relationship with the target). Features that cause big accuracy drops are most important. Unlike model coefficients:
- It works for any model (even black-box ones like Random Forest)
- It measures actual impact on predictions, not just correlation with target
- It accounts for feature interactions

---

**Q10. Explain the student-room satisfaction model. What does it predict?**

**A:** It predicts a satisfaction rating (1-10) for a given (student profile, hostel) combination. It learns from historical survey data where students rated their actual rooms. Key features include budget gap (can they afford it?), AC match (did they get what they prefer?), distance (is it close?), mess rating (food quality), and whether this was their first-choice hostel.

---

**Q11. What is the difference between the satisfaction model's TRAINING and its USE in optimization?**

**A:** During *training*, the model learns from (student, hostel, actual_room) → satisfaction_rating records. During *use in optimization*, it predicts a score for every possible (student, hostel) pair, producing a matrix. The optimizer then uses these predicted scores as benefit values to decide who goes where. The model is trained once; the optimization uses it repeatedly.

---

## Part B: Optimization

**Q12. What is the assignment problem?**

**A:** The assignment problem is: given N workers and N tasks, with a cost for each (worker, task) pair, find the assignment that minimizes total cost (or maximizes total benefit). In our case: workers = students, tasks = beds, benefit = priority × predicted satisfaction. It's a classical combinatorial optimization problem.

---

**Q13. Explain the Hungarian algorithm.**

**A:** The Hungarian algorithm solves the assignment problem in polynomial time O(n³). The key idea:
1. Subtract the minimum of each row from that row.
2. Subtract the minimum of each column from that column.
3. Find the minimum number of lines to cover all zeros.
4. If the number of lines equals n, optimal assignment found (choose zeros).
5. Otherwise, adjust values and repeat.

In practice, we use `scipy.optimize.linear_sum_assignment` which implements this efficiently.

---

**Q14. Why did you use negative satisfaction in the cost matrix?**

**A:** `linear_sum_assignment` *minimises* cost. We want to *maximise* satisfaction. By using `cost = -satisfaction`, minimising cost becomes equivalent to maximising satisfaction. This is a standard trick in optimisation.

---

**Q15. What is the difference between a hard constraint and a soft goal?**

**A:** A *hard constraint* can never be violated — it defines feasibility. Example: a male student CANNOT be placed in a girls' hostel. A *soft goal* is something we try to optimise but can trade off. Example: we prefer students to get their first-choice hostel, but we don't require it if it would leave someone without a room.

---

**Q16. How are hard constraints enforced in the Hungarian allocation?**

**A:** We set the cost to `+infinity` (1e9 in our code) for any infeasible (student, bed) pair. Since the algorithm minimises cost, it will never choose an infinite-cost assignment unless absolutely necessary. If a student has no feasible bed, the algorithm still assigns them a bed — but this cell is detected post-hoc and the student is waitlisted.

---

**Q17. How does the roommate local search work? Is it guaranteed to find the global optimum?**

**A:** The local search tries every pair of students in different rooms and checks if swapping them reduces the number of "mismatched rooms" (rooms containing both early birds and night owls). If yes, the swap is accepted. This repeats until no improving swap exists. It is **not** guaranteed to find the global optimum (it finds a *local* optimum). A global approach would require searching all possible permutations, which is NP-hard.

---

**Q18. What is the greedy FCFS baseline and why is it suboptimal?**

**A:** Greedy First-Come-First-Served processes students in priority order. Each student gets their highest-satisfaction feasible hostel with available capacity. It's suboptimal because: if student A wants hostel H1 and gets it, but student B (slightly lower priority) really really needs H1 and would get terrible satisfaction elsewhere, the greedy algorithm ignores this global trade-off. The Hungarian algorithm finds solutions that may sacrifice A slightly to greatly help B.

---

## Evaluation

**Q19. Why do you evaluate using "true satisfaction" instead of the model's own predictions?**

**A:** Using the model's predictions for evaluation would be circular: the model assigned students to maximise predicted satisfaction, so of course predicted satisfaction looks high! The true hidden function gives an honest measure of how good the allocation actually is. This is the same principle as having a separate test set — you evaluate with data the model hasn't optimised for.

---

**Q20. What is the Gini coefficient? What does it measure in this context?**

**A:** The Gini coefficient measures inequality of a distribution. Gini=0 means everyone has exactly the same satisfaction score (perfect equality). Gini=1 means one person has all the satisfaction and everyone else has none. In our context, a lower Gini means the allocation is fairer — satisfaction is more evenly distributed, not concentrated among a few lucky students.

---

**Q21. What does the "gap to Oracle" tell us?**

**A:** The Oracle uses the TRUE satisfaction function, while Hungarian uses the model's PREDICTIONS. The gap between their scores (typically Oracle − Hungarian ≈ 0.3–0.5 points) represents the **cost of imperfect prediction**. A more accurate preference model would close this gap. This shows the direct value of improving Part A (prediction) for Part B (allocation).

---

**Q22. What does a constraint violation of 0 mean?**

**A:** It means every hard constraint is satisfied for every allocated student — no male student in a girls' hostel, no student paying more than budget + tolerance, no disabled student above the ground floor. This is the most critical metric: a system with violations cannot be deployed.

---

## General / Advanced

**Q23. What is the difference between overfitting and underfitting?**

**A:** *Overfitting* is when a model learns the training data too well, including noise, and generalises poorly to new data (high CV error, low test error relative to training). *Underfitting* is when the model is too simple to capture the pattern (high error on both training and test). Our Ridge regression might underfit; our Gradient Boosting might overfit if not regularised. We use CV to detect both issues.

---

**Q24. What is the relationship between the synthetic data and the real world?**

**A:** We generated synthetic data using a known formula (e.g., demand = beds × occupancy_rate × cohort_scale × fee_penalty + noise). This lets us:
- Control all parameters
- Know the ground truth for evaluation
- Demonstrate the system without accessing private student data

The limitation is that real data has patterns we didn't model (peer influence, specific professors, campus events). A real deployment would require retraining on real institutional data.

---

**Q25. If you had to improve this system for a real university, what would you do first?**

**A:** Three improvements in priority order:
1. **Real data collection:** Partner with the registrar's office to collect actual application and satisfaction data. Even 2-3 years of real data would dramatically improve the preference model.
2. **Dynamic reallocation:** Currently, once allocation is done, it's fixed. A real system should handle cancellations and new applicants throughout the semester.
3. **Multi-objective optimization:** Instead of a single weighted-sum objective, use Pareto optimisation to balance satisfaction vs. fairness explicitly, and let the hostel office choose from the Pareto frontier.
