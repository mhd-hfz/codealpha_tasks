#📊 A/B Testing Analysis
This project performs statistical analysis on A/B test data to determine whether a new change (e.g., UI feature, pricing, etc.) has a statistically significant impact on user conversion.

##📁 Files Included
ab_test_data.csv
→ A synthetic dataset containing user group assignments (control or variant) and binary conversion outcomes (0 or 1).

ab_test_analysis.py
→ Python script that:

Calculates conversion rates

Runs a two-proportion Z-test

Computes 95% confidence intervals

Interprets statistical significance

requirements.txt
→ List of required Python packages for the project.

##▶️ How to Run
Install dependencies:
```bash
pip install -r requirements.txt
```

Run the analysis script:
```bash
python ab_test_analysis.py
```

##📤 Expected Output
The script will generate an analysis report like this:

```bash
=== A/B Test Analysis Report ===
Control Group Conversion Rate: 13.8000% (69/500)
Variant Group Conversion Rate: 14.0000% (70/500)

Z-test statistic: 0.0914  
P-value: 0.9272  

95% Confidence Interval for difference in conversion rates (Variant - Control):  
[-4.0884%, 4.4884%]  

Result: No statistically significant difference (fail to reject null hypothesis).
```

##📈 Interpretation Guide
Conversion Rate
→ Proportion of users who converted in each group.

P-value
→ If p < 0.05, the difference is statistically significant (reject null hypothesis).
→ If p ≥ 0.05, the difference is not statistically significant (fail to reject null hypothesis).

95% Confidence Interval (CI)
→ Range where the true difference in conversion rates likely lies. If the CI includes 0, it suggests no significant difference.

Z-test
→ A statistical test used to compare two proportions (e.g., control vs variant conversion rates).

##📌 Notes
You can replace ab_test_data.csv with your own dataset using the same format:

```bash
group,converted
control,1
variant,0
...
```
The script supports only binary conversion outcomes (0 or 1).