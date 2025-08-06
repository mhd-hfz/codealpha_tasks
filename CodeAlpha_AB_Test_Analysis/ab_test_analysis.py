import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest

# Synthetic dataset creation (you can skip if dataset already exists)
np.random.seed(42)
n_control, n_variant = 500, 500
control_conversions = np.random.binomial(1, 0.12, n_control)
variant_conversions = np.random.binomial(1, 0.15, n_variant)
df_control = pd.DataFrame({'user_id': range(n_control), 'group': 'control', 'converted': control_conversions})
df_variant = pd.DataFrame({'user_id': range(n_control, n_control+n_variant), 'group': 'variant', 'converted': variant_conversions})
df = pd.concat([df_control, df_variant], ignore_index=True)
df.to_csv('ab_test_data.csv', index=False)

# Load dataset
df = pd.read_csv('ab_test_data.csv')

# Split groups
control = df[df['group'] == 'control']
variant = df[df['group'] == 'variant']

# Calculate conversion rates
conv_control = control['converted'].sum()
n_control = control.shape[0]
rate_control = conv_control / n_control

conv_variant = variant['converted'].sum()
n_variant = variant.shape[0]
rate_variant = conv_variant / n_variant

# Proportion z-test
count = np.array([conv_variant, conv_control])
nobs = np.array([n_variant, n_control])
stat, pval = proportions_ztest(count, nobs)

# Confidence interval function
def conf_interval(p1, n1, p2, n2, alpha=0.05):
    diff = p1 - p2
    se = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    z = 1.96
    lower = diff - z*se
    upper = diff + z*se
    return lower, upper

lower, upper = conf_interval(rate_variant, n_variant, rate_control, n_control)

# Display formatted results
print("\n=== A/B Test Analysis Report ===")
print(f"Control Group Conversion Rate: {rate_control:.4%} ({conv_control}/{n_control})")
print(f"Variant Group Conversion Rate: {rate_variant:.4%} ({conv_variant}/{n_variant})")
print(f"\nZ-test statistic: {stat:.4f}")
print(f"P-value: {pval:.4f}")

print("\n95% Confidence Interval for difference in conversion rates (Variant - Control):")
print(f"[{lower:.4%}, {upper:.4%}]")

if pval < 0.05:
    print("\nResult: The difference is statistically significant (reject null hypothesis).")
else:
    print("\nResult: No statistically significant difference (fail to reject null hypothesis).")

print("\nInterpretation:")
print(f"- The variant's conversion rate is {rate_variant:.4%}, the control's is {rate_control:.4%}.")
print(f"- The confidence interval includes zero, so the true difference may be zero.")
print(f"- The p-value indicates that the observed difference could easily be due to chance.")
print("")

