import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest

# Step 1: Create synthetic dataset
np.random.seed(42)

n_control = 500
n_variant = 500

# Simulate conversions: control ~12%, variant ~15%
control_conversions = np.random.binomial(1, 0.12, n_control)
variant_conversions = np.random.binomial(1, 0.15, n_variant)

df_control = pd.DataFrame({
    'user_id': range(n_control),
    'group': 'control',
    'converted': control_conversions
})

df_variant = pd.DataFrame({
    'user_id': range(n_control, n_control + n_variant),
    'group': 'variant',
    'converted': variant_conversions
})

df = pd.concat([df_control, df_variant], ignore_index=True)

# Save to CSV
df.to_csv('ab_test_data.csv', index=False)
print("Synthetic dataset 'ab_test_data.csv' created.")

# Step 2: Load dataset
df = pd.read_csv('ab_test_data.csv')

# Step 3: Define groups
control = df[df['group'] == 'control']
variant = df[df['group'] == 'variant']

# Step 4: Calculate conversion rates
conv_control = control['converted'].sum()
n_control = control.shape[0]
rate_control = conv_control / n_control

conv_variant = variant['converted'].sum()
n_variant = variant.shape[0]
rate_variant = conv_variant / n_variant

print(f"Control Conversion Rate: {rate_control:.4f} ({conv_control}/{n_control})")
print(f"Variant Conversion Rate: {rate_variant:.4f} ({conv_variant}/{n_variant})")

# Step 5: Proportion z-test
count = np.array([conv_variant, conv_control])
nobs = np.array([n_variant, n_control])
stat, pval = proportions_ztest(count, nobs)

print(f"Z-test statistic: {stat:.4f}")
print(f"P-value: {pval:.4f}")

# Step 6: Confidence interval for difference in proportions
def conf_interval(p1, n1, p2, n2, alpha=0.05):
    diff = p1 - p2
    se = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    z = 1.96  # 95% CI
    lower = diff - z*se
    upper = diff + z*se
    return lower, upper

lower, upper = conf_interval(rate_variant, n_variant, rate_control, n_control)
print(f"95% Confidence Interval for difference: [{lower:.4f}, {upper:.4f}]")
