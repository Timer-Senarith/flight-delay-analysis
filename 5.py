import numpy as np
import pandas as pd
from scipy.stats import poisson, expon

# Load dataset
df = pd.read_csv(
    "/Users/timersenarith/Desktop/Northeastern University/ALY 6105 Applied Statistics/Assignment 5/ord_january_2015-2.csv"
)

# Check dataset
print(df.head())
print(df.columns.tolist())
print(df.shape)

import numpy as np
import pandas as pd
from scipy.stats import poisson, expon


# Folder for saving separate output files.
output_folder = (
    "/Users/timersenarith/Desktop/Northeastern University/"
    "ALY 6105 Applied Statistics/Assignment 5/"
)

# Inspect missing values. Do not replace them with zero.
print("\nMISSING VALUES")
print(df.isna().sum())

# Work with a copy of the original dataset.
updated_df = df.copy()

# Required variable: minutes beyond the 15 minute threshold.
# Missing arrival delays remain missing.
updated_df["duration_of_delay"] = updated_df["arrival_delay"] - 15

# Combine year, month, and day into a date.
updated_df["date"] = pd.to_datetime(
    updated_df[["year", "month", "day"]]
)

# Exclude cancelled flights from the analysis.
flights = updated_df[updated_df["cancelled"] == 0].copy()

# Count missing arrival delays among non-cancelled flights.
missing_arrivals = flights["arrival_delay"].isna().sum()

# Select delayed flights with a recorded arrival delay.
# Exactly 15 minutes late is included.
delayed_flights = flights[
    flights["arrival_delay"].notna()
    & (flights["duration_of_delay"] >= 0)
].copy()

print("\nDATA PREPARATION")
print("Original number of flights:", len(df))
print("Cancelled flights excluded:", (df["cancelled"] == 1).sum())
print("Non-cancelled flights missing arrival delay:", missing_arrivals)


# STEP 2: DATA SUMMARY

total_flights = len(flights)
number_of_days = flights["date"].nunique()
total_delayed = len(delayed_flights)

mean_duration = delayed_flights["duration_of_delay"].mean()

# ddof=1 calculates the sample standard deviation.
std_duration = delayed_flights["duration_of_delay"].std(ddof=1)

# Check that all expected January dates are represented.
expected_dates = pd.date_range("2015-01-01", "2015-01-31")
observed_dates = pd.DatetimeIndex(flights["date"].unique()).sort_values()

missing_dates = expected_dates.difference(observed_dates)
unexpected_dates = observed_dates.difference(expected_dates)

print("\nDATE COVERAGE")
print("First date:", flights["date"].min().date())
print("Last date:", flights["date"].max().date())
print("Missing January dates:", missing_dates.strftime("%Y-%m-%d").tolist())
print("Unexpected dates:", unexpected_dates.strftime("%Y-%m-%d").tolist())

# Stop if there is insufficient data for the requested models.
if number_of_days == 0 or total_delayed < 2:
    raise ValueError("Insufficient data to calculate the required statistics.")

if mean_duration <= 0:
    raise ValueError("The Exponential model requires a positive mean duration.")

summary = pd.DataFrame({
    "Statistic": [
        "Total flights excluding cancellations",
        "Number of days",
        "Total delayed flights",
        "Mean duration_of_delay (minutes)",
        "Sample standard deviation (minutes)"
    ],
    "Value": [
        total_flights,
        number_of_days,
        total_delayed,
        mean_duration,
        std_duration
    ]
})

print("\nSTEP 2: DATA SUMMARY")
print(f"Total flights excluding cancellations: {total_flights:,}")
print(f"Number of days: {number_of_days}")
print(f"Total delayed flights: {total_delayed:,}")
print(f"Mean duration_of_delay: {mean_duration:.4f} minutes")
print(f"Sample standard deviation: {std_duration:.4f} minutes")


# STEP 3: POISSON DISTRIBUTION

# 3.1 Calculate the average number of delayed flights per day.
lambda_value = total_delayed / number_of_days

print("\nSTEP 3.1: POISSON PARAMETER")
print(
    f"Lambda = {total_delayed} / {number_of_days}"
    f" = {lambda_value:.4f} delayed flights per day"
)

# 3.2 Calculate theoretical probabilities using Poisson functions.
poisson_theoretical = np.array([
    poisson.pmf(250, mu=lambda_value),

    # Include 200 through 300 by subtracting P(X <= 199).
    poisson.cdf(300, mu=lambda_value)
    - poisson.cdf(199, mu=lambda_value),

    poisson.cdf(250, mu=lambda_value),

    # sf(275) gives P(X > 275).
    poisson.sf(275, mu=lambda_value)
])

# 3.3 Count delayed flights for every represented date.
# Reindex includes dates with zero delayed flights.
daily_counts = (
    delayed_flights.groupby("date")
    .size()
    .reindex(observed_dates, fill_value=0)
)

daily_table = daily_counts.rename_axis("date").reset_index(
    name="delayed_flights"
)

print("\nSTEP 3.3: DAILY DELAYED FLIGHT COUNTS")
print(daily_table.to_string(index=False))

# Count the number of days satisfying each condition.
poisson_observed_counts = np.array([
    (daily_counts == 250).sum(),
    ((daily_counts >= 200) & (daily_counts <= 300)).sum(),
    (daily_counts <= 250).sum(),
    (daily_counts > 275).sum()
])

poisson_observed = poisson_observed_counts / number_of_days

poisson_comparison = pd.DataFrame({
    "Outcome": [
        "X = 250",
        "200 <= X <= 300",
        "X <= 250",
        "X > 275"
    ],
    "Theoretical Probability": poisson_theoretical,
    "Observed Days": poisson_observed_counts,
    "Observed Proportion": poisson_observed,
    "Absolute Difference": np.abs(
        poisson_theoretical - poisson_observed
    )
})

print("\nPOISSON COMPARISON")
print(
    poisson_comparison.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)


# STEP 4: EXPONENTIAL DISTRIBUTION

# Use delayed flights only for every calculation in Step 4.
durations = delayed_flights["duration_of_delay"]

# 4.1 Calculate rate and scale.
rate = 1 / mean_duration
scale = 1 / rate

print("\nSTEP 4.1: EXPONENTIAL PARAMETERS")
print(f"Rate = 1 / {mean_duration:.4f} = {rate:.8f} per minute")
print(f"Scale = 1 / rate = {scale:.4f} minutes")

# 4.2 Theoretical probabilities use distribution functions only.
# loc=0 fixes the distribution's starting point at zero.
exponential_theoretical = np.array([
    expon.cdf(30, loc=0, scale=scale),
    expon.sf(60, loc=0, scale=scale),
    expon.cdf(90, loc=0, scale=scale)
    - expon.cdf(30, loc=0, scale=scale)
])

# Inverse CDF gives the theoretical 90th percentile.
percentile_90 = expon.ppf(0.90, loc=0, scale=scale)

print("\nSTEP 4.2: THEORETICAL 90TH PERCENTILE")
print(f"90th percentile: {percentile_90:.1f} minutes")

# 4.3 Observed proportions use actual delayed-flight records.
# Preserve the strict inequalities specified in the assignment.
exponential_observed_counts = np.array([
    (durations < 30).sum(),
    (durations > 60).sum(),
    ((durations > 30) & (durations < 90)).sum()
])

exponential_observed = exponential_observed_counts / total_delayed

exponential_comparison = pd.DataFrame({
    "Outcome": [
        "X < 30",
        "X > 60",
        "30 < X < 90"
    ],
    "Theoretical Probability": exponential_theoretical,
    "Observed Flights": exponential_observed_counts,
    "Observed Proportion": exponential_observed,
    "Absolute Difference": np.abs(
        exponential_theoretical - exponential_observed
    )
})

print("\nEXPONENTIAL COMPARISON")
print(
    exponential_comparison.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)


# STEP 5: SYNTHESIS SUPPORT

# Compare average absolute differences across assigned outcomes.
poisson_mean_difference = poisson_comparison[
    "Absolute Difference"
].mean()

exponential_mean_difference = exponential_comparison[
    "Absolute Difference"
].mean()

model_comparison = pd.DataFrame({
    "Model": ["Poisson", "Exponential"],
    "Mean Absolute Difference": [
        poisson_mean_difference,
        exponential_mean_difference
    ]
})

print("\nSTEP 5: DESCRIPTIVE MODEL COMPARISON")
print(
    model_comparison.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)

print(
    "\nThese averages describe agreement for the assigned outcomes."
    "\nThey are not a formal goodness-of-fit comparison because"
    "\nthe models describe different variables and different events."
)



# EXPORT RESULTS TO SEPARATE CSV FILES

# Save all original rows with the two added columns.
# The source CSV is not overwritten.
updated_df.to_csv(
    output_folder + "ord_january_2015_updated.csv",
    index=False,
    na_rep=""
)

summary.to_csv(
    output_folder + "summary_statistics.csv",
    index=False
)

daily_table.to_csv(
    output_folder + "daily_delayed_flights.csv",
    index=False
)

poisson_comparison.to_csv(
    output_folder + "poisson_comparison.csv",
    index=False,
    float_format="%.4f"
)

exponential_comparison.to_csv(
    output_folder + "exponential_comparison.csv",
    index=False,
    float_format="%.4f"
)

model_comparison.to_csv(
    output_folder + "model_comparison.csv",
    index=False,
    float_format="%.4f"
)

print("\nResults saved in:", output_folder)