# Flight Delay Probability Analysis

A Python analysis of flights departing Chicago O'Hare International Airport (ORD) during January 2015. This project explores how well Poisson and Exponential distributions describe daily delay counts and delay durations.

Completed for ALY6105 Applied Statistics at Northeastern University.

## Project objectives

- Prepare flight records while preserving missing values.
- Calculate summary statistics for delayed flights.
- Model the number of delayed flights per day using a Poisson distribution.
- Model delay durations using an Exponential distribution.
- Compare theoretical probabilities with observed proportions.

## Dataset and preparation

The course dataset contains 23,484 flights across 31 days. It is a curated subset of the 2015 Flight Delays and Cancellations dataset, attributed to the U.S. Department of Transportation.

A flight is classified as delayed when its arrival delay is at least 15 minutes.

The analysis defines:

```python
duration_of_delay = arrival_delay - 15
```

Cancelled flights are excluded. Missing arrival delays are preserved rather than replaced with zero.

The 50 non-cancelled diverted flights with missing arrival delays remain in the total flight count but are excluded from delay-based calculations.

## Summary results

| Measure | Result |
|---|---:|
| Flights excluding cancellations | 22,366 |
| Days covered | 31 |
| Observed delayed flights | 7,303 |
| Mean duration beyond the 15-minute threshold | 52.1090 minutes |
| Sample standard deviation | 62.0688 minutes |
| Average delayed flights per day | 235.5806 |

## Findings

The Exponential model matched the selected outcomes more closely than the Poisson model.

| Model | Mean absolute difference |
|---|---:|
| Poisson | 0.3706 |
| Exponential | 0.0476 |

For days with 200–300 delayed flights, the Poisson model predicted a probability of 0.9919, compared with an observed proportion of 0.1290.

The Exponential model's theoretical 90th percentile was 120.0 minutes beyond the delay threshold, equivalent to an arrival delay of approximately 135.0 minutes.

These comparisons are descriptive rather than formal goodness-of-fit tests. The models describe different variables and events.

## Tools

- Python
- Pandas
- NumPy
- SciPy

## Run the analysis

1. Download or clone this repository.
2. Obtain the course dataset, ord_january_2015-2.csv. It is not currently included in this repository.
3. Update the CSV input path and output folder in 5.py to match your computer.
4. Install dependencies with: python3 -m pip install pandas numpy scipy
5. Run the analysis with: python3 5.py

## Outputs

The script prints and exports summary statistics, daily delayed-flight counts, probability comparison tables, and a descriptive model comparison. It also exports an updated copy of the dataset.

## Limitations

The analysis covers one airport's departures during one month. Weather, congestion and shared aircraft schedules may create correlated delays and changing daily rates. Recorded durations are whole-minute values, while the Exponential model is continuous. Both models were evaluated on the same data used to estimate their parameters.

## Author

Timer Senarith
