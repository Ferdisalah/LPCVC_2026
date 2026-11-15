
EXT_SOURCE DISTRIBUTION AUDIT
=============================

1. EXT_SOURCE_SUMMARY_BY_GROUP.csv
   Quantiles, mean, standard deviation, missingness,
   low-score proportions, separately for ALL/TARGET_0/TARGET_1.

2. EXT_SOURCE_FIXED_BINS.csv
   Number and percentage of customers in each score interval.
   Includes the default rate for each interval.

3. EXT_SOURCE_SCORE_DECILES.csv
   Score deciles and default rate inside each decile.
   Deciles are calculated using all observed scores of each feature.

4. EXT_SOURCE_LOW_THRESHOLD_SHARES.csv
   Percentage of observed customers with score <= 0.1, 0.2, etc.

5. EXT_SOURCE_MISSINGNESS.csv
   Missing-value rates overall and by TARGET.
   Also compares default rates among missing and observed values.

6. EXT_SOURCE_UNIVARIATE_ASSOCIATION.csv
   AUC_RAW_SCORE_FOR_DEFAULT:
       < 0.50 means lower scores tend to be associated with default.
       > 0.50 means higher scores tend to be associated with default.

   This is a univariate diagnostic, not a multivariate model result.

7. 01_EXT_SOURCE_HISTOGRAMS.png
   Score distributions for TARGET=0 and TARGET=1.
   Each class is density-normalized; the curves compare shapes,
   not the number of customers.

8. 02_EXT_SOURCE_ECDF.png
   Cumulative distributions. A curve further to the left
   indicates generally smaller observed scores.

9. 03_DEFAULT_RATE_BY_SCORE_BINS.png
   Default rate within each score interval.

10. 04_EXT_SOURCE_MISSINGNESS.png
    Missingness rates by TARGET.

11. 05_LOW_SCORE_SHARES.png
    Proportion of customers below each score threshold.

Important:
- TARGET=0 means non-default.
- TARGET=1 means default.
- Missing scores are not treated as zero.
- The analysis does not establish causality.
- AUC is calculated using observed values only.
