> DE cap 50 column: the single best parameter set of fitted_params_MA_vax_de50.json, simulated deterministically (no interval); its fit-parameter rows span a posterior that was still drifting. Chain not converged.

# Run comparison

## Fit

| parameter | plain sum cap 5 | DE cap 50 best |
|---|---|---|
| posterior draws | 4000 | 4000 |
| beta_baseline | 0.0339 (0.03–0.0387) | 0.0331 (0.028–0.0385) |
| humidity_impact | 0.418 (0.0702–0.748) | 0.439 (0.153–0.721) |
| seed_scale_E | 1.96 (1.45–2.54) | 0.594 (0.204–1.39) |
| IHR multiplier bounds | 0.1–5 | 0.1–50 |
| IHR multiplier 0 | 4.80 (4.22–5.00) | 21.87 (7.50–45.59) |
|   near upper bound 0 | 72% | 1% |
| IHR multiplier 1-4 | 2.69 (2.32–2.98) | 11.98 (4.08–25.08) |
|   near upper bound 1-4 | 0% | 0% |
| IHR multiplier 5-12 | 2.41 (2.10–2.68) | 10.59 (3.63–22.18) |
|   near upper bound 5-12 | 0% | 0% |
| IHR multiplier 13-17 | 1.23 (1.07–1.36) | 5.39 (1.86–11.22) |
|   near upper bound 13-17 | 0% | 0% |
| IHR multiplier 18-49 | 0.69 (0.60–0.75) | 3.04 (1.04–6.35) |
|   near upper bound 18-49 | 0% | 0% |
| IHR multiplier 50-64 | 1.11 (0.97–1.22) | 4.95 (1.68–10.38) |
|   near upper bound 50-64 | 0% | 0% |
| IHR multiplier 65+ | 1.10 (0.97–1.20) | 4.96 (1.68–10.40) |
|   near upper bound 65+ | 0% | 0% |
| fitted IHR 0 | 3.345 (2.939–3.482) % | 15.248 (5.226–31.783) % |
| fitted IHR 1-4 | 1.872 (1.621–2.075) % | 8.351 (2.843–17.481) % |
| fitted IHR 5-12 | 0.660 (0.575–0.734) % | 2.904 (0.996–6.082) % |
| fitted IHR 13-17 | 0.337 (0.293–0.373) % | 1.478 (0.510–3.077) % |
| fitted IHR 18-49 | 0.385 (0.336–0.420) % | 1.705 (0.583–3.565) % |
| fitted IHR 50-64 | 1.180 (1.031–1.291) % | 5.250 (1.782–11.005) % |
| fitted IHR 65+ | 10.024 (8.774–10.917) % | 45.084 (15.286–94.503) % |

## Baseline attack rate, median [95% interval]

Vaccinated = vaccinated at any point in the season; their infections are those after vaccination.

### Total

| age group | plain sum cap 5 | DE cap 50 best |
|---|---|---|
| 0 | 5.0% [4.6–5.7%] | 0.5% [0.5–0.5%] |
| 1-4 | 5.7% [5.3–6.6%] | 0.6% [0.6–0.6%] |
| 5-12 | 8.7% [8.0–10.0%] | 0.9% [0.9–0.9%] |
| 13-17 | 9.1% [8.5–10.5%] | 1.0% [1.0–1.0%] |
| 18-49 | 8.4% [7.8–9.6%] | 0.9% [0.9–0.9%] |
| 50-64 | 7.1% [6.6–8.2%] | 0.7% [0.7–0.7%] |
| 65+ | 4.5% [4.2–5.2%] | 0.5% [0.5–0.5%] |
| All | 7.3% [6.8–8.4%] | 0.8% [0.8–0.8%] |

### Vaccinated

| age group | plain sum cap 5 | DE cap 50 best |
|---|---|---|
| 0 | 2.1% [2.0–2.5%] | 0.2% [0.2–0.2%] |
| 1-4 | 3.2% [3.0–3.7%] | 0.3% [0.3–0.3%] |
| 5-12 | 4.6% [4.3–5.3%] | 0.5% [0.5–0.5%] |
| 13-17 | 4.7% [4.4–5.5%] | 0.5% [0.5–0.5%] |
| 18-49 | 5.8% [5.4–6.7%] | 0.6% [0.6–0.6%] |
| 50-64 | 5.3% [4.9–6.1%] | 0.5% [0.5–0.5%] |
| 65+ | 4.1% [3.8–4.7%] | 0.4% [0.4–0.4%] |
| All | 4.9% [4.5–5.6%] | 0.5% [0.5–0.5%] |

### Unvaccinated

| age group | plain sum cap 5 | DE cap 50 best |
|---|---|---|
| 0 | 7.7% [7.2–8.9%] | 0.8% [0.8–0.8%] |
| 1-4 | 8.0% [7.5–9.2%] | 0.8% [0.8–0.8%] |
| 5-12 | 11.2% [10.4–12.9%] | 1.2% [1.2–1.2%] |
| 13-17 | 11.7% [10.9–13.4%] | 1.2% [1.2–1.2%] |
| 18-49 | 9.3% [8.6–10.7%] | 1.0% [1.0–1.0%] |
| 50-64 | 8.3% [7.7–9.5%] | 0.9% [0.9–0.9%] |
| 65+ | 5.2% [4.8–6.0%] | 0.5% [0.5–0.5%] |
| All | 8.9% [8.2–10.2%] | 0.9% [0.9–0.9%] |

### Share vaccinated

| age group | plain sum cap 5 | DE cap 50 best |
|---|---|---|
| 0 | 49.8% | 50.7% |
| 1-4 | 47.3% | 47.8% |
| 5-12 | 38.9% | 39.6% |
| 13-17 | 37.0% | 37.7% |
| 18-49 | 27.4% | 27.7% |
| 50-64 | 38.0% | 38.4% |
| 65+ | 61.2% | 61.5% |
| All | 38.5% | 38.9% |

## Total hospitalizations, median (95% interval)

| scenario | plain sum cap 5 | DE cap 50 best |
|---|---|---|
| no vax | 21,869 (20,989–22,374) | 34,306 (34,306–34,306) |
| baseline | 8,360 (8,270–8,461) | 8,373 (8,373–8,373) |
| 50% coverage (all ages) | 5,513 (5,447–5,599) | 5,215 (5,215–5,215) |
| 60% coverage (all ages) | 4,139 (4,086–4,218) | 3,857 (3,857–3,857) |
| 70% coverage (all ages) | 3,055 (3,015–3,117) | 2,832 (2,832–2,832) |

## Hospitalizations averted, total population, median (95% interval)

### Baseline vs no vax

| run | reference hospitalizations | scenario hospitalizations | averted | % of reference | per 100K population | per 100K doses |
|---|---|---|---|---|---|---|
| plain sum cap 5 | 21,869 (20,989–22,374) | 8,360 (8,270–8,461) | 13,510 (12,639–13,994) | 61.8% (60.1–62.5%) | 185.8 (173.8–192.4) | 480.9 (449.9–498.1) |
| DE cap 50 best | 34,306 (34,306–34,306) | 8,373 (8,373–8,373) | 25,933 (25,933–25,933) | 75.6% (75.6–75.6%) | 356.6 (356.6–356.6) | 923.1 (923.1–923.1) |

### 50% coverage (all ages) vs baseline

| run | reference hospitalizations | scenario hospitalizations | averted | % of reference | per 100K population | per 100K doses |
|---|---|---|---|---|---|---|
| plain sum cap 5 | 8,360 (8,270–8,461) | 5,513 (5,447–5,599) | 2,850 (2,789–2,894) | 34.1% (33.4–34.4%) | 39.2 (38.3–39.8) | 290.3 (284.1–294.9) |
| DE cap 50 best | 8,373 (8,373–8,373) | 5,215 (5,215–5,215) | 3,158 (3,158–3,158) | 37.7% (37.7–37.7%) | 43.4 (43.4–43.4) | 321.7 (321.7–321.7) |

### 60% coverage (all ages) vs baseline

| run | reference hospitalizations | scenario hospitalizations | averted | % of reference | per 100K population | per 100K doses |
|---|---|---|---|---|---|---|
| plain sum cap 5 | 8,360 (8,270–8,461) | 4,139 (4,086–4,218) | 4,222 (4,154–4,280) | 50.5% (49.8–50.8%) | 58.0 (57.1–58.9) | 269.5 (265.2–273.3) |
| DE cap 50 best | 8,373 (8,373–8,373) | 3,857 (3,857–3,857) | 4,516 (4,516–4,516) | 53.9% (53.9–53.9%) | 62.1 (62.1–62.1) | 288.3 (288.3–288.3) |

### 70% coverage (all ages) vs baseline

| run | reference hospitalizations | scenario hospitalizations | averted | % of reference | per 100K population | per 100K doses |
|---|---|---|---|---|---|---|
| plain sum cap 5 | 8,360 (8,270–8,461) | 3,055 (3,015–3,117) | 5,307 (5,237–5,373) | 63.5% (62.9–63.8%) | 73.0 (72.0–73.9) | 232.6 (229.5–235.5) |
| DE cap 50 best | 8,373 (8,373–8,373) | 2,832 (2,832–2,832) | 5,541 (5,541–5,541) | 66.2% (66.2–66.2%) | 76.2 (76.2–76.2) | 242.8 (242.8–242.8) |
