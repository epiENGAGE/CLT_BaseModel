import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import json
    import matplotlib.pyplot as plt
    from pathlib import Path

    return Path, json, mo, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Parameters
    """)
    return


@app.cell
def _(np):
    params = {
        # ── Simulation settings ────────────────────────────────────────────────
        "start_date": "2025-09-01",
        "num_days":   250,

        # ── Transmission ──────────────────────────────────────────────────────
        "beta_baseline":             0.039,
        "humidity_impact":           0.25,
        "relative_suscept":          1.0,
        "I_relative_infectiousness": 1.0,
        "IV_relative_infectiousness": 1.0,

        # ── Progression rates (per day) ────────────────────────────────────────
        "E_to_I_rate":   0.5,    # ~2-day latent period
        "EV_to_IV_rate": 0.5,
        "I_out_rate":    0.333,  # ~3-day infectious period
        "H_out_rate":    0.17,   # ~6-day hospital stay

        # ── Age-stratified proportions (7 groups: 0, 1-4, 5-12, 13-17, 18-49, 50-64, 65+) ──
        "I_to_H_prop": np.array([
            0.006971411624, 0.006971411624, 0.002741943394, 0.002741943394,
            0.005613008764, 0.01060470654,  0.09090913146,
        ]),
        "IV_to_H_prop": np.array([   # lower for vaccinated
            0.006357927401, 0.006357927401, 0.002500652376, 0.002500652376,
            0.005113450984, 0.009660887656, 0.06272730071,
        ]),
        "H_to_D_prop": np.array([
            0.01741109215, 0.01741109215, 0.0117181651, 0.0117181651,
            0.02633277317, 0.06301864177,  0.07988898095,
        ]),

        # Fraction of susceptibility retained after vaccination (1=no protection)
        "vax_susceptibility": np.array([0.57, 0.57, 0.57, 0.57, 0.79, 0.79, 1.0]),

        # Days from vaccination to immunity (shifts the vax schedule forward)
        "vax_transfer_delay_days": 14,

        # ── Initial exposed seeds ──────────────────────────────────────────────
        "E0_counts": np.array([2.0, 8.0, 17.0, 12.0, 85.0, 41.0, 35.0]),
    }
    return (params,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Input loading
    """)
    return


@app.cell
def _(Path, json, np, pd):
    def load_inputs(data_folder, params):
        """
        Load all CSVs, build per-day schedule arrays, and return a single dict
        that contains everything run_simulation() needs.
        """
        data_folder = Path(data_folder)
        num_days = params["num_days"]
        delay    = params["vax_transfer_delay_days"]

        # Population
        df_pop     = pd.read_csv(data_folder / "data" / "massachusetts_population.csv")
        population = df_pop["population"].to_numpy(dtype=float)

        # Contact matrices from JSON config
        config_path = data_folder / "model_config_MA.json"
        with open(config_path) as f:
            cfg = json.load(f)
        total_C  = np.array(cfg["params"]["total_contact_matrix"])
        school_C = np.array(cfg["params"]["school_contact_matrix"])
        work_C   = np.array(cfg["params"]["work_contact_matrix"])

        # Date range
        dates = pd.date_range(start=params["start_date"], periods=num_days, freq="D")

        # Humidity timeseries
        df_hum = pd.read_csv(data_folder / "data" / "schedules" / "ma_absolute_humidity.csv")
        df_hum["date"] = pd.to_datetime(df_hum["date"])
        df_hum = df_hum.set_index("date")

        # School/work calendar
        df_cal = pd.read_csv(data_folder / "data" / "schedules" / "MA_school_work_calendar.csv")
        df_cal["date"] = pd.to_datetime(df_cal["date"])
        df_cal = df_cal.set_index("date")

        # Vaccination schedule (proportions per age group)
        df_vax = pd.read_csv(data_folder / "data" / "vaccination" / "MA_flu_daily_vaccinations_proportions_array.csv")
        df_vax["date"] = pd.to_datetime(df_vax["date"])
        df_vax["daily_vaccines"] = (
            df_vax["daily_vaccines"].apply(json.loads).apply(lambda x: np.array(x).flatten())
        )
        df_vax = df_vax.set_index("date")

        # Observed hospitalizations (new daily admissions)
        df_obs = pd.read_csv(data_folder / "data" / "hospitalizations_ts" / "MA_flu_daily_hospitalizations_total.csv")
        df_obs["date"] = pd.to_datetime(df_obs["Date"])
        df_obs = df_obs.set_index("date")[["total"]]

        # Pre-compute per-day arrays
        A = len(population)
        humidity_arr  = np.zeros(num_days)
        is_school_arr = np.zeros(num_days)
        is_work_arr   = np.zeros(num_days)
        vax_arr       = np.zeros((num_days, A))

        for t, date in enumerate(dates):
            if date in df_hum.index:
                humidity_arr[t] = df_hum.loc[date, "absolute_humidity"]
            if date in df_cal.index:
                is_school_arr[t] = df_cal.loc[date, "is_school_day"]
                is_work_arr[t]   = df_cal.loc[date, "is_work_day"]
            raw_date = date - pd.Timedelta(days=delay)
            if raw_date in df_vax.index:
                vax_arr[t] = df_vax.loc[raw_date, "daily_vaccines"]

        return {
            "params":         params,
            "population":     population,
            "dates":          dates,
            "total_C":        total_C,
            "school_C":       school_C,
            "work_C":         work_C,
            "humidity_arr":   humidity_arr,
            "is_school_arr":  is_school_arr,
            "is_work_arr":    is_work_arr,
            "vax_arr":        vax_arr,
            "df_obs":         df_obs,
        }

    return (load_inputs,)


@app.cell
def _(load_inputs, params):
    inputs = load_inputs("generic_core/examples/massachusetts_vax", params)
    return (inputs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simulation
    """)
    return


@app.cell
def _(np, pd):
    def run_simulation(inputs):
        """
        Run the SEIR+vaccination model and return a DataFrame of daily outputs.

        Compartments (all shape: num_age_groups):
            S   - Susceptible (unvaccinated)
            E   - Exposed (unvaccinated)
            I   - Infectious (unvaccinated)
            R   - Recovered
            SV  - Susceptible (vaccinated, not yet infected)
            EV  - Exposed (vaccinated)
            IV  - Infectious (vaccinated)
            H   - Hospitalized
            D   - Dead (cumulative)
        """
        p          = inputs["params"]
        population = inputs["population"]
        dates      = inputs["dates"]
        num_days   = p["num_days"]

        # Unpack params
        beta            = p["beta_baseline"]
        hum_impact      = p["humidity_impact"]
        rel_suscept     = p["relative_suscept"]
        I_rel_inf       = p["I_relative_infectiousness"]
        IV_rel_inf      = p["IV_relative_infectiousness"]
        E_to_I          = p["E_to_I_rate"]
        EV_to_IV        = p["EV_to_IV_rate"]
        I_out           = p["I_out_rate"]
        H_out           = p["H_out_rate"]
        I_to_H          = p["I_to_H_prop"]
        IV_to_H         = p["IV_to_H_prop"]
        H_to_D          = p["H_to_D_prop"]
        vax_suscept     = p["vax_susceptibility"]

        # Contact matrices
        total_C  = inputs["total_C"]
        school_C = inputs["school_C"]
        work_C   = inputs["work_C"]

        # Schedule arrays
        humidity_arr  = inputs["humidity_arr"]
        is_school_arr = inputs["is_school_arr"]
        is_work_arr   = inputs["is_work_arr"]
        vax_arr       = inputs["vax_arr"]

        # Initial conditions
        E0 = p["E0_counts"]
        S  = population - E0;  E  = E0.copy()
        I  = np.zeros_like(population);  R  = np.zeros_like(population)
        SV = np.zeros_like(population);  EV = np.zeros_like(population)
        IV = np.zeros_like(population);  H  = np.zeros_like(population)
        D  = np.zeros_like(population)

        records = []

        for t in range(num_days):

            # ── Effective contact matrix ─────────────────────────────────────────
            # C = total - (1-school_open)*school_C - (1-work_open)*work_C
            C = (
                total_C
                - (1.0 - is_school_arr[t]) * school_C
                - (1.0 - is_work_arr[t])   * work_C
            )

            # ── Humidity modifier: higher humidity → lower beta ──────────────────
            beta_adj = beta * (1.0 + hum_impact * np.exp(-180.0 * humidity_arr[t]))

            # ── Force of infection ───────────────────────────────────────────────
            wtd_inf_prop = (I * I_rel_inf + IV * IV_rel_inf) / population
            foi = beta_adj * (C @ wtd_inf_prop)   # shape (A,)

            # ── Flows: unvaccinated pathway ──────────────────────────────────────
            S_to_E = foi * rel_suscept * S
            E_to_I_flow = E_to_I * E
            I_to_H_flow = I_out * I_to_H * I
            I_to_R_flow = I_out * (1.0 - I_to_H) * I

            # ── Vaccination: exact count S → SV (delay already in vax_arr) ───────
            S_to_SV = np.minimum(np.rint(vax_arr[t] * S), S)

            # ── Flows: vaccinated pathway ────────────────────────────────────────
            SV_to_EV  = foi * vax_suscept * SV
            EV_to_IV_flow = EV_to_IV * EV
            IV_to_H_flow  = I_out * IV_to_H * IV
            IV_to_R_flow  = I_out * (1.0 - IV_to_H) * IV

            # ── Flows: hospital ──────────────────────────────────────────────────
            H_to_D_flow = H_out * H_to_D * H
            H_to_R_flow = H_out * (1.0 - H_to_D) * H

            # ── Euler update ─────────────────────────────────────────────────────
            S  = S  - S_to_E   - S_to_SV
            E  = E  + S_to_E   - E_to_I_flow
            I  = I  + E_to_I_flow - I_to_H_flow - I_to_R_flow
            R  = R  + I_to_R_flow + IV_to_R_flow + H_to_R_flow
            SV = SV + S_to_SV  - SV_to_EV
            EV = EV + SV_to_EV - EV_to_IV_flow
            IV = IV + EV_to_IV_flow - IV_to_H_flow - IV_to_R_flow
            H  = H  + I_to_H_flow + IV_to_H_flow - H_to_D_flow - H_to_R_flow
            D  = D  + H_to_D_flow

            records.append({
                "date":           dates[t],
                "S":              S.sum(),   "E":  E.sum(),  "I":  I.sum(),
                "R":              R.sum(),   "SV": SV.sum(), "EV": EV.sum(),
                "IV":             IV.sum(),  "H":  H.sum(),  "D":  D.sum(),
                "new_H":          (I_to_H_flow + IV_to_H_flow).sum(),
                "new_D":          H_to_D_flow.sum(),
                "new_infections": S_to_E.sum(),
            })

        return pd.DataFrame(records).set_index("date")

    return (run_simulation,)


@app.cell
def _(inputs, run_simulation):
    df_results = run_simulation(inputs)
    return (df_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Results
    """)
    return


@app.cell
def _(df_results, inputs, plt):
    df_obs = inputs["df_obs"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # New infections
    axes[0].plot(df_results.index, df_results["new_infections"], label="New infections")
    axes[0].set_ylabel("Daily count")
    axes[0].set_title("New infections")
    axes[0].legend()

    # New hospitalizations vs observed
    axes[1].plot(df_results.index, df_results["new_H"], color="orange", label="Model: new_H")
    axes[1].scatter(df_obs.index, df_obs["total"], s=12, color="red", label="Observed", zorder=3)
    axes[1].set_ylabel("Daily count")
    axes[1].set_title("New hospitalizations")
    axes[1].legend()

    # Hospital census
    axes[2].plot(df_results.index, df_results["H"], color="steelblue", label="H (census)")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Hospital census")
    axes[2].legend()

    fig.autofmt_xdate()
    fig.tight_layout()
    plt.gca()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
