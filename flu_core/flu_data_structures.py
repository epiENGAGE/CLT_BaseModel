import torch
import numpy as np
import pandas as pd
import datetime
import warnings

from typing import Optional
from dataclasses import dataclass, fields, field

import clt_toolkit as clt


# A "MM_DD" parameter describing a date near the simulation start that
#   resolves further than this many days away is almost certainly a
#   mis-specification rather than a deliberate choice -- see
#   `resolve_mm_dd_near_date`
INFECTION_IMMUNITY_START_DATE_WARN_DAYS = 120


def resolve_mm_dd_near_date(mm_dd: str,
                            reference_date: datetime.date,
                            param_name: str = "date",
                            warn_days: Optional[int] = INFECTION_IMMUNITY_START_DATE_WARN_DAYS
                            ) -> datetime.date:
    """
    Resolves a year-less "MM_DD" parameter to the actual calendar date
    closest to `reference_date`.

    A "MM_DD" string is ambiguous about which year it means, and simply
    reusing `reference_date`'s year silently picks the wrong one for any
    season that spans a new year. For a simulation starting 2024-08-15,
    "01_01" under that rule resolves to 2024-01-01 -- 227 days *before*
    the start -- when the intended date is almost certainly 2025-01-01,
    139 days after it. Choosing whichever of the previous, current, and
    next year lands closest to `reference_date` resolves both cases the
    way a user would expect, and is unambiguous as long as the intended
    date is within ~6 months of the reference date.

    Args:
        mm_dd (str):
            date in "MM_DD" format, e.g. "01_01".
        reference_date (datetime.date):
            the date to resolve relative to -- generally the
            simulation's `start_real_date`.
        param_name (str):
            name of the parameter being resolved, used in messages.
        warn_days (Optional[int]):
            warn if the resolved date is more than this many days from
            `reference_date` -- pass None to skip the check.

    Returns:
        datetime.date
    """

    month, day = (int(x) for x in mm_dd.split('_'))

    candidates = []
    for year in (reference_date.year - 1, reference_date.year, reference_date.year + 1):
        try:
            candidates.append(datetime.date(year, month, day))
        except ValueError:
            # Feb 29 in a non-leap year -- that year simply has no such
            #   date, so it is not a candidate
            continue

    if not candidates:
        raise ValueError(
            f"`{param_name}` = '{mm_dd}' does not correspond to a real date in any "
            f"year adjacent to {reference_date}.")

    resolved = min(candidates, key=lambda d: abs((d - reference_date).days))

    if warn_days is not None and abs((resolved - reference_date).days) > warn_days:
        warnings.warn(
            f"`{param_name}` = '{mm_dd}' resolved to {resolved}, which is "
            f"{abs((resolved - reference_date).days)} days from {reference_date}. "
            f"This parameter is meant to describe a date near the simulation start "
            f"(within a couple of months); a gap this large usually means the "
            f"month/day is mis-specified.")

    return resolved


@dataclass
class FluSubpopState(clt.SubpopState):
    """
    Data container for pre-specified and fixed set of
    Compartment initial values and EpiMetric initial values
    for `FluSubpopModel`.

    Each field below should be A x R np.ndarray, where
    A is the number of age groups and R is the number of risk groups.
    Note: this means all arrays should be 2D. Even if there is
    1 age group and 1 risk group (no group stratification),
    each array should be 1x1, which is two-dimensional.
    For example, np.array([[100]]) is correct --
    np.array([100]) is wrong.

    Attributes:
        S (np.ndarray of nonnegative integers):
            susceptible compartment for age-risk groups --
            (holds current_val of Compartment "S").
        E (np.ndarray of nonnegative integers):
            exposed compartment for age-risk groups --
            (holds current_val of Compartment "E").
        IP (np.ndarray of nonnegative integers):
            infected pre-symptomatic compartment for age-risk groups
            (holds current_val of Compartment "IP").
        ISR (np.ndarray of nonnegative integers):
            infected symptomatic (that will recover) compartment
            for age-risk groups
            (holds current_val of Compartment "ISR").
        ISH (np.ndarray of nonnegative integers):
            infected symptomatic compartment (that will be hospitalized)
            for age-risk groups
            (holds current_val of Compartment "ISH").
        IA (np.ndarray of nonnegative integers):
            infected asymptomatic compartment for age-risk groups
            (holds current_val of Compartment "IA").
        HR (np.ndarray of nonnegative integers):
            hospital compartment (that will recover)
            for age-risk groups
            (holds current_val of Compartment "HR").
        HD (np.ndarray of nonnegative integers):
            hospital compartment (that will die)
            for age-risk groups
            (holds current_val of Compartment "HD").
        R (np.ndarray of nonnegative integers):
            recovered compartment for age-risk groups
            (holds current_val of Compartment "R").
        D (np.ndarray of nonnegative integers):
            dead compartment for age-risk groups
            (holds current_val of Compartment "D").
        M (np.ndarray of nonnegative floats):
            infection-induced population-level immunity
            for age-risk groups (holds current_val
            of EpiMetric "M").
        MV (np.ndarray of nonnegative floats):
            vaccine-induced population-level immunity
            for age-risk groups (holds current_val
            of EpiMetric "MV").
        absolute_humidity (positive float):
            grams of water vapor per cubic meter g/m^3,
            used as seasonality parameter that influences
            transmission rate beta_baseline.
        flu_contact_matrix (np.ndarray of positive floats):
            A x A array, where A is the number of age
            groups -- element (a, a') corresponds to the number
            of contacts that a person in age group a
            has with people in age-risk group a'.
        beta_reduce (float in [0,1]):
            starting value of DynamicVal "beta_reduce" on
            starting day of simulation -- this DynamicVal
            emulates a simple staged-alert policy
        daily_vaccines (np.ndarray of positive ints):
            holds current value of DailyVaccines instance,
            corresponding proportion of individuals in each
            age and risk group who received influenza
            vaccine on that day (generally derived from
            historical data)
        mobility_modifier (np.ndarray of positive floats):
            holds current value of MobilityModifier schedule,
            representing the proportion of time spent away
            from home by age group (A x R array)
    """

    S: Optional[np.ndarray] = None
    E: Optional[np.ndarray] = None
    IP: Optional[np.ndarray] = None
    ISR: Optional[np.ndarray] = None
    ISH: Optional[np.ndarray] = None
    IA: Optional[np.ndarray] = None
    HR: Optional[np.ndarray] = None
    HD: Optional[np.ndarray] = None
    R: Optional[np.ndarray] = None
    D: Optional[np.ndarray] = None

    M: Optional[np.ndarray] = None
    MV: Optional[np.ndarray] = None

    absolute_humidity: Optional[float] = None
    flu_contact_matrix: Optional[np.ndarray] = None
    beta_reduce: Optional[float] = 0.0

    daily_vaccines: Optional[np.ndarray] = None
    mobility_modifier: Optional[np.ndarray] = None


@dataclass(frozen=True)
class FluSubpopParams(clt.SubpopParams):
    """
    Data container for pre-specified and fixed epidemiological
    parameters in `FluSubpopModel`.

    Each field of datatype np.ndarray must be A x R,
    where A is the number of age groups and R is the number of
    risk groups. Note: this means all arrays should be 2D.
    See FluSubpopState docstring for important formatting note
    on 2D arrays.

    Note: the user does not have to specify `total_pop_age_risk` --
    this is automatically computed when a `FluSubpopModel` is
    instantiated. This is to ensure that the total population
    (summed across all compartments) actually equals `total_pop_age_risk` --
    and the user doesn't change one without updating the other.

    Attributes:
        num_age_groups (positive int):
            number of age groups.
        num_risk_groups (positive int):
            number of risk groups.
        start_real_date (datetime.date):
            real-world date that corresponds to start sof
            simulation.
        beta_baseline (positive float): transmission rate.
        total_pop_age_risk (np.ndarray of positive ints):
            total number in population, summed across all
            age-risk groups.
        humidity_impact (positive float):
            coefficient that determines how much absolute
            humidity affects beta_baseline.
        inf_induced_saturation (np.ndarray of positive floats):
            constant(s) modeling saturation of antibody
            production of infected individuals.
        inf_induced_immune_wane (positive float):
            rate at which infection-induced immunity
            against infection wanes.
        vax_induced_immune_wane (positive float):
            rate at which vaccine-induced immunity
            against infection wanes.
        inf_induced_inf_risk_reduce (positive float):
            reduction in risk of getting infected
            after getting infected
        inf_induced_hosp_risk_reduce (positive float):
            reduction in risk of hospitalization
            after getting infected
        inf_induced_death_risk_reduce (positive float):
            reduction in risk of death
            after getting infected
        vax_induced_inf_risk_reduce (positive float):
            SEASON-AVERAGE reduction in risk of getting infected
            after getting vaccinated -- i.e. the average protection
            a vaccinated individual has over the whole vaccination
            season, with waning already folded in. This is NOT the
            value applied in the model: the model applies
            `vax_induced_inf_risk_reduce_initial`, the peak
            (zero-waning) value back-derived from this one, and lets
            protection wane from there via the `MV` epi metric. See
            `adjust_VE_for_seasonal_waning` to turn that derivation
            off and use this value directly as the peak instead.
        vax_induced_hosp_risk_reduce (positive float):
            season-average reduction in risk of hospitalization
            after getting vaccinated -- see
            `vax_induced_inf_risk_reduce` for the season-average
            versus peak distinction.
        vax_induced_death_risk_reduce (positive float):
            season-average reduction in risk of death after getting
            vaccinated -- see `vax_induced_inf_risk_reduce` for the
            season-average versus peak distinction.
        vax_induced_inf_risk_reduce_initial (np.ndarray of positive floats):
            "peak" (zero-waning) vaccine-induced reduction in risk
            of getting infected -- this is the value actually applied
            in the model, as `1 - MV * vax_induced_inf_risk_reduce_initial`.
            Derived from vax_induced_inf_risk_reduce,
            vax_induced_immune_wane, and the daily_vaccines schedule
            so that, averaged over the vaccination season and
            accounting for waning, it reproduces
            vax_induced_inf_risk_reduce. Capped at 1.0. Recomputed on
            `reset_simulation`, so it stays consistent with any
            schedule or parameter overrides. See
            `flu_components.compute_vax_induced_risk_reduce_initial`.
        vax_induced_hosp_risk_reduce_initial (np.ndarray of positive floats):
            analogous "peak" (zero-waning) value for
            vax_induced_hosp_risk_reduce.
        vax_induced_death_risk_reduce_initial (np.ndarray of positive floats):
            analogous "peak" (zero-waning) value for
            vax_induced_death_risk_reduce.
        adjust_VE_for_seasonal_waning (bool):
            if True (default), `vax_induced_inf_risk_reduce_initial`,
            `vax_induced_hosp_risk_reduce_initial`, and
            `vax_induced_death_risk_reduce_initial` are computed from
            the corresponding `vax_induced_*_risk_reduce` season-average
            value, `vax_induced_immune_wane`, and the `daily_vaccines`
            schedule -- see
            `compute_vax_induced_risk_reduce_initial`. If False, this
            adjustment is skipped and the `_initial` values are simply
            set equal to the corresponding `vax_induced_*_risk_reduce`
            values (i.e. waning is not accounted for when determining
            the peak vaccine efficacy applied to newly-vaccinated
            individuals), which means the `vax_induced_*_risk_reduce`
            inputs are interpreted as peak rather than season-average
            efficacies.
        VE_season_dose_window_quantile (float in [0, 0.5) or None):
            controls how much of the `daily_vaccines` schedule counts
            as the "vaccination season" when back-deriving the
            `vax_induced_*_risk_reduce_initial` values (only used when
            `adjust_VE_for_seasonal_waning` is True).

            The season average is taken as an unweighted average over
            the days of the season window, so a schedule with a long
            low-dose tail (a few doses trickling through the spring,
            say) stretches that window over months with almost no
            vaccination, pulling the average down and inflating the
            derived peak efficacy. Setting this to q trims the window
            to the days spanning the central (1 - 2q) of the season's
            cumulative doses, dropping the sparse tails at both ends.

            None or 0 (default) uses the full first-dose-to-last-dose
            window. 0.05 and 0.10 are reasonable choices. This has no
            effect on schedules whose doses are spread evenly across
            the season, and a substantial effect on campaign-shaped
            schedules that taper off.
        vax_protection_delay_days: (positive int):
            number of days after vaccination until vaccine
            protection is effective.
        vax_immunity_reset_date_mm_dd: (str or None):
            date (in "mm_dd" format) each year when vaccine
            immunity resets, and date from which to start
            calculating contribution of vaccines to
            vaccine-induced immunity.
        infection_immunity_start_date_mm_dd: (str or None):
            date (in "mm_dd" format) that the input initial value
            of infection-induced immunity (M) corresponds to.
            If this date is before start_real_date, M(0) is
            decayed forward (waning only) to start_real_date. If
            this date is after start_real_date, M(0) is set to 0
            at start_real_date and the input M(0) is instead
            added to M once this date is reached during the
            simulation.

            The calendar year is resolved to whichever of
            start_real_date's year, the year before, or the year
            after puts the date CLOSEST to start_real_date -- so
            for a season starting 2024-08-15, "01_01" resolves to
            2025-01-01 (139 days after the start) rather than
            2024-01-01 (227 days before it). A resolved date more
            than `INFECTION_IMMUNITY_START_DATE_WARN_DAYS` days
            from start_real_date raises a warning, since this
            parameter is meant to describe an immunity snapshot
            taken within a couple of months of the simulation
            start.
        infection_immunity_injection_val (np.ndarray of nonnegative floats):
            the M value waiting to be injected once
            infection_immunity_start_date_mm_dd is reached, or an
            array of zeros if no injection is pending. The user does
            not specify this -- `FluSubpopModel` derives it from the
            input M(0) and keeps it in sync on `reset_simulation`. It
            lives on the params (rather than only on the
            `InfInducedImmunity` epi metric) so that the torch
            metapopulation model, which has no epi metric objects,
            can apply the same injection.
        R_to_S_rate (positive float):
            rate at which people in R move to S.
        E_to_I_rate (positive float):
            rate at which people in E move to I (both
            IP and IA, infected pre-symptomatic and infected
            asymptomatic)
        IP_to_IS_rate (positive float):
            rate a which people in IP (infected pre-symptomatic)
            move to IS (infected symptomatic)
        ISR_to_R_rate (positive float):
            rate at which people in IS (infected symptomatic)
            move to R.
        IA_to_R_rate (positive float):
            rate at which people in IA (infected asymptomatic)
            move to R
        ISH_to_H_rate (positive float):
            rate at which people in IS (infected symptomatic)
            move to H.
        HR_to_R_rate (positive float):
            rate at which people in H move to R.
        HD_to_D_rate (positive float):
            rate at which people in H move to D.
        E_to_IA_prop (np.ndarray of positive floats in [0,1]):
            proportion exposed who are asymptomatic based on
            age-risk groups.
        IP_to_ISH_prop (np.ndarray of positive floats in [0,1]):
            proportion infected who are hospitalized
            based on age-risk groups.
        ISH_to_HD_prop (np.ndarray of positive floats in [0,1]):
            proportion hospitalized who die based on
            age-risk groups.
        IP_relative_inf (positive float):
            relative infectiousness of pre-symptomatic to symptomatic
            people (IP to IS compartment).
        IA_relative_inf (positive float):
            relative infectiousness of asymptomatic to symptomatic
            people (IA to IS compartment).
        relative_suscept (np.ndarray of positive floats in [0,1]):
            relative susceptibility to infection by age group
        total_contact_matrix (np.ndarray of positive floats):
            A x A contact matrix (where A is the number
            of age groups), where element i,j is the average
            contacts from age group j that an individual in
            age group i has
        school_contact_matrix (np.ndarray of positive floats):
            A x A contact matrix (where A is the number
            of age groups), where element i,j is the average
            contacts from age group j that an individual in
            age group i has at school -- this matrix plus the
            work_contact_matrix must be less than the
            total_contact_matrix, element-wise
        work_contact_matrix (np.ndarray of positive floats):
            A x A contact matrix (where A is the number
            of age groups), where element i,j is the average
            contacts from age group j that an individual in
            age group i has at work -- this matrix plus the
            work_contact_matrix must be less than the
            total_contact_matrix, element-wise
    """

    num_age_groups: Optional[int] = None
    num_risk_groups: Optional[int] = None
    start_real_date: Optional[datetime.date] = None
    beta_baseline: Optional[float] = None
    total_pop_age_risk: Optional[np.ndarray] = None
    humidity_impact: Optional[float] = None

    inf_induced_saturation: Optional[float] = None
    inf_induced_immune_wane: Optional[float] = None
    vax_induced_immune_wane: Optional[float] = None
    inf_induced_inf_risk_reduce: Optional[float] = None
    inf_induced_hosp_risk_reduce: Optional[float] = None
    inf_induced_death_risk_reduce: Optional[float] = None
    vax_induced_inf_risk_reduce: Optional[float] = None
    vax_induced_hosp_risk_reduce: Optional[float] = None
    vax_induced_death_risk_reduce: Optional[float] = None
    vax_induced_inf_risk_reduce_initial: Optional[np.ndarray] = None
    vax_induced_hosp_risk_reduce_initial: Optional[np.ndarray] = None
    vax_induced_death_risk_reduce_initial: Optional[np.ndarray] = None
    adjust_VE_for_seasonal_waning: Optional[bool] = True
    VE_season_dose_window_quantile: Optional[float] = None
    vax_protection_delay_days: Optional[int] = 0
    vax_immunity_reset_date_mm_dd: Optional[str] = None
    infection_immunity_start_date_mm_dd: Optional[str] = None
    infection_immunity_injection_val: Optional[np.ndarray] = None

    R_to_S_rate: Optional[float] = None
    E_to_I_rate: Optional[float] = None
    IP_to_IS_rate: Optional[float] = None
    ISR_to_R_rate: Optional[float] = None
    IA_to_R_rate: Optional[float] = None
    ISH_to_H_rate: Optional[float] = None
    HR_to_R_rate: Optional[float] = None
    HD_to_D_rate: Optional[float] = None
    
    E_to_IA_prop: Optional[np.ndarray] = None
    IP_to_ISH_prop: Optional[torch.Tensor] = None
    ISH_to_HD_prop: Optional[torch.Tensor] = None

    IP_relative_inf: Optional[float] = None
    IA_relative_inf: Optional[float] = None

    relative_suscept: Optional[np.ndarray] = None

    total_contact_matrix: Optional[np.ndarray] = None
    school_contact_matrix: Optional[np.ndarray] = None
    work_contact_matrix: Optional[np.ndarray] = None


@dataclass
class FluSubpopSchedules:
    """
    Data container for dataframes used to specify schedules
    for each `FluSubpopModel` instance.

    THE FORMAT FOR EACH DATAFRAME IS VERY IMPORTANT -- please
    read and implement carefully.

    Attributes:
        absolute_humidity (pd.DataFrame):
            must have columns "date" and "absolute_humidity" --
            "date" entries must correspond to consecutive calendar days
            and must either be strings with `"YYYY-MM-DD"` format or
            `datetime.date` objects -- "value" entries correspond to
            absolute humidity on those days
        flu_contact_matrix (pd.DataFrame):
            must have columns "date", "is_school_day", and "is_work_day"
            -- "date" entries must correspond to consecutive calendar
            days and must either be strings with `"YYYY-MM-DD"` format
            or `datetime.date` object and "is_school_day" and
            "is_work_day" entries are are floats between 0 and 1 
            indicating if that date is a school day or work day
        daily_vaccines (pd.DataFrame):
            must have "date" and "daily_vaccines" -- "date" entries must
            correspond to consecutive calendar days and must either
            be strings with `"YYYY-MM-DD"` format or `datetime.date`
            objects -- "value" entries correspond to historical
            proportion vaccinated on those days
        mobility_modifier (pd.DataFrame):
            must have columns "mobility_modifier" and either "date"
            or "day_of_week" -- "date" entries must
            correspond to consecutive calendar days and must either
            be strings with `"YYYY-MM-DD"` format or `datetime.date`
            objects -- "day_of_week" entries are strings with values
            from Monday to Sunday (case doesn't matter).
            "mobility_modifier" entries are JSON-encoded A x R
            arrays representing the proportion of time spent away from
            home by age-risk group on those days
    """

    absolute_humidity: Optional[pd.DataFrame] = None
    flu_contact_matrix: Optional[pd.DataFrame] = None
    daily_vaccines: Optional[pd.DataFrame] = None
    mobility_modifier: Optional[pd.DataFrame] = None


@dataclass
class FluTravelStateTensors:
    """
    Data container for tensors for `FluMetapopModel` -- used to store arrays
    that contain data across all subpopulations (collected from each
    location/subpopulation model). Note that not all fields in
    `FluSubpopState` are included -- we only include compartments
    needed for the travel model computation, for efficiency.

    Attributes:
        IP (torch.Tensor of nonnegative integers):
            presymptomatic infected compartment for location-age-risk
            groups -- the lth element holds current_val of
            Compartment "IP" on the lth location / subpopulation
            on the associated `MetapopModel`.
        ISR (torch.Tensor of nonnegative integers):
            symptomatic infected compartment (that will recover)
            for location-age-risk groups -- the lth element holds
            current_val of Compartment "ISR" on the lth 
            location / subpopulation on the associated `MetapopModel`.
        ISH (torch.Tensor of nonnegative integers):
            symptomatic infected compartment (that will be hospitalized)
            for location-age-risk groups -- the lth element holds
            current_val of Compartment "ISH" on the lth 
            location / subpopulation on the associated `MetapopModel`.
        IA (torch.Tensor of nonnegative integers):
            asymptomatic infected compartment for location-age-risk
            groups -- the lth element holds current_val of
            Compartment "IA" on the lth location / subpopulation
            on the associated `MetapopModel`.
        HR (torch.Tensor of nonnegative integers):
            hospital compartment (that will recover) for location-age-risk
            groups -- the lth element holds current_val of
            Compartment "HR" on the lth location / subpopulation
            on the associated `MetapopModel`.
        HD (torch.Tensor of nonnegative integers):
            hospital compartment (that will die) for location-age-risk
            groups -- the lth element holds current_val of
            Compartment "HD" on the lth location / subpopulation
            on the associated `MetapopModel`.
        flu_contact_matrix (torch.Tensor of nonnegative integers):
            contact matrix for location-age-risk groups -- the
            lth element holds current_val of `FluContactMatrix`
            `Schedule` for subpopulation l -- this value is a
            combination of the total contact matrix, the
            work contact matrix, and the school contact matrix
            (and the value is adjusted depending on whether
            the date is a work or school day)
        init_vals (dict):
            dictionary of torch.Tensor instances, where keys
            correspond to "IP", "ISR", "ISH", "IA", "HR", and "HD", and
            values correspond to their initial values for
            location-age-risk groups.
        mobility_modifier (torch.Tensor of positive floats):
            mobility modifier for location-age-risk groups -- the
            lth element holds current_val of `MobilityModifier`
            `Schedule` for subpopulation l -- represents the proportion
            of time spent away from home by age group
    """

    IP: torch.Tensor = None
    ISR: torch.Tensor = None
    ISH: torch.Tensor = None
    IA: torch.Tensor = None
    HR: torch.Tensor = None
    HD: torch.Tensor = None

    flu_contact_matrix: torch.Tensor = None
    mobility_modifier: torch.Tensor = None

    init_vals: dict = field(default_factory=dict)

    # Note: `init_vals: dict = {}` does NOT work --
    #   gives "mutable default" argument

    def save_current_vals_as_init_vals(self):

        for field in fields(self):
            if field.name == "init_vals":
                continue
            self.init_vals[field.name] = getattr(self, field.name).clone()

    def reset_to_init_vals(self):

        for name, val in self.init_vals.items():
            setattr(self, name, val.clone())


@dataclass
class FluTravelParamsTensors:
    """
    Data container for tensors for `FluMetapopModel` -- used to store arrays
    that contain data across all subpopulations (collected from parameters
    on each location/subpopulation model, as well as from the
    metapopulation's associated `FluMixingParams` instance).
    Note that not all fields in `FluSubpopParams` are included
    -- we only include parameters needed for the travel model
    computation, for efficiency.

    Attributes:
        num_locations (torch.Tensor, 0-dimensional):
            number of locations (subpopulations) in the
            metapopulation model and therefore the travel
            model.
        travel_proportions (torch.Tensor):
            L x L array, where L is the number of locations
            or subpopulations, where element i,j corresponds
            to the proportion of the population in location i
            who travels to location j (on average).

    See `FluSubpopParams` docstring for other attributes.

    Fields are analogous -- but (most) are size (L, A, R) for
    location-age-risk or size 0 tensors. Exceptions are
    `travel_proportions`, which is size (L, L),
    and any of the contact matrices, which are size (L, A, A).
    """

    num_locations: Optional[torch.tensor] = None
    num_age_groups: Optional[torch.tensor] = None
    num_risk_groups: Optional[torch.tensor] = None

    travel_proportions: torch.Tensor = None

    IP_relative_inf: torch.Tensor = None
    IA_relative_inf: torch.Tensor = None

    relative_suscept: torch.Tensor = None

    total_contact_matrix: Optional[torch.Tensor] = None
    school_contact_matrix: Optional[torch.Tensor] = None
    work_contact_matrix: Optional[torch.Tensor] = None

    def standardize_shapes(self) -> None:
        """
        If field is size (L, A, R) for location-age-risk or size 0 tensors,
            or is not a special variable listed below, then apply dimension
            expansion so that fields are size (L, A, R) tensors for tensor multiplication.

        Exceptions are `travel_proportions`, which is size (L, L),
        and any of the contact matrices, which are size (L, A, A).

        Not all dimension combinations are considered not all make sense --
        we assume that we only have risk IF we have age, for example.
        """

        L = int(self.num_locations.item())
        A = int(self.num_age_groups.item())
        R = int(self.num_risk_groups.item())

        error_str = " Each SubpopParams field must have size (L, A, R) " \
                    "(for location-age-risk groups) or size 0 -- please check files " \
                    "and inputs, then try again."

        for name, value in vars(self).items():

            # Ignore the field that corresponds to a dictionary
            if name == "init_vals":
                continue

            elif name == "travel_proportions":
                if value.size() != torch.Size([L, L]):
                    raise Exception(str(name) + error_str)

            # `total_contact_matrix`, `school_contact_matrix`, `work_contact_matrix`
            elif "contact_matrix" in name:
                if value.size() == torch.Size([L, A, A]):
                    continue
                elif value.size() != torch.Size([A, A]):
                    raise Exception(str(name) + error_str)
                else:
                    setattr(self, name, value.view(1, A, A).expand(L, A, A))
            
            # string parameters (and unset optional parameters)
            elif value is None or isinstance(value, str) or isinstance(value, datetime.date):
                continue

            # If scalar or already L x A x R, do not need to adjust
            #   dimensions
            elif value.size() == torch.Size([]):
                continue

            elif value.size() == torch.Size([L, A, R]):
                continue

            elif value.size() == torch.Size([L]):
                setattr(self, name, value.view(L, 1, 1).expand(L, A, R))

            elif value.size() == torch.Size([A, R]):
                setattr(self, name, value.view(1, A, R).expand(L, A, R))
            
            else:
                value_size = str(value.size())
                raise Exception(str(name) + ' with size ' + value_size + error_str)


@dataclass
class FluFullMetapopStateTensors(FluTravelStateTensors):
    """
    Data container for tensors for `FluMetapopModel` -- used to store arrays that
    contain data across all subpopulations (collected from each
    location/subpopulation model). In contrast to `FluTravelStateTensors`,
    ALL fields in `FluSubpopState` are included -- this is
    for running the simulation via torch.

    Attributes:
        flu_contact_matrix (torch.Tensor of nonnegative integers):
            contact matrix for location-age-risk groups -- the
            lth element holds current_val of `FluContactMatrix`
            `Schedule` for subpopulation l -- this value is a
            combination of the total contact matrix, the
            work contact matrix, and the school contact matrix
            (and the value is adjusted depending on whether
            the date is a work or school day)
        init_vals (dict):
            dictionary of torch.Tensor instances, where keys
            correspond to "IP", "ISR", "ISH" "IA", "HR", and "HD", and values
            correspond to their initial values for location-age-risk
            groups.

    See `FluSubpopState` and `FluTravelStateTensors` for other
        attributes -- other attributes here correspond to
        `FluSubpopState`, but are size (L, A, R) tensors for
        location-age-risk or size 0 tensors.
    """

    # `IP`, `ISR`, `ISH`, `IA`, `HR`, `HD`, `flu_contact_matrix`, `mobility_modifier`
    #   already in parent class
    # Same with `init_vals`

    S: Optional[torch.Tensor] = None
    E: Optional[torch.Tensor] = None
    R: Optional[torch.Tensor] = None
    D: Optional[torch.Tensor] = None

    M: Optional[torch.Tensor] = None
    MV: Optional[torch.Tensor] = None

    absolute_humidity: Optional[float] = None
    daily_vaccines: Optional[torch.Tensor] = None


@dataclass
class FluFullMetapopParamsTensors(FluTravelParamsTensors):
    """
    Data container for tensors for `FluMetapopModel` -- used to store arrays that
    contain data across all subpopulations (collected from parameters
    on each location/subpopulation model, as well as from the
    metapopulation's associated `FluMixingParams` instance).
    Note that in contrast to `FluTravelParamsTensors`,
    ALL fields in `FluSubpopParams` are included --
    this is for running the simulation via torch.

    Attributes:
        num_locations (torch.Tensor, 0-dimensional):
            number of locations (subpopulations) in the
            metapopulation model and therefore the travel
            model.
        travel_proportions (torch.Tensor):
            L x L array, where L is the number of locations
            or subpopulations, where element i,j corresponds
            to the proportion of the population in location i
            who travels to location j (on average).

    See `FluSubpopParams` docstring for other attributes.
    Other fields are analogous except they are size (L, A, R)
    tensors or size 0 tensors.
    """

    # non_numerical_params: Optional[dict] = None
    start_real_date: Optional[datetime.date] = None
    beta_baseline: Optional[torch.Tensor] = None
    total_pop_age_risk: Optional[torch.Tensor] = None
    humidity_impact: Optional[torch.Tensor] = None

    inf_induced_saturation: Optional[torch.Tensor] = None
    inf_induced_immune_wane: Optional[torch.Tensor] = None
    vax_induced_immune_wane: Optional[torch.Tensor] = None
    inf_induced_inf_risk_reduce: Optional[torch.Tensor] = None
    inf_induced_hosp_risk_reduce: Optional[torch.Tensor] = None
    inf_induced_death_risk_reduce: Optional[torch.Tensor] = None
    vax_induced_inf_risk_reduce: Optional[torch.Tensor] = None
    vax_induced_hosp_risk_reduce: Optional[torch.Tensor] = None
    vax_induced_death_risk_reduce: Optional[torch.Tensor] = None
    vax_induced_inf_risk_reduce_initial: Optional[torch.Tensor] = None
    vax_induced_hosp_risk_reduce_initial: Optional[torch.Tensor] = None
    vax_induced_death_risk_reduce_initial: Optional[torch.Tensor] = None
    adjust_VE_for_seasonal_waning: Optional[torch.Tensor] = True
    VE_season_dose_window_quantile: Optional[torch.Tensor] = None
    vax_protection_delay_days: Optional[torch.Tensor] = 0
    vax_immunity_reset_date_mm_dd: Optional[str] = None
    infection_immunity_start_date_mm_dd: Optional[str] = None
    infection_immunity_injection_val: Optional[torch.Tensor] = None

    R_to_S_rate: Optional[torch.Tensor] = None
    E_to_I_rate: Optional[torch.Tensor] = None
    IP_to_IS_rate: Optional[torch.Tensor] = None
    ISR_to_R_rate: Optional[float] = None
    IA_to_R_rate: Optional[float] = None
    ISH_to_H_rate: Optional[float] = None
    HR_to_R_rate: Optional[float] = None
    HD_to_D_rate: Optional[float] = None
    
    E_to_IA_prop: Optional[torch.Tensor] = None
    IP_to_ISH_prop: Optional[torch.Tensor] = None
    ISH_to_HD_prop: Optional[torch.Tensor] = None

    IP_relative_inf: Optional[torch.Tensor] = None
    IA_relative_inf: Optional[torch.Tensor] = None

    relative_suscept: Optional[torch.Tensor] = None


class FluPrecomputedTensors:
    """
    Stores precomputed quantities that are repeatedly
    used, for computational efficiency.
    """

    def __init__(self,
                 total_pop_LAR_tensor: torch.Tensor,
                 params: FluTravelParamsTensors) -> None:

        self.total_pop_LAR_tensor = total_pop_LAR_tensor

        self.L = int(params.num_locations.item())
        self.A = int(params.num_age_groups.item())
        self.R = int(params.num_risk_groups.item())

        self.total_pop_LA = torch.sum(self.total_pop_LAR_tensor, dim=2)

        # Remove the diagonal!
        self.nonlocal_travel_prop = params.travel_proportions.clone().fill_diagonal_(0.0)

        # We don't need einsum for residents traveling
        #   -- Dave and Remy helped me check this
        # \sum_{k \not = \ell} v^{\ell \rightarrow k}
        # Note we already have k \not = \ell because we set the diagonal of
        #   nonlocal_travel_prop to 0
        self.sum_residents_nonlocal_travel_prop = self.nonlocal_travel_prop.sum(dim=1)


@dataclass(frozen=True)
class FluMixingParams:
    """
    Contains parameters corresponding to inter-subpopulation
    (metapopulation model) specifications: the number of
    subpopulations included, and the travel proportions between them.

    Params:
        num_locations (int):
            Number of locations (subpopulations) in the
            metapopulation model.
        travel_proportions (np.ndarray of shape (A, R)):
            L x L array of floats in [0,1], where L is the number
            of locations (subpopulations), and the i-jth element
            is the proportion of people in subpopulation i that
            travel to subpopulation j.
    """

    num_locations: Optional[int]
    travel_proportions: Optional[np.ndarray]


@dataclass
class FluFullMetapopScheduleTensors:

    absolute_humidity: Optional[list[torch.tensor]] = None
    is_school_day: Optional[list[torch.tensor]] = None
    is_work_day: Optional[list[torch.tensor]] = None
    daily_vaccines: Optional[list[torch.tensor]] = None
    mobility_modifier: Optional[list[torch.tensor]] = None

