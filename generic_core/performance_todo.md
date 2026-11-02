# Performance: possible future work

Optional ideas for making `generic_core` simulations faster, recorded so they
are not lost. None of this is needed for correctness. See `limitations.md`
for current constraints and `architecture.md` for the design these build on.

---

## Where the time goes (measured 2026-11-02)

One deterministic MA_vax run (`run_simulations_MA_vax_IHR_ub_5.py` baseline:
single population, 7 age groups x 1 risk group, 11 transition variables,
307 days x 7 timesteps/day):

- build ≈ 0.10 s, simulate ≈ 0.21 s, i.e. **≈ 100 µs per timestep**.
- The arithmetic per timestep is a few hundred flops on 7x1 arrays (well
  under 1 µs compiled). Almost all of the 100 µs is Python overhead.
- The profile is flat: thousands of small calls (`sample_transitions`,
  `enforce_outflow_capacity`, `numpy_rate`, `np.full`, `np.sum`,
  `get_probabilities_array`, ...). There is no single hot spot to optimise.

So line-level fixes give little. The overhead comes from the structure:
every compartment and transition variable is a Python object that is
stepped one at a time, on tiny arrays, one run at a time.

Simulation speed matters mainly for:

- **MCMC / ABC fitting.** The IHR_ub_5 fit was 66 walkers x 4000 steps
  ≈ 264k simulations ≈ 15 CPU-hours (≈ 1.5 h on 10 processes). emcee and
  pyabc call the numpy simulator. The torch path is used only for the
  gradient (Adam/LBFGS) point fits.
- **Large ensembles** (replicates x scenarios x parameter sets).
- **Metapopulation runs.** Cost grows roughly linearly with the number of
  subpopulations.

A single interactive run (0.2 s) is not a problem.

**Done:** `save_history` now uses `ndarray.copy()` instead of `copy.deepcopy`
(`clt_toolkit/base_components.py`, `copy_history_val`). Results are
bit-identical. This saves ≈ 5% per run.

**Rejected for now: C++ core.** It would add a third simulator alongside
numpy and torch (C++ cannot compute gradients, so torch would stay). Every
new rate template would need C++ and a recompile. The core would also become
hard for non-C++ readers to inspect or debug. Revisit only for very large
metapopulations (hundreds of subpops), agent-level models, or shipping a
frozen model to users without Python.

---

## TODO A — Simulate many runs at once (batch dimension)

**Idea.** Give every state, parameter and schedule array a leading batch
dimension: `(B, A, R)` instead of `(A, R)`, and `(B, L, A, R)` for metapop.
One Python-level timestep then advances B runs (walkers, particles,
replicates, scenarios), so the per-call overhead is paid once instead of B
times. Expected gain: close to B-fold on simulation time for fitting and
ensembles, while staying in numpy.

**This combines with the existing process pool:** each worker simulates a
batch instead of a single run.

**What can vary within a batch:** anything numeric, i.e. parameter values,
initial conditions, seed scales, dose schedules and dose multipliers,
`m(t)` increments. **What must be shared:** model structure (compartments,
transitions, templates), start date, number of days, `timesteps_per_day`.
Batches built for fitting or ensembles always meet this.

**Which simulator to batch.**

- *Torch path (`torch_generic.py`)*: the easier code change, but it is
  **deterministic-only** (it implements `binom_deterministic_no_round`
  only). Batching it would speed up deterministic MCMC/ABC and
  deterministic ensembles, but not stochastic runs.
  - It is already functional (dicts of `(L, A, R)` tensors).
  - The rate code uses `sum(dim=-1)` and `matmul`, which already broadcast
    over an extra leading dimension.
  - Work would be in `_standardize_param_tensor`, `build_schedules_dict`,
    history stacking and the per-timestep transition step.
  - Adding stochastic sampling there (`torch.binomial`, not differentiable)
    is a possible extension.
- *Numpy path (`clt_toolkit` + `generic_model.py`)*: needed for stochastic
  batching, and a **large change**. `Compartment`, `TransitionVariable`,
  `EpiMetric` and every `axis=1` reduction in the rate/metric templates
  assume `(A, R)`. `clt_toolkit` is shared with `flu_core`, so flu would be
  affected too. A more contained option is a separate batched stepping
  function for generic models, kept consistent with the object model by
  parity tests (as the torch path is today).

**Things to handle.**

- *Branches on simulated values* must become per-run selects (`np.where` /
  `torch.where`). Today's M-injection and MV-reset events branch only on
  the calendar date, so they stay shared. They would need changing only if
  their dates became fitted or varied parameters.
- *`enforce_outflow_capacity` and `scheduled_exact`* need per-run handling
  when one run in the batch hits the cap and another doesn't.
- *Random numbers*: one generator drawing for a whole batch maps seeds to
  runs differently, so stochastic results would be statistically equivalent
  but not run-for-run identical to today's seeds.
- *Fitting integration*:
  - emcee supports batched log-prob directly (`vectorize=True`).
  - Accept-reject is trivial.
  - pyabc samples one particle at a time, but `fitting.py` already runs the
    ABC-SMC generations in its own loop, so particles can be batched there.
  - Runs that produce NaN or are rejected early are handled with masks.
- *Memory*: saved history grows by B. Fine for calibration targets. Full
  histories of big ensembles need splitting into chunks (as the export code
  already does).

**First step.** Before any rewrite, benchmark a hand-batched torch run
(B = 66) against 66 numpy runs. That checks that torch's per-op overhead on
small float64 CPU tensors doesn't eat the gain.

---

## TODO B — Rewrite the simulator core in JAX

**Idea.** Replace the time-stepping core (the numpy object stepping used by
generic models plus `torch_generic.py`) with one JAX implementation:

- `jax.jit` compiles the step function.
- `jax.vmap` adds the batch dimension automatically, which covers TODO A,
  including stochastic runs (`jax.random.binomial`).
- `jax.grad` replaces the torch path for gradient fits. It also opens up
  gradient-based samplers (NUTS/HMC via numpyro or blackjax), which should
  handle 33 fitted parameters much better than emcee's ensemble sampler.

So instead of keeping two simulators consistent, there would be one.

**What stays numpy/pandas.** Everything outside the step function: config
parsing and splitting, template registries and validation, schedule
building from CSVs, `model_factory`, `results_io`/duckdb, analysis,
plotting, the notebook, and all of `flu_core`/`clt_toolkit`. Arrays are
converted to JAX on the way into the simulator and back to numpy on the way
out. emcee and pyabc keep working because they just receive numpy values.

**What the core would look like.** Like `torch_generic.py`, not like
`generic_model.py`. Most lines translate directly (`jax.numpy` follows the
numpy API, and the `torch_rate` methods already take
`(state_dict, params_dict)`), but the object model (mutable `Compartment` /
`TransitionVariable` attributes, history lists) does not survive inside the
compiled region. Style changes:

- Arrays are immutable: `x[i] = v` becomes `x = x.at[i].set(v)`, and there
  is no in-place `+=`.
- The day/timestep loop becomes `jax.lax.scan`. A Python loop would be
  traced and compiled as about 2k separate steps.
- Branches on simulated values become `jnp.where` / `lax.cond`.
  Calendar-date events can be precomputed as per-day masks before the loop.
- Random numbers use explicit keys: `jax.random.binomial(key, n, p)`.
- Enable float64 (`jax_enable_x64`). JAX defaults to float32.

**Costs.**

- A one-time rewrite of the core, plus parity tests against the numpy
  object model (as `tests/test_generic_torch.py` does for torch).
- Compile time of a few seconds per model configuration and batch shape.
  A different batch size recompiles.
- Harder to debug than numpy: printing inside compiled code needs
  `jax.debug.print`, and errors refer to traced values.
- Stochastic results would not be run-for-run identical to the numpy RNG.
- CPU is the realistic target on macOS (Metal support is experimental).

**Relation to TODO A.** These are alternatives. A is the smaller step and
keeps both existing simulators. B is a bigger one-time rewrite that leaves
a single simulator to keep correct.
