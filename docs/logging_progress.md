# Logging integration — progress

Live tracker for wiring `reticolo::log` (`include/reticolo/core/log.hpp`) into the
rest of the library. One subsystem at a time. Each entry must reach **approved**
before implementation begins.

## Locked decisions

- **Style**: heavy `┃` sigil per severity (`·`/`┃`/`⚠`/`✖`), tag column 4 chars
  (truncate/pad), elapsed timestamp `HHH:MM:SS.mmm`, split stdout/stderr.
- **Threading**: TLS scope stack via `log::scope()` RAII. Bindings propagate
  to nested calls automatically — library code never receives a logger.
- **Constructor-time logging in headers**: yes. Each subsystem logs itself
  once at construction. Uniformity > header weight.
- **No domain-specific render helpers.** Each algorithm phase calls
  `log::info("tag", "fmt …", args …)` inline at the emit site. The only
  things in `log_helpers.hpp` are `log::act<A>` and `log::algo<A>`, which
  are generic dispatchers over `describe()` — they exist because they
  share their implementation across all callers (action / algorithm /
  replica), not because they format a specific message.
- **Default level**: `info`. `debug` reserved for development.
- **Global on/off switch**: `log::off()` / `log::on()` (cfg.enabled flag).
  Cheap short-circuit before any formatting; also gates `banner()`. All tests
  link `tests/test_main.cpp` which calls `log::off()` before Catch runs, so
  Lattice / RNG / Writer / algorithm / replica construction lines stay
  invisible to test output. Apps that want output call nothing — `enabled`
  is `true` by default.
- **Verbosity mode**: `log::Mode` enum (`normal` / `silent` for now, extensible).
  Algorithms' `trajectory()`/`sweep()`/`update()` take an optional
  `log::Mode log_mode = log::Mode::normal` argument and log from inside. The
  internal step counter advances on every call (silent or not) — counter
  reflects work done, not lines printed.
- **Inner-loop ban**: no logger calls inside `s_full`, `compute_force`,
  `for_each_update`, RNG draws, integrator step bodies, `Series::append`,
  Wolff DFS body, Metropolis per-site update. Trusted via review.
- **Parameter format**: always one parameter per line via `.param()`. First
  line is the concept name + type params (`Wilson<SU2>`) emitted with
  `.line(...)`; every constructor argument gets its own `.param(...)` call,
  which prepends a 2-space indent so parameters sit visually nested under
  the concept name. Uniform regardless of parameter count.

## Tag taxonomy (4-char hard cap)

| Tag    | Owns                             | Frequency             |
|--------|----------------------------------|-----------------------|
| `init` | lattices, fields, geometry       | once at construction  |
| `rng`  | FastRng / MT19937 / RanLux       | seed + reseed         |
| `act`  | all actions (Wilson, phi4, …)    | once at construction  |
| `hmc`  | HMC trajectories                 | once per trajectory   |
| `metr` | Metropolis sweeps                | once per sweep        |
| `wolf` | Wolff cluster updates            | once per update       |
| `intg` | Leapfrog / Omelyan2              | construction only     |
| `llr`  | Replica + WindowedAction         | construction + NR/RM  |
| `exch` | Replica exchange                 | per exchange step     |
| `meas` | Observables                      | once per measurement  |
| `io`   | Writer open/close, Series open   | lifecycle only        |
| `cli`  | parsed args                      | once at startup       |
| `phys` | physics-axis warns               | rare                  |
| `num`  | numerics-axis warns              | rare                  |

## Subsystem checklist

| Subsystem  | Status   | Notes |
|------------|----------|-------|
| actions    | done     | `describe()` member on each of 8 actions; `log::act(a)` helper in `log_helpers.hpp`. Apps still need to call it (next pass). |
| algorithms | done     | `log_tag` + `describe()` + `step_count_` on Hmc/Metropolis/Wolff. `trajectory()`/`sweep()`/`update()` log from inside; opt-out via `log::Mode::silent`. Replicas pass silent. |
| integrators| done     | `static constexpr std::string_view name` on Leapfrog/Omelyan2/Omelyan4 — consumed by `HMC<...>` describe line. |
| llr        | done     | Replica gained `id` + `log_tag` + `describe()`; ctor self-announces with its own scope bound; `thermalize`/`sample` self-log inline with `log::Mode` arg. All 4 LLR apps (u1/phi4/bose_gas/su2) wired end-to-end. Placeholder rename: `main` (unscoped, no parallel region) vs `----` (in parallel region — bug). App entry simplified to one `log::start(outpath)` call. |
| core       | done     | `Lattice` / `LinkLattice` / `MatrixLinkLattice` ctors log `init` lines (`scalar_name<T>` + `shape_str` + sites/links count). Sibling ctors (those sharing an existing `Indexing`) stay silent — they're internal HMC plumbing (mom/force/old_field). `FastRng` / `Mt19937Rng` / `RanluxRng` ctors log `rng` lines with the seed. Tests don't see any of this because `tests/test_main.cpp` calls `log::off()` first. |
| io         | done     | `Writer` ctor logs `io  opened {path}` once. Series append is hot and stays silent. |
| cli        | skipped  | Parser stamps argv into `/vars@*` already; duplicating it as a log line adds noise without insight. |
| obs        | skipped  | Stateless free functions — no construction event. Apps log measurements inline if they want them. |
| apps (LLR) | done     | `log::start(outpath)` + `log::act(base)` + scope at parallel-for top + inline `log::info` for phase/iter/exch markers + per-iter `log::info` line. |
| apps (HMC) | done     | `phi4_hmc`, `phi6_hmc`, `sine_gordon_hmc`, `bose_gas_hmc`, `u1_hmc`, `su2_hmc`, `su3_hmc`: `log::start(outpath)` + `log::act(base)` + `log::algo(hmc)` + therm silent + phase markers via inline `log::info`. |
| apps (MC)  | done     | `u1_metropolis`, `on_sigma_metropolis`: same pattern with `log::algo(metro)`. |
| apps (Wolff)| done    | `xy_wolff`, `on_sigma_wolff`: same pattern with `log::algo(wolff)`. |
| apps (tune)| done     | `tune_phi4`: `log::set_min_level(warn)` to suppress per-step noise during timing — algorithm-only wall time is the signal. |
| apps (bench)| done    | All `bench_*`: `log::set_min_level(warn)` only. No banner, no announces — pure timing rigs. |

## Open questions

- How do action types render their template params? `Wilson<SU2>` vs raw
  typeid spelling — leaning towards a tiny `gauge_group_name<G>()` traits
  helper colocated with the gauge groups, no new central registry.
- Should `log::debug` for RNG seed echoes be enabled by default? Or strictly
  development-only? Probably the latter — production runs already stamp the
  seed into HDF5 `/run@*` metadata.
