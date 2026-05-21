# Logging

The reticolo logger is threadsafe, OpenMP-aware, and zero-config in the
common case. Most apps interact with it through a single call:

```cpp
log::start(outpath);
```

That's it. Banner prints; per-run files land next to the HDF5 output; every
`Lattice`, `RNG`, `Writer`, action, algorithm, and replica auto-announces.
Trajectory / sweep / cluster updates self-log from inside the algorithm
method on each call.

This document is the full surface. Most apps need only the first section.

## The visual identity

Every line follows a fixed schema:

```
┃ <run>  <elapsed>     <tag>   <message>
```

Concrete sample from `phi4_hmc`:

```
┃ main 000:00:00.001  init  Lattice<double>  shape=4×4  sites=16
┃ main 000:00:00.001  rng   FastRng  seed=0x2a
┃ main 000:00:00.001  act   Phi4<double>
┃                             κ=0.137
┃                             λ=1.000
┃ main 000:00:00.002  io    opened /tmp/phi4_demo.h5
┃ main 000:00:00.002  hmc   HMC<Leapfrog>
┃                             τ=1.000
┃                             n_md=20
┃ main 000:00:00.002  hmc   prod   2 trajectories
┃ main 000:00:00.002  hmc   traj      3  ΔH=+1.359e-02  accept
```

**Sigil = severity.** The leftmost character changes with the level — `·`
for debug, `┃` for info, `⚠` for warn (yellow, → stderr), `✖` for error
(red, → stderr). Continuation lines (multi-line entries) keep the sigil so
the column reads continuously.

**Run column.** `main` for unscoped lines; `r000`, `r001`, … inside a
`log::scope("rNNN")`. Reserved for legitimate main-thread logs and for
per-replica lines. The string `----` only appears as a bug indicator: a
log call inside an OpenMP parallel region without a bound scope.

**Elapsed time.** `HHH:MM:SS.mmm` since program start. Banner stamps the
absolute start timestamp once.

**Tag.** 4 chars, hard-truncated. The taxonomy:

| tag    | who emits it                                     |
| ------ | ------------------------------------------------ |
| `init` | `Lattice`, `LinkLattice`, `MatrixLinkLattice` ctors |
| `rng`  | `FastRng`, `Mt19937Rng`, `RanluxRng` ctors      |
| `act`  | actions (via `log::act(action)`)                 |
| `hmc`  | `Hmc::step`, plus app phase markers              |
| `metr` | `Metropolis::step`                               |
| `wolf` | `Wolff::step`                                    |
| `repl` | `llr::Replica` ctor + `thermalize` + `sample`    |
| `llr`  | main-thread LLR orchestration                    |
| `exch` | replica exchange events                          |
| `io`   | `Writer` open                                    |

Apps emit ad-hoc orchestration lines (phase markers, iter completion, …)
inline via `log::info("hmc", "prod {} trajectories", n)` — there are
no specialised helpers.

## The two control axes

### Verbosity — `log::Mode` (per call)

Algorithm methods that log self-emit on each call. The opt-out is
per-call, via the `log::Mode` enum:

```cpp
hmc.step();                                // emits a line per traj
hmc.step(log::Mode::silent);               // computes the step, skips the log line
```

`log::Mode::normal` (default) — log on completion.
`log::Mode::silent` — don't log this call. The counter still advances,
so trajectory numbering is contiguous across silent / verbose mixes.

Typical use: thermalisation calls pass `silent` (uninteresting), production
calls use the default. LLR's `Replica::thermalize` / `Replica::sample`
take the same arg with the same semantics.

### Global switch — `log::off()` / `log::on()`

Total kill switch. `log::off()` short-circuits before any formatting; even
`log::banner()` is suppressed. Apps that need this:

- **Tests** — `tests/test_main.cpp` calls `log::off()` once before any
  Catch fixture runs. The 8 test cases in `tests/unit/test_log.cpp` that
  inspect logger output re-enable inside an RAII `StreamCapture` helper.
- **Benchmarks** — `bench_actions.cpp` etc. call `log::off()` so per-step
  emission doesn't skew the wall-time measurement.
- **`tune_phi4`** — same.

Default is `on` — apps that don't touch the logger get full output.

### Mode vs off: which to use

- `log::Mode::silent` is *per call site*. Use it when one specific loop
  shouldn't log but the rest of the run should.
- `log::off()` is *global*. Use it when nothing in the run should log.

They compose: an app that calls `log::off()` then `hmc.step()` gets
no output regardless of which Mode you pass.

## OpenMP — scopes and per-run files

Single-threaded apps need nothing. The interesting case is LLR, where
N replicas thermalise / measure concurrently:

```cpp
#pragma omp parallel for schedule(dynamic, 1)
for (std::size_t n = 0; n < n_rep_u; ++n) {
    auto _ = log::scope(std::string{reps[n]->id()});   // bind r0NN for this iteration
    auto& r = *reps[n];
    r.thermalize(n_therm);    // logs as `r0NN  repl  thermalize n=…`
    r.sample(n_meas);         // logs as `r0NN  repl  sample n=…  ⟨dE⟩=…`
}
```

`log::scope` is an RAII guard that pushes a string onto a thread-local
stack. Every subsequent log call from that thread reads the top of the
stack and emits the tag. Functions called from inside the scope (e.g.
`Replica::thermalize` → `Hmc::step`) inherit the binding without
having to receive a logger as a parameter.

**Per-run files.** `log::start(outpath)` calls `log::init_parallel(parent)`
which enables per-replica log files: `<outdir>/run.r000.log`,
`run.r001.log`, etc. Each replica's lines are duplicated into its own
file, so `tail -f run.r017.log` shows just that replica's progress.
stdout / stderr remain the interleaved view of everything.

The unscoped warning catches the bug case — if a log call happens inside
an OpenMP parallel region with no scope bound, stderr gets:

```
⚠ logger: log call inside parallel region without scope()
```

…and the line emits with `----` in the run column.

## What auto-announces

These types log on construction (provided the logger is on):

- `Lattice<T>` / `LinkLattice<T>` / `MatrixLinkLattice<G, T>` — shape +
  scalar type + site count.
- `FastRng` / `Mt19937Rng` / `RanluxRng` — seed.
- `io::Writer` — output path.
- `llr::Replica` — `Replica<scalar>` + window params, scoped to its own id.

These types have a `describe(log::Entry&)` method but do **not** auto-log;
apps call them explicitly:

- Actions (`act::Phi4`, `act::Wilson<G>`, …) are aggregates with no
  constructor to hook — apps call `log::act(action)` once.
- Algorithms (`alg::Hmc`, `alg::Metropolis`, `alg::Wolff`) are classes
  but don't auto-log — apps call `log::algo(hmc)` once.

The split between aggregate-action-explicit and class-replica-implicit is
a known wart; see `docs/logging_progress.md` for the rationale.

## Custom log lines from app code

App-level orchestration uses `log::info(tag, "fmt …", args …)` directly —
no specialised helpers. The format strings live at the call site:

```cpp
log::info("hmc", "therm  {} trajectories", n_therm);
log::info("llr", "NR phase  {} iters × {} replicas", n_nr, n_rep);
log::info("exch", "step  {:>3}  accepted  {}/{}", s, accepted, attempts);
```

`log::warn(tag, …)` and `log::error(tag, …)` go to stderr with their own
sigils and colour. `log::debug(tag, …)` is suppressed by default
(`set_min_level(Level::debug)` to enable).

Multi-line entries use a fluent builder:

```cpp
log::info("act")
    .line("Phi4<{}>", scalar_name<T>())
    .param("κ={:.3f}", kappa)
    .param("λ={:.3f}", lambda);
```

`.line(...)` is a continuation line at the message column; `.param(...)`
indents 2 spaces. Multi-line entries emit atomically — no thread can
interleave between an entry's first and last line.

## API summary

```cpp
// init / banner
log::start(std::filesystem::path const& outpath);   // parallel mode + banner
log::start();                                       // serial mode + banner

// global switch
log::off();                                         // suppress everything
log::on();                                          // reverse it
[[nodiscard]] bool log::enabled();

// severity filter
log::set_min_level(log::Level::debug | info | warn | error);
log::set_color(bool);                               // override auto-detect

// emission
log::info(tag, fmt, args...);    // → stdout
log::warn(tag, fmt, args...);    // → stderr
log::error(tag, fmt, args...);   // → stderr
log::debug(tag, fmt, args...);   // → stdout, suppressed by default
log::info(tag).line(...).param(...);     // multi-line builder

// thread-local scope
[[nodiscard]] log::Scope log::scope(std::string run_id);

// describe / announce
log::act(action);     // calls action.describe(); emits with "act" tag
log::algo(algo);      // calls algo.describe(); emits with algo's log_tag

// verbosity for algorithm method calls
enum class log::Mode { normal, silent };
```
