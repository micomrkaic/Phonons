# crystal_phonons

A 2D phonon simulator in C/SDL2 with an in-window control panel. Atoms on
a square lattice oscillate under nearest-neighbour harmonic forces. Beside
the lattice, the panel renders the analytical dispersion ω(k) along
Γ–X–M–Γ, exposes the lattice parameters as live sliders, lists the key
bindings, and prints the running diagnostics. Press V to record an mp4
of the window.

## Build

Requires SDL2, SDL2_ttf, the DejaVu fonts, and (optionally, for video
capture) ffmpeg with libx264. On Debian/Ubuntu:

```sh
sudo apt install libsdl2-dev libsdl2-ttf-dev fonts-dejavu ffmpeg
make
```

| Target            | What it does                                        |
|-------------------|-----------------------------------------------------|
| `make`            | Release build with `-O2 -Wall -Wextra -Wpedantic`   |
| `make debug`      | `-O0 -g3` with AddressSanitizer + UBSan             |
| `make run`        | Build, then run                                     |
| `make check-deps` | Verify SDL2, SDL2_ttf, DejaVu font, ffmpeg          |
| `make clean`      | Remove object file and binary                       |

## Run

```sh
./crystal_phonons
```

Nothing is printed to the terminal during the run. All diagnostics, key
bindings, and adjustable parameters live in the right-hand panel.

### Layout

| Region              | Contents                                              |
|---------------------|-------------------------------------------------------|
| Left, 720×720       | Lattice. Disks coloured by `\|u\|`, `\|v\|`, or local KE. |
| Right, top          | Dispersion ω(k) along Γ–X–M–Γ, both branches.         |
| Right, middle       | Five mouse-draggable parameter sliders.               |
| Right, bottom-left  | Key bindings.                                         |
| Right, bottom-right | Live diagnostics (t, KE, PE, E_total, kT_eff, …).     |

A yellow vertical marker in the dispersion plot tracks the currently
selected `k_n`, so dragging the `k_n` slider shows where on the path the
1/2 buttons will excite.

### Sliders

| Slider                | Range          | When it takes effect                  |
|-----------------------|----------------|---------------------------------------|
| `K_L`                 | 0.10 – 4.00    | Immediately (forces use it each step) |
| `K_T`                 | 0.05 – 2.00    | Immediately                           |
| `kT_target`           | 0.00 – 2.00    | When you press **T**                  |
| `amp` (excitation)    | 0.00 – 0.50    | When you press **1**, **2**, or **3** |
| `k_n` (mode index)    | 1 – N/2 (24)   | When you press **1** or **2**         |

`K_T` is clamped above 0.05: at exactly zero the lattice has soft modes
along the principal axes and any thermal energy diverges.

### Keys

| Key       | Action                                                       |
|-----------|--------------------------------------------------------------|
| `1`       | TA mode at k = (2π·k_n/N, 0), polarisation ŷ                |
| `2`       | LA mode at k = (2π·k_n/N, 0), polarisation x̂                |
| `3`       | M-point mode (k = π, π)                                      |
| `T`       | Maxwell–Boltzmann velocities at current `kT_target`          |
| `D`       | Cycle colour scalar: \|u\|, \|v\|, local KE                  |
| `R`       | Reset displacements and velocities                           |
| `V`       | Start / stop video capture                                   |
| `Space`   | Pause / resume                                               |
| `Q`, esc  | Quit                                                         |

### Video capture

Press **V** to start recording. The program pipes the raw RGBA frame
buffer to `ffmpeg`:

```
ffmpeg -y -f rawvideo -pix_fmt rgba -s 1340x760 -r 60 -i - \
       -c:v libx264 -preset veryfast -pix_fmt yuv420p \
       phonons_YYYYMMDD_HHMMSS.mp4
```

A red `REC` badge with the output filename appears in the corner of the
lattice while recording. Press **V** again to stop and close the file.
Output lands in the current working directory. `SIGPIPE` is ignored, so
if `ffmpeg` is missing or dies, recording silently stops without taking
the simulator down with it.

The frame readback (≈4 MB/frame at 60 fps ≈ 240 MB/s) is the bottleneck.
On a fast machine `libx264 -preset veryfast` keeps up; if your machine is
slower, edit the command in `start_recording()` to use `ultrafast` or
encode to a smaller format.

## Physics

### Model

N×N square lattice with periodic boundaries. Each nearest-neighbour bond
carries separate longitudinal (K_L) and transverse (K_T) harmonic
stiffnesses, so a bond along +x stores

> U₊ₓ = ½ K_L (Δuₓ)² + ½ K_T (Δu_y)²

With this choice x- and y-polarised motions decouple into two acoustic
branches:

> ωₓ²(k) = (4 K_L/m) sin²(kₓ/2) + (4 K_T/m) sin²(k_y/2)
> ω_y²(k) = (4 K_T/m) sin²(kₓ/2) + (4 K_L/m) sin²(k_y/2)

Long-wavelength sound speeds are c_L = a √(K_L/m) and c_T = a √(K_T/m).

### Integration

Velocity Verlet at fixed dt. Total energy is the conservation diagnostic;
it is bounded (no secular drift) for symplectic Verlet, modulo round-off.

### Specific heat

Classical molecular dynamics has no ℏ, so equipartition holds:
⟨KE⟩ = ⟨PE⟩ = N kT and C_v = 2 N k_B (Dulong–Petit; the factor 2 is the
two displacement degrees of freedom per atom in 2D). The interesting
quantum behaviour comes from weighting each mode by its Bose–Einstein
occupation:

> C_v(T) = Σ_k Σ_branch (ℏω_k)² /(k_B T²) · eˣ / (eˣ − 1)²,
>     x = ℏω_k / k_B T

Implemented as `cv_quantum(double T)` (units ℏ = k_B = 1) and shown live
on the diagnostics panel for the current `kT_target`. Asymptotics in 2D:

- High T: C_v → 2 N² k_B (Dulong–Petit limit)
- Low T:  C_v ∝ T² (Debye T^d with d = 2)

## Validation

The integrator and the dispersion implementation were checked numerically:

- LA period at k = (2π·4/N, 0): measured 12.16, analytic 2π/ω_LA = 12.14
  (~0.2 % error from time-step discretization at dt = 0.04).
- Energy drift over ~3000 Verlet steps: ~10⁻⁴ relative.
- ω_LA / ω_TA = √(K_L/K_T) ≈ 1.826 as expected.
- `cv_quantum(T = 100)` returns 4605.9 k_B, vs Dulong–Petit limit
  2N² − 2 = 4606. At kT ∈ [0.05, 0.5] the ratio C_v / T² is approximately
  constant, confirming 2D Debye scaling. Below kT ≈ 0.02 the curve
  flattens because the discrete BZ has a finite minimum frequency
  ω_min ≈ 2π c / (N a).

## Defaults

Tunable as macros at the top of `crystal_phonons.c`:

| Symbol     | Value | Meaning                            |
|------------|-------|------------------------------------|
| `N`        | 48    | Lattice side (N × N atoms)         |
| `MASS`     | 1.0   | Atomic mass                        |
| `DT`       | 0.04  | Integrator step                    |
| `SUBSTEPS` | 3     | Physics substeps per video frame   |
| `WIN_W`    | 1340  | Window width                       |
| `WIN_H`    | 760   | Window height                      |
| `LAT_PX`   | 720   | Lattice region edge length in px   |

`g_KL`, `g_KT`, `g_kT_target`, `g_amp`, `g_kn` are runtime variables
exposed via the sliders.

## Extensions worth doing

- **Diatomic chain / two-atom basis.** Duplicate the lattice with a second
  sublattice and unequal masses; you get an *optical* branch with a gap at
  Γ — the standard textbook setup for IR-active modes and the Reststrahlen
  band.
- **NNN diagonal springs.** The current model has zero shear modulus in
  the long-wavelength limit when K_T → 0. Adding next-nearest-neighbour
  diagonal bonds lets you tune the Poisson ratio independently and gets
  closer to the elasticity of a real 2D crystal.
- **Langevin thermostat.** Replace the one-shot Maxwell–Boltzmann
  initialisation with γv + √(2γ kT) noise so you can quench, anneal, or
  hold a steady-state temperature gradient across the lattice — the
  latter visualises phonon thermal transport.
- **Real-time mode projection.** Every K steps, FFT the displacement
  field and plot |u(k)|² on top of the dispersion panel. Energy localises
  on the dispersion curves; for thermal initial conditions the occupation
  is Bose–Einstein at high T and degrades toward classical equipartition.
- **Quantum C_v(T) overlay panel** on log–log axes with the Debye T² and
  Dulong–Petit asymptotes drawn for comparison.

## Files

- `crystal_phonons.c` — single-file source, ~28 KB
- `Makefile` — release / debug / run / check-deps / clean
- `README.md` — this file
