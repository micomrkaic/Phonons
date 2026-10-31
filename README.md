# crystal_phonons

An interactive 2D lattice-dynamics playground in C/SDL2. An N×N square
lattice of atoms with nearest- and next-nearest-neighbour springs, FPU
anharmonicity, a two-atom basis, mass disorder, defects, a driven atom,
Langevin heat baths and four boundary conditions. The panel shows the
analytic dispersion with the **measured** S(k,ω) underneath, a live
Brillouin-zone map of mode energies, energy time series, the temperature
profile, and the quantum C_v(T).

Click the dispersion curve and a wave packet launches at that (k, branch);
click the lattice and it ripples; press F2–F10 for guided demos.

## Build and install

Requires SDL2 ≥ 2.0.18 (for `SDL_RenderGeometry`; older versions fall back
to slower drawing), SDL2_ttf, DejaVu fonts, and optionally ffmpeg with
libx264 for video capture.

```sh
sudo apt install libsdl2-dev libsdl2-ttf-dev fonts-dejavu ffmpeg   # Debian/Ubuntu
brew install sdl2 sdl2_ttf ffmpeg                                  # macOS (uses Arial/Menlo)
make
make install        # ~/.local/bin/crystal_phonons, ~/.local/share/man/man1/crystal_phonons.1
make uninstall
```

| Target            | What it does                                                 |
|-------------------|--------------------------------------------------------------|
| `make`            | Release build, `-O2 -Wall -Wextra -Wpedantic -Wshadow`       |
| `make debug`      | `-O0 -g3` with AddressSanitizer + UBSan                      |
| `make run`        | Build, then run                                              |
| `make selftest`   | Build, then run the numerical validation suite               |
| `make install`    | Binary to `$(PREFIX)/bin`, man page to `$(PREFIX)/share/man/man1`; `PREFIX` defaults to `~/.local` |
| `make uninstall`  | Remove both                                                  |
| `make wasm`       | Browser build with Emscripten (`emcc` on PATH); no video capture |
| `make check-deps` | Verify SDL2, SDL2_ttf, fonts, ffmpeg                         |
| `make clean`      | Remove build artifacts                                       |

`man` finds pages under `~/.local/share/man` automatically when
`~/.local/bin` is on your `PATH`. `PREFIX=/usr/local` and `DESTDIR=` work
as usual.

## Run

```sh
crystal_phonons                    # 64×64, starts with an LA wave packet
crystal_phonons -n 128 --demo fput
crystal_phonons --selftest
```

Options: `-n N` (power of two, 16–512), `-d/--demo NAME`, `-s SEED`,
`-o DIR` (videos and screenshots), `--shot FILE --frames K` (render a BMP
and exit), `--selftest`, `-h`, `-V`. Full reference: `man crystal_phonons`,
or F1 in the window.

## Window

| Region                 | Contents                                                       |
|------------------------|----------------------------------------------------------------|
| Left                   | Lattice: atoms (size ∝ m^¼), bonds coloured by strain, or a heat map |
| Top centre             | ω(k) along Γ–X–M–Γ; selected k with its group-velocity tangent; driver ω; S(k,ω) underlay |
| Top right              | Mode energy E(k) over the full Brillouin zone (2D FFT, log scale); click to pick k |
| Middle                 | Slider tabs: Lattice, Excite, Thermal, View                    |
| Bottom left            | E/KE/PE and E(k), E(2k), E(3k), E(4k) time series · T(x) profile · C_v(T) log–log |
| Bottom right           | Diagnostics: ΔE/E₀, kT_eff, C_v, v_g, heat currents, S(k,ω) progress, fps |

Colour fields (D): |u|, u_x, u_y, |v|, KE, div u, curl u, mass. Signed
fields use a diverging map, so div u isolates longitudinal motion and
curl u transverse motion. The default colour scale tracks the data slowly,
so a jump in temperature or amplitude stays visible for a second or two; C
locks it or switches to fixed physical scales.

## Demos

| Key | Name      | What to watch |
|-----|-----------|---------------|
| F2  | packet    | Crests move at v_p through an envelope moving at v_g; the crosshair is the predicted v_g |
| F3  | focusing  | Point pulse on an anisotropic lattice: crossed elliptical fronts, lattice caustics |
| F4  | fput      | α-FPU: energy in mode k spreads to 2k, 3k… and comes back |
| F5  | breathers | β-FPU zone-boundary mode breaks up into localized breathers |
| F6  | impurity  | Light atom's local mode lies above ω_max, so it keeps ringing alone |
| F7  | disorder  | Driven atom in a mass-disordered lattice; waves stay near the source |
| F8  | heat      | Harmonic crystal between baths: flat T(x). Raise β → gradient |
| F9  | diatomic  | m_B/m_A = 3: thermal S(k,ω) fills in optical branches and the gap |
| F10 | driven    | Driver radiates; right-click the dispersion plot above ω_max → it stops |

## Controls (short form)

| Key            | Action |
|----------------|--------|
| 1–4, Enter     | Excite branch at selected k (standing / travelling / packet, cycled with E; A = additive) |
| Arrows         | Move selected k |
| T, H           | Maxwell–Boltzmann at kT; bath off / global Langevin / hot–cold walls |
| B              | Boundary: periodic / fixed / free / absorbing |
| D, G, C        | Colour field; style (atoms / bonds / heat map); colour scale |
| S, Shift+S     | S(k,ω) underlay; clear |
| L              | Lower plot |
| Tab, P         | Mouse tool (poke, heavy, light, vacancy, driver, erase); driver polarisation |
| N, Shift+N     | Lattice 32 / 64 / 128 / 256 |
| R, Shift+R     | Reset motion; reset everything |
| V, I           | Record mp4; save BMP |
| Space, .       | Pause; single step |
| F1 / ?, Q      | Help; quit |

Mouse: left on lattice applies the tool (poke: click = radial pulse, drag
= push); right on lattice always pokes; left on the dispersion plot excites
the nearest branch at that k; right sets the driver frequency; the k-map
selects k (left also excites).

## Physics

Units: a = 1, m_A = 1, ℏ = k_B = 1.

**Model.** A bond along x carries
U = ½K_L s² + α s³/3 + β s⁴/4 + ½K_T t², with s = Δu_x (stretch) and
t = Δu_y (x and y swap for bonds along y). Diagonal bonds carry
½K_D (ê·Δu)², which couples the two polarisations; with K_T = 0 this is a
central-force lattice with finite shear stiffness. The checkerboard basis
puts mass m_B on sites with odd x + y.

**Dispersion.** The harmonic dynamical matrix in the e^{ik·r} convention
with true site positions is real, 2×2 (monatomic) or 4×4 (basis), and is
diagonalised by Jacobi rotations. Monatomic:

```
D_xx = 2K_L(1−cos k_x) + 2K_T(1−cos k_y) + K_D(2 − cos(k_x+k_y) − cos(k_x−k_y))
D_yy = 2K_T(1−cos k_x) + 2K_L(1−cos k_y) + K_D(2 − cos(k_x+k_y) − cos(k_x−k_y))
D_xy = K_D(cos(k_x−k_y) − cos(k_x+k_y))
```

Excitations use the eigenvectors, so a launched mode stays in its branch
(leakage < 10⁻¹⁴ in the self-test). Packets carry the matching velocity
field u̇ = ω A ê sin(k·r) and therefore travel; the start point is offset
against v_g so the packet crosses the screen.

**Integration.** Velocity Verlet, Δt = 0.04. Baths apply the exact
Ornstein–Uhlenbeck update v ← c v + √((1−c²)kT/m) ξ, c = e^{−γΔt}. The
absorbing boundary is a fixed wall plus a graded friction sponge.

**S(k,ω).** Every 6 steps, u(k) is computed along the path in O(N²)
(projection onto lines, then 1D FFTs) and stored. Every 128 samples a
Hann-windowed 512-point FFT in time is accumulated: Δω ≈ 0.051, Nyquist
≈ 13.1. Display is log-scaled and normalised per k column.

**Heat capacity.** Classical MD obeys equipartition (C_v = Dulong–Petit).
The quantum C_v(T) panel sums x²/(4 sinh²(x/2)), x = ω/T, over every normal
mode of the ideal crystal. In 2D, C_v ∝ T² at low T until the finite-lattice
gap ω_min ≈ 2πc/N cuts it off.

## Validation

`crystal_phonons --selftest` (no window):

```
PASS  eigensolver vs closed form           max |dw| = 8.88e-16
PASS  standing LA period                   measured 16.1031  analytic 16.1033  (-1.01e-05)
PASS  K_D mode k=(5,3) branch 1            w 0.46794 vs 0.46794, leak 5.2e-17
PASS  K_D mode k=(5,3) branch 2            w 0.71564 vs 0.71561, leak 2.1e-16
PASS  diatomic optical k=(3,1) b3          w 1.85517 vs 1.85475, leak 9.6e-15
PASS  diatomic optical k=(3,1) b4          w 1.86567 vs 1.86524, leak 9.8e-15
PASS  F = -grad PE (periodic BC)           max error 6.7e-10
PASS  F = -grad PE (fixed BC)              max error 3.9e-10
PASS  F = -grad PE (free BC)               max error 4.3e-10
PASS  F = -grad PE (absorbing BC)          max error 6.6e-10
PASS  energy conservation (anharmonic)     max |dE|/E = 8.4e-04 over 5000 steps
PASS  packet moves at v_g                  measured 0.9200  v_g 0.9239
PASS  Langevin kT_eff                      0.3002 vs 0.3000
PASS  C_v -> Dulong-Petit at high T        C/C_DP = 1.000000
PASS  C_v ~ T^2 at low T (2D Debye)        C(2T)/C(T) = 4.339
PASS  S(k,w) peaks on dispersion           worst offset 0.18 bins (dw = 0.0511)
```

The ~2×10⁻⁴ frequency offsets for the optical modes are the expected Verlet
error (ωΔt)²/24.

## Video capture

V pipes raw RGBA frames to `ffmpeg -c:v libx264 -preset veryfast` and writes
`phonons_YYYYMMDD_HHMMSS.mp4` to the output directory. Frames are read back
**before** `SDL_RenderPresent` (the back buffer is undefined after a
present, which gave black or garbled frames on some drivers in v1).
Recording stops if the window is resized.

## Files

- `crystal_phonons.c` — single-file source
- `crystal_phonons.1` — man page
- `Makefile` — build / debug / selftest / install / uninstall / wasm / check-deps / clean
- `README.md` — this file
