/*
 * crystal_phonons.c -- 2D lattice-dynamics playground (SDL2 + SDL_ttf).
 *
 * An N x N square lattice of atoms with nearest-neighbour (K_L, K_T) and
 * optional diagonal next-nearest-neighbour (K_D) springs, FPU alpha/beta
 * anharmonicity on the nearest-neighbour bonds, a checkerboard two-atom
 * basis, mass disorder, point defects, Langevin baths, a driven atom and
 * four boundary conditions.
 *
 * The right-hand panel shows the analytic dispersion with the measured
 * dynamic structure factor S(k, omega) underneath, a live Brillouin-zone
 * map of mode energies, tabbed parameter sliders, a lower plot (energy
 * time series / temperature profile / quantum C_v(T)) and diagnostics.
 *
 * Single file.  See README.md and crystal_phonons(1) for usage and physics.
 * `crystal_phonons --selftest` runs the numerical validation suite.
 */

#define _POSIX_C_SOURCE 200809L
#ifdef __APPLE__
#  define _DARWIN_C_SOURCE 1
#endif

#include <SDL.h>
#include <SDL_ttf.h>
#ifdef __EMSCRIPTEN__
#  include <emscripten.h>
#endif

#include <limits.h>
#include <math.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

#define PHONONS_VERSION "2.0"

#if SDL_VERSION_ATLEAST(2, 0, 18)
#  define HAVE_GEOMETRY 1
#else
#  define HAVE_GEOMETRY 0
#endif

/* ===================================================================
 * Constants
 * =================================================================== */

#define DT        0.04       /* Verlet time step                          */
#define S_STRIDE  6          /* Verlet steps between S(k,w) samples       */
#define SW_LEN    512        /* S(k,w) time window in samples (2^m)       */
#define TS_LEN    900        /* time-series ring length (frames)          */
#define DISP_PTS  360        /* points per analytic dispersion curve      */
#define NMIN      16
#define NMAX      512

/* layout, in logical pixels (window scales via SDL logical size) */
#define WIN_W    1600
#define WIN_H    900
#define LAT_X    20
#define LAT_Y    20
#define LAT_PX   860
#define PANEL_X  900
#define PANEL_R  1580
#define DISP_X   948
#define DISP_Y   44
#define DISP_W   430
#define DISP_H   220
#define BZ_X     1400
#define BZ_Y     44
#define BZ_S     180
#define TAB_Y    306
#define TAB_W    100
#define TAB_H    24
#define SL_X     (PANEL_X + 20)
#define SL_W     600
#define SL_Y0    364
#define SL_DY    36
#define SL_H     14
#define LOW_X    948
#define LOW_Y    640
#define LOW_W    380
#define LOW_H    170
#define DIAG_X   1365
#define DIAG_Y   612
#define LINE_DY  16
#define STAT_Y   850

enum { BC_PERIODIC, BC_FIXED, BC_FREE, BC_ABSORB, BC_COUNT };
static const char *bc_names[] = { "periodic", "fixed", "free", "absorbing" };

enum { TH_OFF, TH_GLOBAL, TH_WALLS, TH_COUNT };
static const char *th_names[] = { "off", "global Langevin", "hot/cold walls" };

enum { SH_STANDING, SH_TRAVEL, SH_PACKET, SH_COUNT };
static const char *sh_names[] = { "standing wave", "travelling wave", "wave packet" };

enum { F_ABSU, F_UX, F_UY, F_ABSV, F_KE, F_DIV, F_CURL, F_MASS, F_COUNT };
static const char *field_names[] = { "|u|", "u_x", "u_y", "|v|", "KE",
                                     "div u", "curl u", "mass" };
static const bool   field_signed[] = { false, true, true, false, false,
                                       true, true, true };
static const double field_ref[]    = { 0.5, 0.5, 0.5, 0.5, 0.25,
                                       0.25, 0.25, 2.0, /* bonds: */ 0.15 };
#define SLOT_BOND F_COUNT

enum { ST_ATOMS, ST_BONDS, ST_HEAT, ST_COUNT };
static const char *style_names[] = { "atoms", "bonds", "heatmap" };

enum { CS_AUTO, CS_LOCK, CS_FIXED, CS_COUNT };
static const char *cs_names[] = { "auto", "locked", "fixed" };

enum { TOOL_POKE, TOOL_HEAVY, TOOL_LIGHT, TOOL_VACANCY, TOOL_DRIVER,
       TOOL_ERASE, TOOL_COUNT };
static const char *tool_names[] = { "poke", "heavy atom (x4)",
                                    "light atom (x1/4)", "vacancy",
                                    "driver", "erase defect" };

enum { LOW_TS, LOW_PROFILE, LOW_CV, LOW_COUNT };
static const char *low_names[] = { "Energy time series",
                                   "Temperature profile T(x)",
                                   "Quantum C_v(T), log-log" };

enum { TAB_LATTICE, TAB_EXCITE, TAB_THERMAL, TAB_VIEW, TAB_COUNT };
static const char *tab_names[] = { "Lattice", "Excite", "Thermal", "View" };

/* ===================================================================
 * Parameters (slider-backed)
 * =================================================================== */

typedef struct {
    double KL, KT, KD, alpha, beta, mratio, disorder;   /* lattice   */
    double amp, pk_sigma, poke_sigma, drive_w, drive_F; /* excite    */
    double kT, kT_hot, kT_cold, gamma;                  /* thermal   */
    double exag, gain, speed;                           /* view      */
} Params;

static const Params prm_default = {
    1.0, 0.30, 0.0, 0.0, 0.0, 1.0, 0.0,
    0.25, 6.0, 1.5, 1.0, 0.3,
    0.05, 1.0, 0.1, 0.3,
    1.0, 1.0, 4.0
};
static Params prm;
static Params prm_last;

/* ===================================================================
 * Simulation state
 * =================================================================== */

static int N  = 64;          /* lattice side (power of two)              */
static int P  = 66;          /* padded stride = N + 2 (ghost layer)      */
static int NN = 64 * 64;

static double *ux, *uy, *vx, *vy, *fx, *fy;   /* padded P*P arrays     */
static double *ms;           /* effective mass per site                 */
static double *rnd;          /* frozen uniform(-1,1) for mass disorder  */
static double *defm;         /* defect mass multiplier (1 = none)       */
static double *sponge;       /* absorbing-boundary friction per site    */
static unsigned char *alive; /* 0 = vacancy (or ghost under free BC)    */

static int    bc        = BC_PERIODIC;
static int    thermo    = TH_OFF;
static int    shape     = SH_PACKET;
static bool   additive  = false;
static bool   paused    = false;
static bool   step_once = false;
static int    driver_I  = -1;
static int    drive_pol = 0;          /* 0 = x, 1 = y */
static long   step_no   = 0;
static double sim_time  = 0.0;
static int    sel_nx = 4, sel_ny = 0, sel_b = 1;

static double heat_in = 0.0, heat_out = 0.0;   /* bath work, this frame */
static double J_in = 0.0, J_out = 0.0;         /* EMA heat currents     */

static double E_ref = 0.0;
static bool   E_ref_set = false;

static bool   pk_track = false;                /* packet v_g tracker    */
static double pk_x0, pk_y0, pk_vgx, pk_vgy, pk_t0;

static bool   headless = false;                /* --selftest            */

/* diagnostics */
static double d_KE = 0, d_PE = 0, d_kTeff = 0, d_maxu = 0;
static int    d_nalive = 0;
static double d_cv = 0, d_cv_T = -1.0;

/* ===================================================================
 * SDL / UI state
 * =================================================================== */

static SDL_Window   *win  = NULL;
static SDL_Renderer *ren  = NULL;
static TTF_Font *font_title = NULL, *font_lbl = NULL, *font_mono = NULL;
static SDL_Texture *heat_tex = NULL, *bz_tex = NULL, *s_tex = NULL;
static int heat_tex_n = 0, bz_tex_n = 0, s_tex_w = 0, s_tex_h = 0;

static bool  running    = true;
static int   field      = F_ABSU;
static int   style      = ST_ATOMS;
static int   tool       = TOOL_POKE;
static int   low_mode   = LOW_TS;
static int   cur_tab    = TAB_LATTICE;
static int   cscale     = CS_AUTO;
static bool  show_help  = false;
static bool  s_show     = false;
static char  toast_msg[320] = "";
static Uint32 toast_until = 0;
static Uint32 now_ms = 0;
static unsigned frame_no = 0;
static double fps = 0.0;
static const char *out_dir = ".";
static const char *shot_file = NULL;
static int   shot_frames = 120;

static double cs_ema[F_COUNT + 1];
static double cs_lock[F_COUNT + 1];

/* mouse */
static int    drag_slider = -1;
static int    lat_drag    = 0;     /* 0 none, 1 poke-drag, 2 paint-drag */
static int    last_site   = -1;
static double last_cx = 0, last_cy = 0;

/* video */
static FILE    *video_pipe = NULL;
static uint8_t *video_buf  = NULL;
static bool     recording  = false;
static int      rec_w = 0, rec_h = 0, rec_out_w = 0, rec_out_h = 0;
static char     video_name[512] = "";

/* colours */
static const SDL_Color COL_BG     = {  16,  16,  20, 255 };
static const SDL_Color COL_FG     = { 220, 220, 222, 255 };
static const SDL_Color COL_DIM    = { 135, 135, 145, 255 };
static const SDL_Color COL_TITLE  = { 242, 242, 242, 255 };
static const SDL_Color COL_KEY    = { 110, 200, 240, 255 };
static const SDL_Color COL_REC    = { 230,  60,  60, 255 };
static const SDL_Color COL_MARKER = { 240, 220,  90, 255 };
static const SDL_Color COL_HOT    = { 255,  95,  70, 255 };
static const SDL_Color COL_COLD   = {  80, 150, 255, 255 };
static const SDL_Color COL_BRANCH[4] = {
    { 240, 130,  60, 255 }, {  80, 180, 230, 255 },
    { 120, 210, 110, 255 }, { 220, 110, 220, 255 }
};
static const SDL_Color COL_MODE[4] = {
    { 250, 210,  80, 255 }, { 110, 220, 140, 255 },
    { 120, 170, 255, 255 }, { 230, 120, 220, 255 }
};

static SDL_Color lut_seq[256], lut_div[256], lut_s[256];

/* ===================================================================
 * Small utilities
 * =================================================================== */

static void *xcalloc(size_t n, size_t sz)
{
    void *p = calloc(n ? n : 1, sz);
    if (!p) { fprintf(stderr, "crystal_phonons: out of memory\n"); exit(1); }
    return p;
}

static void *xrealloc(void *p, size_t sz)
{
    void *q = realloc(p, sz ? sz : 1);
    if (!q) { fprintf(stderr, "crystal_phonons: out of memory\n"); exit(1); }
    return q;
}

static double clampd(double v, double lo, double hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

static int imod(int a, int n) { int r = a % n; return r < 0 ? r + n : r; }

static inline int  ID(int x, int y)      { return (y + 1) * P + (x + 1); }
static inline bool is_real(int x, int y) { return x >= 0 && x < N && y >= 0 && y < N; }

static void toast(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(toast_msg, sizeof toast_msg, fmt, ap);
    va_end(ap);
    toast_until = now_ms + 5000;
}

/* ----- RNG: xorshift128+ seeded by splitmix64 ----- */
static uint64_t rng_s[2] = { 0x9E3779B97F4A7C15ull, 0xD1B54A32D192ED03ull };

static uint64_t splitmix64(uint64_t *x)
{
    uint64_t z = (*x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static void rng_seed(uint64_t s)
{
    rng_s[0] = splitmix64(&s);
    rng_s[1] = splitmix64(&s);
}

static uint64_t rng_next(void)
{
    uint64_t s1 = rng_s[0];
    const uint64_t s0 = rng_s[1];
    const uint64_t r = s0 + s1;
    rng_s[0] = s0;
    s1 ^= s1 << 23;
    rng_s[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
    return r;
}

static double rng_u01(void)
{
    return (double)(rng_next() >> 11) * (1.0 / 9007199254740992.0);
}

static double randn(void)
{
    static bool   has_spare = false;
    static double spare = 0.0;
    if (has_spare) { has_spare = false; return spare; }
    double u1;
    do u1 = rng_u01(); while (u1 <= 1e-300);
    const double u2 = rng_u01();
    const double r  = sqrt(-2.0 * log(u1));
    spare = r * sin(2.0 * M_PI * u2);
    has_spare = true;
    return r * cos(2.0 * M_PI * u2);
}

/* ===================================================================
 * FFT (radix-2, complex, in place, forward = e^{-i})
 * =================================================================== */

typedef struct { int n; int *rev; double *cs, *sn; } FFTPlan;

static FFTPlan plan_N = { 0, NULL, NULL, NULL };
static FFTPlan plan_W = { 0, NULL, NULL, NULL };

static void fft_free(FFTPlan *p)
{
    free(p->rev); free(p->cs); free(p->sn);
    p->rev = NULL; p->cs = p->sn = NULL; p->n = 0;
}

static void fft_init(FFTPlan *p, int n)
{
    fft_free(p);
    p->n   = n;
    p->rev = xcalloc((size_t)n, sizeof *p->rev);
    p->cs  = xcalloc((size_t)(n / 2 + 1), sizeof *p->cs);
    p->sn  = xcalloc((size_t)(n / 2 + 1), sizeof *p->sn);
    int lg = 0;
    while ((1 << lg) < n) ++lg;
    for (int i = 0; i < n; ++i) {
        int r = 0;
        for (int b = 0; b < lg; ++b) if (i & (1 << b)) r |= 1 << (lg - 1 - b);
        p->rev[i] = r;
    }
    for (int k = 0; k <= n / 2; ++k) {
        p->cs[k] =  cos(2.0 * M_PI * k / n);
        p->sn[k] = -sin(2.0 * M_PI * k / n);
    }
}

static void fft_run(const FFTPlan *p, double *re, double *im)
{
    const int n = p->n;
    for (int i = 0; i < n; ++i) {
        const int j = p->rev[i];
        if (i < j) {
            double t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        const int half = len >> 1, step = n / len;
        for (int i = 0; i < n; i += len)
            for (int k = 0; k < half; ++k) {
                const double wr = p->cs[k * step], wi = p->sn[k * step];
                const int a = i + k, b = a + half;
                const double tr = re[b] * wr - im[b] * wi;
                const double ti = re[b] * wi + im[b] * wr;
                re[b] = re[a] - tr; im[b] = im[a] - ti;
                re[a] += tr;        im[a] += ti;
            }
    }
}

static double *col_re = NULL, *col_im = NULL;

static void fft2d(double *re, double *im)
{
    for (int r = 0; r < N; ++r) fft_run(&plan_N, re + (size_t)r * N, im + (size_t)r * N);
    for (int c = 0; c < N; ++c) {
        for (int r = 0; r < N; ++r) { col_re[r] = re[r * N + c]; col_im[r] = im[r * N + c]; }
        fft_run(&plan_N, col_re, col_im);
        for (int r = 0; r < N; ++r) { re[r * N + c] = col_re[r]; im[r * N + c] = col_im[r]; }
    }
}

/* ===================================================================
 * Lattice allocation, ghosts, masses, sponge
 * =================================================================== */

static void free_lattice(void)
{
    free(ux); free(uy); free(vx); free(vy); free(fx); free(fy);
    free(ms); free(rnd); free(defm); free(sponge); free(alive);
    ux = uy = vx = vy = fx = fy = ms = rnd = defm = sponge = NULL;
    alive = NULL;
}

static void alloc_lattice(int n)
{
    free_lattice();
    N = n; P = n + 2; NN = n * n;
    const size_t s = (size_t)P * (size_t)P;
    ux = xcalloc(s, sizeof *ux); uy = xcalloc(s, sizeof *uy);
    vx = xcalloc(s, sizeof *vx); vy = xcalloc(s, sizeof *vy);
    fx = xcalloc(s, sizeof *fx); fy = xcalloc(s, sizeof *fy);
    ms = xcalloc(s, sizeof *ms); rnd = xcalloc(s, sizeof *rnd);
    defm = xcalloc(s, sizeof *defm); sponge = xcalloc(s, sizeof *sponge);
    alive = xcalloc(s, 1);
    for (size_t i = 0; i < s; ++i) { defm[i] = 1.0; ms[i] = 1.0; }
    for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
            alive[ID(x, y)] = 1;
            rnd[ID(x, y)] = 2.0 * rng_u01() - 1.0;
        }
}

static void fill_ghosts(double *a)
{
    if (bc == BC_PERIODIC) {
        for (int y = 0; y < N; ++y) {
            a[ID(-1, y)] = a[ID(N - 1, y)];
            a[ID(N,  y)] = a[ID(0, y)];
        }
        memcpy(&a[ID(-1, -1)], &a[ID(-1, N - 1)], (size_t)P * sizeof *a);
        memcpy(&a[ID(-1, N)],  &a[ID(-1, 0)],     (size_t)P * sizeof *a);
    } else {
        for (int y = -1; y <= N; ++y) { a[ID(-1, y)] = 0.0; a[ID(N, y)] = 0.0; }
        for (int x = -1; x <= N; ++x) { a[ID(x, -1)] = 0.0; a[ID(x, N)] = 0.0; }
    }
}

static void fill_alive_ghosts(void)
{
    if (bc == BC_PERIODIC) {
        for (int y = 0; y < N; ++y) {
            alive[ID(-1, y)] = alive[ID(N - 1, y)];
            alive[ID(N,  y)] = alive[ID(0, y)];
        }
        memcpy(&alive[ID(-1, -1)], &alive[ID(-1, N - 1)], (size_t)P);
        memcpy(&alive[ID(-1, N)],  &alive[ID(-1, 0)],     (size_t)P);
    } else {
        const unsigned char v = (bc == BC_FREE) ? 0 : 1;
        for (int y = -1; y <= N; ++y) { alive[ID(-1, y)] = v; alive[ID(N, y)] = v; }
        for (int x = -1; x <= N; ++x) { alive[ID(x, -1)] = v; alive[ID(x, N)] = v; }
    }
}

static double nominal_mass(int x, int y)
{
    return ((x + y) & 1) ? prm.mratio : 1.0;
}

static void update_masses(void)
{
    for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
            const int I = ID(x, y);
            ms[I] = nominal_mass(x, y) * (1.0 + prm.disorder * rnd[I]) * defm[I];
        }
}

static int sponge_width(void) { int w = N / 8; return w < 4 ? 4 : w; }

static void update_sponge(void)
{
    memset(sponge, 0, (size_t)P * (size_t)P * sizeof *sponge);
    if (bc != BC_ABSORB) return;
    const int w = sponge_width();
    for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
            int d = x;
            if (y < d) d = y;
            if (N - 1 - x < d) d = N - 1 - x;
            if (N - 1 - y < d) d = N - 1 - y;
            if (d < w) {
                const double t = (double)(w - d) / (double)w;
                sponge[ID(x, y)] = 0.8 * t * t;
            }
        }
}

static int bath_width(void) { int w = N / 16; return w < 2 ? 2 : w; }

/* 0 = no bath, 1 = hot, 2 = cold.  With periodic BCs the cold strip sits
 * at N/2 so the two reservoirs are not adjacent across the seam. */
static int bath_of_col(int x)
{
    const int w = bath_width();
    if (x < w) return 1;
    const int c0 = (bc == BC_PERIODIC) ? N / 2 : N - w;
    if (x >= c0 && x < c0 + w) return 2;
    return 0;
}

/* ===================================================================
 * Forces, integrator, energy
 * =================================================================== */

static void compute_forces(void)
{
    fill_ghosts(ux);
    fill_ghosts(uy);
    const double KL = prm.KL, KT = prm.KT, KD2 = 0.5 * prm.KD;
    const double a = prm.alpha, b = prm.beta;
    const bool diag = prm.KD > 0.0;

    for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
            const int I = ID(x, y);
            if (!alive[I]) { fx[I] = fy[I] = 0.0; continue; }
            const double uxi = ux[I], uyi = uy[I];
            double ax = 0.0, ay = 0.0;
            int J;

            J = I + 1;                     /* +x: s = stretch of bond   */
            if (alive[J]) {
                const double s = ux[J] - uxi;
                ax += s * (KL + s * (a + b * s));
                ay += KT * (uy[J] - uyi);
            }
            J = I - 1;                     /* -x                         */
            if (alive[J]) {
                const double s = uxi - ux[J];
                ax -= s * (KL + s * (a + b * s));
                ay += KT * (uy[J] - uyi);
            }
            J = I + P;                     /* +y                         */
            if (alive[J]) {
                const double s = uy[J] - uyi;
                ay += s * (KL + s * (a + b * s));
                ax += KT * (ux[J] - uxi);
            }
            J = I - P;                     /* -y                         */
            if (alive[J]) {
                const double s = uyi - uy[J];
                ay -= s * (KL + s * (a + b * s));
                ax += KT * (ux[J] - uxi);
            }
            if (diag) {
                /* (1,1) diagonals: force = (K_D/2)(dx+dy)(1,1)          */
                J = I + P + 1;
                if (alive[J]) {
                    const double t = KD2 * ((ux[J] - uxi) + (uy[J] - uyi));
                    ax += t; ay += t;
                }
                J = I - P - 1;
                if (alive[J]) {
                    const double t = KD2 * ((ux[J] - uxi) + (uy[J] - uyi));
                    ax += t; ay += t;
                }
                /* (1,-1) diagonals: force = (K_D/2)(dx-dy)(1,-1)        */
                J = I - P + 1;
                if (alive[J]) {
                    const double t = KD2 * ((ux[J] - uxi) - (uy[J] - uyi));
                    ax += t; ay -= t;
                }
                J = I + P - 1;
                if (alive[J]) {
                    const double t = KD2 * ((ux[J] - uxi) - (uy[J] - uyi));
                    ax += t; ay -= t;
                }
            }
            fx[I] = ax; fy[I] = ay;
        }

    if (driver_I >= 0 && alive[driver_I] && prm.drive_F != 0.0) {
        const double F = prm.drive_F * sin(prm.drive_w * sim_time);
        if (drive_pol == 0) fx[driver_I] += F; else fy[driver_I] += F;
    }
}

static void langevin_step(void)
{
    const bool walls = thermo == TH_WALLS, glob = thermo == TH_GLOBAL;
    const bool spg = bc == BC_ABSORB;
    if (!walls && !glob && !spg) return;
    const double g0 = prm.gamma;
    const double c0 = exp(-g0 * DT);
    for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
            const int I = ID(x, y);
            if (!alive[I]) continue;
            double g = 0.0, gT = 0.0;
            int bath = 0;
            if (glob) { g = g0; gT = g0 * prm.kT; }
            else if (walls && (bath = bath_of_col(x)) != 0) {
                g = g0; gT = g0 * (bath == 1 ? prm.kT_hot : prm.kT_cold);
            }
            const double gs = spg ? sponge[I] : 0.0;
            if (g + gs <= 0.0) continue;
            const double gt = g + gs;
            const double T  = gT / gt;
            const double c  = (gs == 0.0) ? c0 : exp(-gt * DT);
            const double m  = ms[I];
            const double sd = sqrt((1.0 - c * c) * T / m);
            const double e0 = 0.5 * m * (vx[I] * vx[I] + vy[I] * vy[I]);
            if (T > 0.0) {
                vx[I] = c * vx[I] + sd * randn();
                vy[I] = c * vy[I] + sd * randn();
            } else {
                vx[I] *= c; vy[I] *= c;
            }
            if (bath) {
                const double e1 = 0.5 * m * (vx[I] * vx[I] + vy[I] * vy[I]);
                if (bath == 1) heat_in += e1 - e0; else heat_out += e1 - e0;
            }
        }
}

static void verlet_step(void)
{
    const double h = 0.5 * DT;
    for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
            const int I = ID(x, y);
            if (!alive[I]) continue;
            const double im = 1.0 / ms[I];
            vx[I] += h * fx[I] * im;
            vy[I] += h * fy[I] * im;
            ux[I] += DT * vx[I];
            uy[I] += DT * vy[I];
        }
    sim_time += DT;
    compute_forces();
    for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
            const int I = ID(x, y);
            if (!alive[I]) continue;
            const double im = 1.0 / ms[I];
            vx[I] += h * fx[I] * im;
            vy[I] += h * fy[I] * im;
        }
    langevin_step();
    ++step_no;
}

static double bond_pe_nn(double s, double t, double KL, double KT)
{
    return s * s * (0.5 * KL + s * (prm.alpha / 3.0 + 0.25 * prm.beta * s))
         + 0.5 * KT * t * t;
}

static void compute_energy(double *KE, double *PE)
{
    fill_ghosts(ux);
    fill_ghosts(uy);
    double ke = 0.0, pe = 0.0;
    for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
            const int I = ID(x, y);
            if (alive[I])
                ke += 0.5 * ms[I] * (vx[I] * vx[I] + vy[I] * vy[I]);
        }
    static const int off[4][2] = { {1, 0}, {0, 1}, {1, 1}, {1, -1} };
    const double KL = prm.KL, KT = prm.KT, KD4 = 0.25 * prm.KD;
    for (int sy = -1; sy <= N; ++sy)
        for (int sx = -1; sx <= N; ++sx) {
            const int S = ID(sx, sy);
            if (!alive[S]) continue;
            const bool sreal = is_real(sx, sy);
            for (int o = 0; o < 4; ++o) {
                const int ex = sx + off[o][0], ey = sy + off[o][1];
                if (ex < -1 || ex > N || ey < -1 || ey > N) continue;
                const bool ereal = is_real(ex, ey);
                const bool count = (bc == BC_PERIODIC) ? sreal : (sreal || ereal);
                if (!count) continue;
                const int E = ID(ex, ey);
                if (!alive[E]) continue;
                const double dx = ux[E] - ux[S], dy = uy[E] - uy[S];
                switch (o) {
                case 0: pe += bond_pe_nn(dx, dy, KL, KT); break;
                case 1: pe += bond_pe_nn(dy, dx, KL, KT); break;
                case 2: pe += KD4 * (dx + dy) * (dx + dy); break;
                default: pe += KD4 * (dx - dy) * (dx - dy); break;
                }
            }
        }
    *KE = ke; *PE = pe;
}

/* ===================================================================
 * Analytic phonons: dynamical matrix and eigenproblem
 * =================================================================== */

static bool is_diatomic(void) { return fabs(prm.mratio - 1.0) > 1e-9; }
static int  n_branches(void)  { return is_diatomic() ? 4 : 2; }

/* Monatomic force-constant matrix (not mass-weighted). */
static void force_const_mono(double kx, double ky,
                             double *dxx, double *dyy, double *dxy)
{
    const double cx = cos(kx), cy = cos(ky);
    const double cp = cos(kx + ky), cm = cos(kx - ky);
    *dxx = 2.0 * prm.KL * (1.0 - cx) + 2.0 * prm.KT * (1.0 - cy) + prm.KD * (2.0 - cp - cm);
    *dyy = 2.0 * prm.KT * (1.0 - cx) + 2.0 * prm.KL * (1.0 - cy) + prm.KD * (2.0 - cp - cm);
    *dxy = prm.KD * (cm - cp);
}

/* Cyclic Jacobi for a real symmetric n x n (n <= 4) matrix.  A is
 * destroyed; eigenvalues in w, eigenvectors in the columns of V. */
static void jacobi_eig(int n, double A[4][4], double w[4], double V[4][4])
{
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) V[i][j] = (i == j) ? 1.0 : 0.0;
    for (int sweep = 0; sweep < 60; ++sweep) {
        double off = 0.0, diag = 0.0;
        for (int p = 0; p < n; ++p) {
            diag += fabs(A[p][p]);
            for (int q = p + 1; q < n; ++q) off += fabs(A[p][q]);
        }
        if (off <= 1e-15 * (diag + 1e-300)) break;
        for (int p = 0; p < n; ++p)
            for (int q = p + 1; q < n; ++q) {
                if (fabs(A[p][q]) < 1e-300) continue;
                const double th = (A[q][q] - A[p][p]) / (2.0 * A[p][q]);
                const double t  = (th >= 0 ? 1.0 : -1.0) / (fabs(th) + sqrt(th * th + 1.0));
                const double c  = 1.0 / sqrt(t * t + 1.0), s = t * c;
                for (int k = 0; k < n; ++k) {
                    const double akp = A[k][p], akq = A[k][q];
                    A[k][p] = c * akp - s * akq;
                    A[k][q] = s * akp + c * akq;
                }
                for (int k = 0; k < n; ++k) {
                    const double apk = A[p][k], aqk = A[q][k];
                    A[p][k] = c * apk - s * aqk;
                    A[q][k] = s * apk + c * aqk;
                }
                for (int k = 0; k < n; ++k) {
                    const double vkp = V[k][p], vkq = V[k][q];
                    V[k][p] = c * vkp - s * vkq;
                    V[k][q] = s * vkp + c * vkq;
                }
            }
    }
    for (int i = 0; i < n; ++i) w[i] = A[i][i];
}

/* Frequencies (ascending) and mass-weighted eigenvectors at k.
 * Components: monatomic (x, y); diatomic (A_x, A_y, B_x, B_y), A = even
 * (x+y) sites with mass 1, B = odd sites with mass m_B/m_A.  Uses the
 * e^{i k.r} convention with true site positions, so D is real. */
static int phonons_at(double kx, double ky, double w[4], double ev[4][4])
{
    double A[4][4] = { { 0 } }, V[4][4], lam[4];
    int n;
    if (!is_diatomic()) {
        n = 2;
        double dxx, dyy, dxy;
        force_const_mono(kx, ky, &dxx, &dyy, &dxy);
        A[0][0] = dxx; A[1][1] = dyy; A[0][1] = A[1][0] = dxy;
    } else {
        n = 4;
        const double mA = 1.0, mB = prm.mratio;
        const double cx = cos(kx), cy = cos(ky);
        const double cp = cos(kx + ky), cm = cos(kx - ky);
        const double S0 = 2.0 * (prm.KL + prm.KT + prm.KD);
        const double Sd = S0 - prm.KD * (cp + cm);
        const double Sxy = -prm.KD * (cp - cm);
        const double Cxx = -(2.0 * prm.KL * cx + 2.0 * prm.KT * cy);
        const double Cyy = -(2.0 * prm.KT * cx + 2.0 * prm.KL * cy);
        const double r = 1.0 / sqrt(mA * mB);
        A[0][0] = Sd / mA;  A[1][1] = Sd / mA;  A[0][1] = A[1][0] = Sxy / mA;
        A[2][2] = Sd / mB;  A[3][3] = Sd / mB;  A[2][3] = A[3][2] = Sxy / mB;
        A[0][2] = A[2][0] = Cxx * r;
        A[1][3] = A[3][1] = Cyy * r;
    }
    jacobi_eig(n, A, lam, V);
    int idx[4] = { 0, 1, 2, 3 };
    for (int i = 1; i < n; ++i)
        for (int j = i; j > 0 && lam[idx[j]] < lam[idx[j - 1]]; --j) {
            const int t = idx[j]; idx[j] = idx[j - 1]; idx[j - 1] = t;
        }
    for (int b = 0; b < n; ++b) {
        w[b] = sqrt(lam[idx[b]] > 0.0 ? lam[idx[b]] : 0.0);
        for (int c = 0; c < n; ++c) ev[b][c] = V[c][idx[b]];
    }
    return n;
}

static double omega_at(double kx, double ky, int b)
{
    double w[4], ev[4][4];
    const int nb = phonons_at(kx, ky, w, ev);
    return w[b < nb ? b : nb - 1];
}

static void group_velocity(double kx, double ky, int b, double *vgx, double *vgy)
{
    const double h = 1e-4;
    *vgx = (omega_at(kx + h, ky, b) - omega_at(kx - h, ky, b)) / (2.0 * h);
    *vgy = (omega_at(kx, ky + h, b) - omega_at(kx, ky - h, b)) / (2.0 * h);
}

/* Path Gamma(0,0) -> X(pi,0) -> M(pi,pi) -> Gamma, parameter s in [0,3]. */
static void path_k(double s, double *kx, double *ky)
{
    if (s < 1.0)      { *kx = M_PI * s; *ky = 0.0; }
    else if (s < 2.0) { *kx = M_PI;     *ky = M_PI * (s - 1.0); }
    else              { const double u = 1.0 - (s - 2.0); *kx = M_PI * u; *ky = M_PI * u; }
}

static void path_snap(double s, int *nx, int *ny)
{
    const int half = N / 2;
    s = clampd(s, 0.0, 3.0);
    if (s < 1.0)      { *nx = (int)lround(s * half); *ny = 0; }
    else if (s < 2.0) { *nx = half; *ny = (int)lround((s - 1.0) * half); }
    else              { const int n = (int)lround((3.0 - s) * half); *nx = n; *ny = n; }
}

/* Position of k (mapped into the irreducible wedge) on the path. */
static bool k_path_pos(int nx, int ny, double *s)
{
    const int half = N / 2;
    int a = imod(nx, N), b = imod(ny, N);
    if (a > half) a = N - a;
    if (b > half) b = N - b;
    if (b > a) { const int t = a; a = b; b = t; }
    if (b == 0)    { *s = (double)a / half; return true; }
    if (a == half) { *s = 1.0 + (double)b / half; return true; }
    if (a == b)    { *s = 2.0 + (double)(half - a) / half; return true; }
    return false;
}

static int signed_n(int n) { n = imod(n, N); return n > N / 2 ? n - N : n; }

/* ===================================================================
 * Dispersion cache, C_v cache
 * =================================================================== */

static double disp_w[4][DISP_PTS];
static int    disp_nb = 2;
static double omega_top = 1.0, omega_plot = 1.0;
static bool   disp_dirty = true;

static void disp_update(void)
{
    disp_nb = n_branches();
    double wmax = 1e-6;
    for (int p = 0; p < DISP_PTS; ++p) {
        const double s = 3.0 * p / (double)(DISP_PTS - 1);
        double kx, ky, w[4], ev[4][4];
        path_k(s, &kx, &ky);
        phonons_at(kx, ky, w, ev);
        for (int b = 0; b < disp_nb; ++b) {
            disp_w[b][p] = w[b];
            if (w[b] > wmax) wmax = w[b];
        }
    }
    omega_top  = wmax;
    omega_plot = 1.08 * wmax;
    if (driver_I >= 0 && 1.05 * prm.drive_w > omega_plot) omega_plot = 1.05 * prm.drive_w;
    disp_dirty = false;
}

static double *cv_w = NULL;
static int     cv_nw = 0;
static double  cv_weight = 1.0, cv_ndp = 1.0;
static bool    cv_dirty = true;
static Uint32  cv_dirty_since = 0;
#define CV_PTS 72
static double  cv_T[CV_PTS], cv_C[CV_PTS];

static double cv_at(double T)
{
    if (T <= 0.0 || !cv_w) return 0.0;
    double C = 0.0;
    for (int i = 0; i < cv_nw; ++i) {
        const double x = cv_w[i] / T;
        if (x > 700.0) continue;
        if (x < 1e-8) { C += 1.0; continue; }
        const double sh = sinh(0.5 * x);
        C += x * x / (4.0 * sh * sh);
    }
    return C * cv_weight;
}

/* All normal-mode frequencies of the ideal (defect-free) crystal; the
 * k = 0 acoustic zero modes are dropped. */
static void cv_rebuild(void)
{
    const int nb = n_branches();
    cv_w = xrealloc(cv_w, (size_t)NN * (size_t)nb * sizeof *cv_w);
    cv_nw = 0;
    cv_weight = (nb == 4) ? 0.5 : 1.0;   /* extended zone counts twice */
    for (int ny = 0; ny < N; ++ny)
        for (int nx = 0; nx < N; ++nx) {
            const double kx = 2.0 * M_PI * nx / N, ky = 2.0 * M_PI * ny / N;
            double w[4], ev[4][4];
            if (nb == 2) {
                double dxx, dyy, dxy;
                force_const_mono(kx, ky, &dxx, &dyy, &dxy);
                const double tr = 0.5 * (dxx + dyy);
                const double dd = sqrt(0.25 * (dxx - dyy) * (dxx - dyy) + dxy * dxy);
                w[0] = sqrt(fmax(0.0, tr - dd));
                w[1] = sqrt(fmax(0.0, tr + dd));
            } else {
                phonons_at(kx, ky, w, ev);
            }
            for (int b = 0; b < nb; ++b)
                if (w[b] > 1e-9) cv_w[cv_nw++] = w[b];
        }
    cv_ndp = cv_nw * cv_weight;
    double wmax = 1e-6;
    for (int i = 0; i < cv_nw; ++i) if (cv_w[i] > wmax) wmax = cv_w[i];
    const double t0 = log10(1e-3 * wmax), t1 = log10(10.0 * wmax);
    for (int j = 0; j < CV_PTS; ++j) {
        cv_T[j] = pow(10.0, t0 + (t1 - t0) * j / (CV_PTS - 1));
        cv_C[j] = cv_at(cv_T[j]) / cv_ndp;
    }
    cv_dirty = false;
    d_cv_T = -1.0;
}

/* ===================================================================
 * Spectra: full 2D FFT -> mode energies (BZ map, FPUT harmonics)
 * =================================================================== */

static double *kre[4] = { NULL }, *kim[4] = { NULL };
static double *emap = NULL, *Dxx_k = NULL, *Dyy_k = NULL, *Dxy_k = NULL;
static bool    fc_dirty = true;

static void spectra_alloc(void)
{
    for (int f = 0; f < 4; ++f) {
        kre[f] = xrealloc(kre[f], (size_t)NN * sizeof(double));
        kim[f] = xrealloc(kim[f], (size_t)NN * sizeof(double));
    }
    emap  = xrealloc(emap,  (size_t)NN * sizeof(double));
    Dxx_k = xrealloc(Dxx_k, (size_t)NN * sizeof(double));
    Dyy_k = xrealloc(Dyy_k, (size_t)NN * sizeof(double));
    Dxy_k = xrealloc(Dxy_k, (size_t)NN * sizeof(double));
    col_re = xrealloc(col_re, (size_t)N * sizeof(double));
    col_im = xrealloc(col_im, (size_t)N * sizeof(double));
    memset(emap, 0, (size_t)NN * sizeof(double));
    fc_dirty = true;
}

static void spectra_update(void)
{
    if (fc_dirty) {
        for (int ny = 0; ny < N; ++ny)
            for (int nx = 0; nx < N; ++nx)
                force_const_mono(2.0 * M_PI * nx / N, 2.0 * M_PI * ny / N,
                                 &Dxx_k[ny * N + nx], &Dyy_k[ny * N + nx],
                                 &Dxy_k[ny * N + nx]);
        fc_dirty = false;
    }
    for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
            const int I = ID(x, y), k = y * N + x;
            const double a = alive[I] ? 1.0 : 0.0;
            kre[0][k] = a * ux[I]; kre[1][k] = a * uy[I];
            kre[2][k] = a * vx[I]; kre[3][k] = a * vy[I];
            kim[0][k] = kim[1][k] = kim[2][k] = kim[3][k] = 0.0;
        }
    for (int f = 0; f < 4; ++f) fft2d(kre[f], kim[f]);
    const double mbar = 0.5 * (1.0 + prm.mratio);
    const double inv  = 1.0 / (double)NN;
    for (int k = 0; k < NN; ++k) {
        const double uxr = kre[0][k], uxi = kim[0][k];
        const double uyr = kre[1][k], uyi = kim[1][k];
        const double v2  = kre[2][k] * kre[2][k] + kim[2][k] * kim[2][k]
                         + kre[3][k] * kre[3][k] + kim[3][k] * kim[3][k];
        const double pe  = Dxx_k[k] * (uxr * uxr + uxi * uxi)
                         + Dyy_k[k] * (uyr * uyr + uyi * uyi)
                         + 2.0 * Dxy_k[k] * (uxr * uyr + uxi * uyi);
        emap[k] = inv * 0.5 * (mbar * v2 + pe);
    }
}

/* Energy in the +k/-k pair (harmonic, ideal-crystal estimate). */
static double mode_energy(int nx, int ny)
{
    const int a = imod(nx, N), b = imod(ny, N);
    const int c = imod(-nx, N), d = imod(-ny, N);
    double e = emap[b * N + a];
    if (a != c || b != d) e += emap[d * N + c];
    return e;
}

/* ===================================================================
 * Measured S(k, omega) along Gamma-X-M-Gamma
 * =================================================================== */

static int     s_np = 0;
static float  *s_ring = NULL;          /* [np][pol][SW_LEN][re,im]      */
static double *s_acc  = NULL;          /* [np][SW_LEN/2]                */
static int     s_head = 0, s_fill = 0, s_since = 0, s_nacc = 0;
static bool    s_tex_dirty = true;
static double *pr_re = NULL, *pr_im = NULL;
static double  w_re[SW_LEN], w_im[SW_LEN], hann[SW_LEN];

static double s_domega(void) { return 2.0 * M_PI / (SW_LEN * S_STRIDE * DT); }
static double s_nyquist(void) { return M_PI / (S_STRIDE * DT); }

static void s_reset(void)
{
    s_head = s_fill = s_since = s_nacc = 0;
    if (s_acc) memset(s_acc, 0, (size_t)s_np * (SW_LEN / 2) * sizeof *s_acc);
    s_tex_dirty = true;
}

static void s_alloc(void)
{
    s_np   = 3 * N / 2 + 1;
    s_ring = xrealloc(s_ring, (size_t)s_np * 2 * SW_LEN * 2 * sizeof *s_ring);
    s_acc  = xrealloc(s_acc,  (size_t)s_np * (SW_LEN / 2) * sizeof *s_acc);
    pr_re  = xrealloc(pr_re, (size_t)N * sizeof *pr_re);
    pr_im  = xrealloc(pr_im, (size_t)N * sizeof *pr_im);
    for (int t = 0; t < SW_LEN; ++t)
        hann[t] = 0.5 - 0.5 * cos(2.0 * M_PI * t / (SW_LEN - 1));
    s_reset();
}

static inline float *s_slot(int p, int pol, int t)
{
    return &s_ring[(((size_t)p * 2 + pol) * SW_LEN + t) * 2];
}

static void s_spectrum(void)
{
    const int H = SW_LEN / 2;
    for (int p = 0; p < s_np; ++p) {
        double *acc = &s_acc[(size_t)p * H];
        for (int pol = 0; pol < 2; ++pol) {
            double mr = 0.0, mi = 0.0;
            for (int t = 0; t < SW_LEN; ++t) {
                const float *z = s_slot(p, pol, (s_head + t) % SW_LEN);
                mr += z[0]; mi += z[1];
            }
            mr /= SW_LEN; mi /= SW_LEN;
            for (int t = 0; t < SW_LEN; ++t) {
                const float *z = s_slot(p, pol, (s_head + t) % SW_LEN);
                w_re[t] = (z[0] - mr) * hann[t];
                w_im[t] = (z[1] - mi) * hann[t];
            }
            fft_run(&plan_W, w_re, w_im);
            acc[0] += w_re[0] * w_re[0] + w_im[0] * w_im[0];
            for (int j = 1; j < H; ++j)
                acc[j] += w_re[j] * w_re[j] + w_im[j] * w_im[j]
                        + w_re[SW_LEN - j] * w_re[SW_LEN - j]
                        + w_im[SW_LEN - j] * w_im[SW_LEN - j];
        }
    }
    ++s_nacc;
    s_since = 0;
    s_tex_dirty = true;
}

/* O(N^2) per sample: project onto lines, then 1D FFTs. */
static void s_sample(void)
{
    const int half = N / 2;
    for (int pol = 0; pol < 2; ++pol) {
        const double *u = pol ? uy : ux;
        /* Gamma -> X: k = (2 pi n/N, 0) */
        for (int x = 0; x < N; ++x) { pr_re[x] = 0.0; pr_im[x] = 0.0; }
        for (int y = 0; y < N; ++y)
            for (int x = 0; x < N; ++x) {
                const int I = ID(x, y);
                if (alive[I]) pr_re[x] += u[I];
            }
        fft_run(&plan_N, pr_re, pr_im);
        for (int n = 0; n <= half; ++n) {
            float *z = s_slot(n, pol, s_head);
            z[0] = (float)pr_re[n]; z[1] = (float)pr_im[n];
        }
        /* X -> M: k = (pi, 2 pi n/N) */
        for (int y = 0; y < N; ++y) { pr_re[y] = 0.0; pr_im[y] = 0.0; }
        for (int y = 0; y < N; ++y)
            for (int x = 0; x < N; ++x) {
                const int I = ID(x, y);
                if (alive[I]) pr_re[y] += (x & 1) ? -u[I] : u[I];
            }
        fft_run(&plan_N, pr_re, pr_im);
        for (int n = 1; n <= half; ++n) {
            float *z = s_slot(half + n, pol, s_head);
            z[0] = (float)pr_re[n]; z[1] = (float)pr_im[n];
        }
        /* M -> Gamma: k = (2 pi n/N)(1, 1) */
        for (int d = 0; d < N; ++d) { pr_re[d] = 0.0; pr_im[d] = 0.0; }
        for (int y = 0; y < N; ++y)
            for (int x = 0; x < N; ++x) {
                const int I = ID(x, y);
                if (alive[I]) pr_re[(x + y) & (N - 1)] += u[I];
            }
        fft_run(&plan_N, pr_re, pr_im);
        for (int n = half - 1; n >= 0; --n) {
            float *z = s_slot(N + (half - n), pol, s_head);
            z[0] = (float)pr_re[n]; z[1] = (float)pr_im[n];
        }
    }
    s_head = (s_head + 1) % SW_LEN;
    if (s_fill < SW_LEN) ++s_fill;
    ++s_since;
    if (s_fill == SW_LEN && s_since >= SW_LEN / 4) s_spectrum();
}

/* ===================================================================
 * Time series and temperature profile
 * =================================================================== */

static float ts_KE[TS_LEN], ts_PE[TS_LEN], ts_E[TS_LEN], ts_m[4][TS_LEN];
static int   ts_head = 0, ts_n = 0;
static double prof_T[NMAX];
static bool   prof_init = false;

static void ts_reset(void) { ts_head = ts_n = 0; }

static void ts_push(void)
{
    ts_KE[ts_head] = (float)d_KE;
    ts_PE[ts_head] = (float)d_PE;
    ts_E[ts_head]  = (float)(d_KE + d_PE);
    for (int m = 0; m < 4; ++m)
        ts_m[m][ts_head] = (float)mode_energy((m + 1) * sel_nx, (m + 1) * sel_ny);
    ts_head = (ts_head + 1) % TS_LEN;
    if (ts_n < TS_LEN) ++ts_n;
}

static void profile_update(void)
{
    for (int x = 0; x < N; ++x) {
        double ke = 0.0; int n = 0;
        for (int y = 0; y < N; ++y) {
            const int I = ID(x, y);
            if (!alive[I]) continue;
            ke += 0.5 * ms[I] * (vx[I] * vx[I] + vy[I] * vy[I]);
            ++n;
        }
        const double T = n ? ke / n : 0.0;     /* 2 dof: KE/atom = kT */
        prof_T[x] = prof_init ? prof_T[x] + 0.02 * (T - prof_T[x]) : T;
    }
    prof_init = true;
}

/* ===================================================================
 * Lattice lifecycle
 * =================================================================== */

static void destroy_textures(void)
{
    if (heat_tex) { SDL_DestroyTexture(heat_tex); heat_tex = NULL; }
    if (bz_tex)   { SDL_DestroyTexture(bz_tex);   bz_tex = NULL; }
    if (s_tex)    { SDL_DestroyTexture(s_tex);    s_tex = NULL; }
    heat_tex_n = bz_tex_n = s_tex_w = s_tex_h = 0;
}

static void zero_motion(void)
{
    const size_t s = (size_t)P * (size_t)P * sizeof(double);
    memset(ux, 0, s); memset(uy, 0, s);
    memset(vx, 0, s); memset(vy, 0, s);
    memset(fx, 0, s); memset(fy, 0, s);
    E_ref_set = false;
    pk_track  = false;
}

static void apply_settings(void)
{
    update_masses();
    update_sponge();
    fill_alive_ghosts();
    compute_forces();
    disp_dirty = true;
    fc_dirty   = true;
    cv_dirty   = true;
    cv_dirty_since = now_ms;
    s_reset();
    E_ref_set = false;
    prm_last = prm;
}

static void set_lattice_size(int n)
{
    if (n < NMIN) n = NMIN;
    if (n > NMAX) n = NMAX;
    alloc_lattice(n);
    fft_init(&plan_N, N);
    if (plan_W.n != SW_LEN) fft_init(&plan_W, SW_LEN);
    spectra_alloc();
    s_alloc();
    ts_reset();
    prof_init = false;
    if (!headless) destroy_textures();
    driver_I = -1;
    step_no = 0; sim_time = 0.0;
    if (sel_nx > N / 2) sel_nx = N / 8;
    if (sel_ny > N / 2) sel_ny = 0;
    for (int i = 0; i <= F_COUNT; ++i) cs_ema[i] = 0.0;
    apply_settings();
}

/* ===================================================================
 * Excitations
 * =================================================================== */

static void min_image(double *d)
{
    if (bc != BC_PERIODIC) return;
    if (*d > 0.5 * N) *d -= N;
    else if (*d < -0.5 * N) *d += N;
}

static void excite(int nx, int ny, int b)
{
    nx = imod(nx, N); ny = imod(ny, N);
    const double kx = 2.0 * M_PI * nx / N, ky = 2.0 * M_PI * ny / N;
    double w[4], ev[4][4];
    const int nb = phonons_at(kx, ky, w, ev);
    if (b >= nb) b = nb - 1;
    if (b < 0) b = 0;
    sel_nx = nx; sel_ny = ny; sel_b = b;
    if (!additive) zero_motion();

    double x0 = 0.0, y0 = 0.0;
    if (shape == SH_PACKET) {
        double vgx, vgy;
        group_velocity(kx, ky, b, &vgx, &vgy);
        const double vg = hypot(vgx, vgy);
        x0 = y0 = 0.5 * (N - 1);
        if (vg > 1e-6) { x0 -= 0.25 * N * vgx / vg; y0 -= 0.25 * N * vgy / vg; }
        pk_track = true;
        pk_x0 = x0; pk_y0 = y0; pk_vgx = vgx; pk_vgy = vgy; pk_t0 = sim_time;
    }
    const double sig = prm.pk_sigma;
    const int nsub = nb / 2;
    for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
            const int I = ID(x, y);
            if (!alive[I]) continue;
            const int s = (nsub == 2) ? ((x + y) & 1) : 0;
            const double ex = ev[b][2 * s], ey = ev[b][2 * s + 1];
            const double A = prm.amp * sqrt((double)nsub / nominal_mass(x, y));
            double ph, env = 1.0;
            if (shape == SH_PACKET) {
                double dx = x - x0, dy = y - y0;
                min_image(&dx); min_image(&dy);
                env = exp(-(dx * dx + dy * dy) / (2.0 * sig * sig));
                ph = kx * dx + ky * dy;
            } else {
                ph = kx * x + ky * y;
            }
            const double c = cos(ph) * env * A, sn = sin(ph) * env * A;
            ux[I] += c * ex; uy[I] += c * ey;
            if (shape != SH_STANDING) {
                vx[I] += w[b] * sn * ex;
                vy[I] += w[b] * sn * ey;
            }
        }
    compute_forces();
    E_ref_set = false;
    toast("Branch %d at k = (%d, %d)\xC2\xB7" "2\xCF\x80/N:  \xCF\x89 = %.3f   [%s%s]",
          b + 1, signed_n(nx), signed_n(ny), w[b], sh_names[shape],
          additive ? ", additive" : "");
}

static void poke_radial(double cx, double cy)
{
    const double sig = prm.poke_sigma;
    const int R = (int)ceil(4.0 * sig);
    const int ix = (int)lround(cx), iy = (int)lround(cy);
    for (int dy = -R; dy <= R; ++dy)
        for (int dx = -R; dx <= R; ++dx) {
            int x = ix + dx, y = iy + dy;
            if (bc == BC_PERIODIC) { x = imod(x, N); y = imod(y, N); }
            else if (!is_real(x, y)) continue;
            const int I = ID(x, y);
            if (!alive[I]) continue;
            const double rx = ix + dx - cx, ry = iy + dy - cy;
            const double g = exp(-(rx * rx + ry * ry) / (2.0 * sig * sig));
            ux[I] += 0.8 * prm.amp * rx / sig * g;
            uy[I] += 0.8 * prm.amp * ry / sig * g;
        }
    compute_forces();
    E_ref_set = false;
}

static void poke_impulse(double cx, double cy, double dirx, double diry)
{
    const double len = hypot(dirx, diry);
    if (len < 1e-9) return;
    if (len > 4.0) { dirx *= 4.0 / len; diry *= 4.0 / len; }
    const double sig = prm.poke_sigma;
    const int R = (int)ceil(4.0 * sig);
    const int ix = (int)lround(cx), iy = (int)lround(cy);
    for (int dy = -R; dy <= R; ++dy)
        for (int dx = -R; dx <= R; ++dx) {
            int x = ix + dx, y = iy + dy;
            if (bc == BC_PERIODIC) { x = imod(x, N); y = imod(y, N); }
            else if (!is_real(x, y)) continue;
            const int I = ID(x, y);
            if (!alive[I]) continue;
            const double rx = ix + dx - cx, ry = iy + dy - cy;
            const double g = exp(-(rx * rx + ry * ry) / (2.0 * sig * sig));
            vx[I] += 0.5 * prm.amp * dirx * g;
            vy[I] += 0.5 * prm.amp * diry * g;
        }
    E_ref_set = false;
}

static void thermalize(double T)
{
    double px = 0.0, py = 0.0, mt = 0.0;
    for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
            const int I = ID(x, y);
            if (!alive[I]) continue;
            const double sg = (T > 0.0) ? sqrt(T / ms[I]) : 0.0;
            vx[I] = sg * randn();
            vy[I] = sg * randn();
            px += ms[I] * vx[I]; py += ms[I] * vy[I]; mt += ms[I];
        }
    if (mt > 0.0) {
        px /= mt; py /= mt;
        for (int y = 0; y < N; ++y)
            for (int x = 0; x < N; ++x) {
                const int I = ID(x, y);
                if (alive[I]) { vx[I] -= px; vy[I] -= py; }
            }
    }
    E_ref_set = false;
}

static void add_disp_noise(double a)
{
    for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
            const int I = ID(x, y);
            if (alive[I]) { ux[I] += a * randn(); uy[I] += a * randn(); }
        }
    compute_forces();
    E_ref_set = false;
}

static void apply_tool_at(int x, int y)
{
    if (!is_real(x, y)) return;
    const int I = ID(x, y);
    switch (tool) {
    case TOOL_HEAVY:   defm[I] = 4.0;  alive[I] = 1; break;
    case TOOL_LIGHT:   defm[I] = 0.25; alive[I] = 1; break;
    case TOOL_VACANCY:
        alive[I] = 0;
        ux[I] = uy[I] = vx[I] = vy[I] = 0.0;
        if (driver_I == I) driver_I = -1;
        break;
    case TOOL_ERASE:   defm[I] = 1.0;  alive[I] = 1; break;
    case TOOL_DRIVER:
        driver_I = (driver_I == I) ? -1 : I;
        alive[I] = 1;
        disp_dirty = true;
        toast(driver_I >= 0 ? "Driver placed (\xCF\x89 = %.2f, F = %.2f, pol %c). "
                              "Right-click the dispersion plot to set \xCF\x89."
                            : "Driver removed%s",
              prm.drive_w, prm.drive_F, drive_pol ? 'y' : 'x');
        break;
    default: break;
    }
    update_masses();
    fill_alive_ghosts();
    compute_forces();
    E_ref_set = false;
}

static void clear_defects(void)
{
    for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
            const int I = ID(x, y);
            defm[I] = 1.0; alive[I] = 1;
        }
    driver_I = -1;
}

/* ===================================================================
 * Advance + diagnostics
 * =================================================================== */

static void advance(int nsteps)
{
    for (int s = 0; s < nsteps; ++s) {
        verlet_step();
        if (step_no % S_STRIDE == 0) s_sample();
    }
}

static void check_param_changes(void)
{
    const Params *o = &prm_last;
    const bool lat  = prm.KL != o->KL || prm.KT != o->KT || prm.KD != o->KD
                   || prm.mratio != o->mratio;
    const bool mass = prm.mratio != o->mratio || prm.disorder != o->disorder;
    const bool anh  = prm.alpha != o->alpha || prm.beta != o->beta;
    if (mass) update_masses();
    if (lat) {
        disp_dirty = true; fc_dirty = true;
        cv_dirty = true; cv_dirty_since = now_ms;
    }
    if (lat || anh || mass) { s_reset(); E_ref_set = false; }
    if (prm.drive_w != o->drive_w) disp_dirty = true;
    prm_last = prm;
}

static bool system_is_open(void)
{
    return thermo != TH_OFF || bc == BC_ABSORB
        || (driver_I >= 0 && prm.drive_F != 0.0);
}

static void diagnostics_update(double dt_sim)
{
    compute_energy(&d_KE, &d_PE);
    double mu = 0.0; int na = 0;
    for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
            const int I = ID(x, y);
            if (!alive[I]) continue;
            ++na;
            const double m = fmax(fabs(ux[I]), fabs(uy[I]));
            if (m > mu || m != m) mu = m;
        }
    d_maxu = mu; d_nalive = na;
    if (!isfinite(d_KE + d_PE) || !isfinite(mu) || mu > 0.4 * N) {
        zero_motion();
        compute_energy(&d_KE, &d_PE);
        toast("Blow-up (anharmonic runaway): motion reset.\n"
              "Lower \xCE\xB1 or the amplitude, or raise \xCE\xB2.");
    }
    if (!E_ref_set) { E_ref = d_KE + d_PE; E_ref_set = true; }
    d_kTeff = na ? d_KE / na : 0.0;
    if (!cv_dirty && fabs(prm.kT - d_cv_T) > 1e-12) {
        d_cv = cv_at(prm.kT); d_cv_T = prm.kT;
    }
    if (dt_sim > 0.0) {
        const double a = 0.02;
        J_in  += a * (heat_in  / dt_sim - J_in);
        J_out += a * (heat_out / dt_sim - J_out);
    }
    heat_in = heat_out = 0.0;
}

/* ===================================================================
 * Colour maps
 * =================================================================== */

static void build_lut(SDL_Color *lut, int nstops, const double *pos,
                      const uint8_t (*rgb)[3])
{
    for (int i = 0; i < 256; ++i) {
        const double t = i / 255.0;
        int k = 0;
        while (k < nstops - 2 && t > pos[k + 1]) ++k;
        const double u = clampd((t - pos[k]) / (pos[k + 1] - pos[k]), 0.0, 1.0);
        lut[i].r = (uint8_t)lround(rgb[k][0] + u * (rgb[k + 1][0] - rgb[k][0]));
        lut[i].g = (uint8_t)lround(rgb[k][1] + u * (rgb[k + 1][1] - rgb[k][1]));
        lut[i].b = (uint8_t)lround(rgb[k][2] + u * (rgb[k + 1][2] - rgb[k][2]));
        lut[i].a = 255;
    }
}

static void build_luts(void)
{
    static const double ps[5] = { 0.0, 0.25, 0.5, 0.75, 1.0 };
    static const uint8_t cs[5][3] = { { 34, 38, 72 }, { 92, 36, 134 },
        { 190, 58, 96 }, { 248, 146, 40 }, { 252, 248, 176 } };
    build_lut(lut_seq, 5, ps, cs);
    static const double pd[5] = { 0.0, 0.25, 0.5, 0.75, 1.0 };
    static const uint8_t cd[5][3] = { { 120, 190, 255 }, { 40, 95, 200 },
        { 26, 26, 32 }, { 200, 55, 40 }, { 255, 170, 110 } };
    build_lut(lut_div, 5, pd, cd);
    static const double pss[5] = { 0.0, 0.3, 0.6, 0.85, 1.0 };
    static const uint8_t css[5][3] = { { 0, 0, 0 }, { 60, 20, 110 },
        { 200, 60, 80 }, { 250, 160, 40 }, { 255, 255, 200 } };
    build_lut(lut_s, 5, pss, css);
}

static SDL_Color colour_for(double v, double S, bool sgn)
{
    const double t = prm.gain * v / (S > 1e-30 ? S : 1e-30);
    if (sgn) {
        int i = (int)lround((clampd(t, -1.0, 1.0) * 0.5 + 0.5) * 255.0);
        return lut_div[i < 0 ? 0 : (i > 255 ? 255 : i)];
    }
    int i = (int)lround(clampd(t, 0.0, 1.0) * 255.0);
    return lut_seq[i];
}

static double scale_for(int slot, double curmax)
{
    const double ref = field_ref[slot];
    if (slot == F_MASS || cscale == CS_FIXED) return ref;
    if (cscale == CS_LOCK) return cs_lock[slot] > 1e-30 ? cs_lock[slot] : ref;
    if (cs_ema[slot] < 1e-12) cs_ema[slot] = curmax;
    else if (!paused) cs_ema[slot] += 0.012 * (curmax - cs_ema[slot]);
    return cs_ema[slot] > 1e-12 ? cs_ema[slot] : ref;
}

/* ===================================================================
 * Text rendering with a small texture cache
 * =================================================================== */

#define TC_SIZE  1024
#define TC_PROBE 8
typedef struct {
    TTF_Font *f; uint32_t col, hash; char s[100];
    SDL_Texture *t; int w, h; unsigned last;
} TCEntry;
static TCEntry tcache[TC_SIZE];

static uint32_t hash_text(const char *s, const TTF_Font *f, uint32_t col)
{
    uint32_t h = 2166136261u ^ col ^ (uint32_t)(uintptr_t)f;
    while (*s) { h ^= (unsigned char)*s++; h *= 16777619u; }
    return h;
}

static void text_draw(TTF_Font *f, SDL_Color c, int x, int y, const char *s)
{
    if (!s || !*s || !f || !ren) return;
    const uint32_t col = ((uint32_t)c.r << 24) | ((uint32_t)c.g << 16)
                       | ((uint32_t)c.b << 8) | c.a;
    if (strlen(s) < sizeof tcache[0].s) {
        const uint32_t h = hash_text(s, f, col);
        const int base = (int)(h % TC_SIZE);
        int victim = -1;
        unsigned oldest = UINT_MAX;
        for (int k = 0; k < TC_PROBE; ++k) {
            const int slot = (base + k) % TC_SIZE;
            TCEntry *e = &tcache[slot];
            if (e->t && e->hash == h && e->f == f && e->col == col && !strcmp(e->s, s)) {
                e->last = frame_no;
                SDL_Rect d = { x, y, e->w, e->h };
                SDL_RenderCopy(ren, e->t, NULL, &d);
                return;
            }
            const unsigned age = e->t ? e->last : 0u;
            if (victim < 0 || age < oldest) { victim = slot; oldest = age; }
        }
        SDL_Surface *surf = TTF_RenderUTF8_Blended(f, s, c);
        if (!surf) return;
        SDL_Texture *t = SDL_CreateTextureFromSurface(ren, surf);
        TCEntry *e = &tcache[victim];
        if (e->t) SDL_DestroyTexture(e->t);
        e->t = t; e->f = f; e->col = col; e->hash = h;
        e->w = surf->w; e->h = surf->h; e->last = frame_no;
        snprintf(e->s, sizeof e->s, "%s", s);
        SDL_FreeSurface(surf);
        if (t) { SDL_Rect d = { x, y, e->w, e->h }; SDL_RenderCopy(ren, t, NULL, &d); }
        return;
    }
    SDL_Surface *surf = TTF_RenderUTF8_Blended(f, s, c);
    if (!surf) return;
    SDL_Texture *t = SDL_CreateTextureFromSurface(ren, surf);
    if (t) {
        SDL_Rect d = { x, y, surf->w, surf->h };
        SDL_RenderCopy(ren, t, NULL, &d);
        SDL_DestroyTexture(t);
    }
    SDL_FreeSurface(surf);
}

static void textf(TTF_Font *f, SDL_Color c, int x, int y, const char *fmt, ...)
{
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    text_draw(f, c, x, y, buf);
}

static int text_width(TTF_Font *f, const char *s)
{
    int w = 0, h = 0;
    if (f && s) TTF_SizeUTF8(f, s, &w, &h);
    return w;
}

static void tcache_clear(void)
{
    for (int i = 0; i < TC_SIZE; ++i)
        if (tcache[i].t) { SDL_DestroyTexture(tcache[i].t); tcache[i].t = NULL; }
}

/* ===================================================================
 * Geometry batching (SDL_RenderGeometry when available)
 * =================================================================== */

#if HAVE_GEOMETRY
static SDL_Vertex *gv = NULL;
static int *gidx = NULL;
static int gv_n = 0, gv_cap = 0, gi_n = 0, gi_cap = 0;

static void geo_begin(void) { gv_n = gi_n = 0; }

static void geo_reserve(int nv, int ni)
{
    if (gv_n + nv > gv_cap) {
        gv_cap = (gv_n + nv) * 2;
        gv = xrealloc(gv, (size_t)gv_cap * sizeof *gv);
    }
    if (gi_n + ni > gi_cap) {
        gi_cap = (gi_n + ni) * 2;
        gidx = xrealloc(gidx, (size_t)gi_cap * sizeof *gidx);
    }
}

static void geo_vertex(float x, float y, SDL_Color c)
{
    SDL_Vertex *v = &gv[gv_n++];
    v->position.x = x; v->position.y = y;
    v->color = c;
    v->tex_coord.x = 0.0f; v->tex_coord.y = 0.0f;
}

static void geo_quad_idx(void)
{
    const int b = gv_n - 4;
    gidx[gi_n++] = b;     gidx[gi_n++] = b + 1; gidx[gi_n++] = b + 2;
    gidx[gi_n++] = b;     gidx[gi_n++] = b + 2; gidx[gi_n++] = b + 3;
}

static void geo_rect(float x, float y, float w, float h, SDL_Color c)
{
    geo_reserve(4, 6);
    geo_vertex(x, y, c); geo_vertex(x + w, y, c);
    geo_vertex(x + w, y + h, c); geo_vertex(x, y + h, c);
    geo_quad_idx();
}

static void geo_line(float x0, float y0, float x1, float y1, float width,
                     SDL_Color c0, SDL_Color c1)
{
    const float dx = x1 - x0, dy = y1 - y0;
    const float len = sqrtf(dx * dx + dy * dy);
    if (len < 1e-4f) return;
    const float nx = -dy / len * 0.5f * width, ny = dx / len * 0.5f * width;
    geo_reserve(4, 6);
    geo_vertex(x0 + nx, y0 + ny, c0); geo_vertex(x1 + nx, y1 + ny, c1);
    geo_vertex(x1 - nx, y1 - ny, c1); geo_vertex(x0 - nx, y0 - ny, c0);
    geo_quad_idx();
}

static void geo_flush(void)
{
    if (gv_n > 0) SDL_RenderGeometry(ren, NULL, gv, gv_n, gidx, gi_n);
    gv_n = gi_n = 0;
}
#else
static void geo_begin(void) {}
static void geo_rect(float x, float y, float w, float h, SDL_Color c)
{
    SDL_SetRenderDrawColor(ren, c.r, c.g, c.b, c.a);
    SDL_FRect r = { x, y, w, h };
    SDL_RenderFillRectF(ren, &r);
}
static void geo_line(float x0, float y0, float x1, float y1, float width,
                     SDL_Color c0, SDL_Color c1)
{
    (void)width; (void)c1;
    SDL_SetRenderDrawColor(ren, c0.r, c0.g, c0.b, c0.a);
    SDL_RenderDrawLineF(ren, x0, y0, x1, y1);
}
static void geo_flush(void) {}
#endif

/* ===================================================================
 * Drawing helpers
 * =================================================================== */

static void set_col(SDL_Color c) { SDL_SetRenderDrawColor(ren, c.r, c.g, c.b, c.a); }
static void set_rgba(int r, int g, int b, int a)
{
    SDL_SetRenderDrawColor(ren, (Uint8)r, (Uint8)g, (Uint8)b, (Uint8)a);
}

static void fill_rect(int x, int y, int w, int h)
{
    SDL_Rect r = { x, y, w, h };
    SDL_RenderFillRect(ren, &r);
}

static void outline_rect(int x, int y, int w, int h)
{
    SDL_Rect r = { x, y, w, h };
    SDL_RenderDrawRect(ren, &r);
}

static void dashed_hline(int x0, int x1, int y, int dash)
{
    for (int x = x0; x < x1; x += 2 * dash)
        SDL_RenderDrawLine(ren, x, y, (x + dash < x1 ? x + dash : x1), y);
}

static void dotted_vline(int x, int y0, int y1)
{
    for (int y = y0; y < y1; y += 4) SDL_RenderDrawPoint(ren, x, y);
}

static double cellpx(void) { return (double)LAT_PX / N; }
static float scr_x(double x) { return (float)(LAT_X + (x + 0.5) * cellpx()); }
static float scr_y(double y) { return (float)(LAT_Y + (N - 0.5 - y) * cellpx()); }

static void mouse_to_lattice(int mx, int my, double *cx, double *cy)
{
    *cx = (mx - LAT_X) / cellpx() - 0.5;
    *cy = N - 0.5 - (my - LAT_Y) / cellpx();
}

static bool in_rect(int mx, int my, int x, int y, int w, int h)
{
    return mx >= x && mx < x + w && my >= y && my < y + h;
}

/* ===================================================================
 * Lattice rendering
 * =================================================================== */

static float *fbuf = NULL;
static int    fbuf_n = 0;

static double field_value(int f, int x, int y)
{
    const int I = ID(x, y);
    switch (f) {
    case F_ABSU: return hypot(ux[I], uy[I]);
    case F_UX:   return ux[I];
    case F_UY:   return uy[I];
    case F_ABSV: return hypot(vx[I], vy[I]);
    case F_KE:   return 0.5 * ms[I] * (vx[I] * vx[I] + vy[I] * vy[I]);
    case F_DIV:  return 0.5 * (ux[I + 1] - ux[I - 1] + uy[I + P] - uy[I - P]);
    case F_CURL: return 0.5 * (uy[I + 1] - uy[I - 1] - (ux[I + P] - ux[I - P]));
    case F_MASS: return log2(ms[I]);
    default:     return 0.0;
    }
}

static double compute_field(void)
{
    if (fbuf_n != NN) { fbuf = xrealloc(fbuf, (size_t)NN * sizeof *fbuf); fbuf_n = NN; }
    fill_ghosts(ux); fill_ghosts(uy);
    double mx = 0.0;
    for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
            const int I = ID(x, y);
            const double v = alive[I] ? field_value(field, x, y) : 0.0;
            fbuf[y * N + x] = (float)v;
            if (fabs(v) > mx) mx = fabs(v);
        }
    return mx;
}

static void draw_lattice(void)
{
    set_rgba(22, 22, 28, 255);
    fill_rect(LAT_X, LAT_Y, LAT_PX, LAT_PX);

    const double fmax_v = compute_field();
    const double S = scale_for(field, fmax_v);
    const bool sgn = field_signed[field];
    const double cell = cellpx();
    const double ex = prm.exag;

    if (style == ST_HEAT) {
        if (!heat_tex || heat_tex_n != N) {
            if (heat_tex) SDL_DestroyTexture(heat_tex);
            SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "linear");
            heat_tex = SDL_CreateTexture(ren, SDL_PIXELFORMAT_ARGB8888,
                                         SDL_TEXTUREACCESS_STREAMING, N, N);
            heat_tex_n = N;
#if SDL_VERSION_ATLEAST(2, 0, 12)
            if (heat_tex) SDL_SetTextureScaleMode(heat_tex, SDL_ScaleModeLinear);
#endif
        }
        if (heat_tex) {
            void *pix; int pitch;
            if (SDL_LockTexture(heat_tex, NULL, &pix, &pitch) == 0) {
                for (int y = 0; y < N; ++y) {
                    uint32_t *row = (uint32_t *)((uint8_t *)pix + (size_t)(N - 1 - y) * pitch);
                    for (int x = 0; x < N; ++x) {
                        if (!alive[ID(x, y)]) { row[x] = 0xFF000000u; continue; }
                        const SDL_Color c = colour_for(fbuf[y * N + x], S, sgn);
                        row[x] = 0xFF000000u | ((uint32_t)c.r << 16) | ((uint32_t)c.g << 8) | c.b;
                    }
                }
                SDL_UnlockTexture(heat_tex);
            }
            SDL_Rect d = { LAT_X, LAT_Y, LAT_PX, LAT_PX };
            SDL_RenderCopy(ren, heat_tex, NULL, &d);
        }
        return;
    }

    geo_begin();
    if (style == ST_BONDS) {
        double smax = 0.0;
        for (int y = 0; y < N; ++y)
            for (int x = 0; x < N; ++x) {
                const int I = ID(x, y);
                if (!alive[I]) continue;
                if (x + 1 < N && alive[I + 1]) smax = fmax(smax, fabs(ux[I + 1] - ux[I]));
                if (y + 1 < N && alive[I + P]) smax = fmax(smax, fabs(uy[I + P] - uy[I]));
            }
        const double Sb = scale_for(SLOT_BOND, smax);
        const float bw = (float)fmax(1.0, 0.18 * cell);
        for (int y = 0; y < N; ++y)
            for (int x = 0; x < N; ++x) {
                const int I = ID(x, y);
                if (!alive[I]) continue;
                const float x0 = scr_x(x + ex * ux[I]), y0 = scr_y(y + ex * uy[I]);
                if (x + 1 < N && alive[I + 1]) {
                    const int J = I + 1;
                    const SDL_Color c = colour_for(ux[J] - ux[I], Sb, true);
                    geo_line(x0, y0, scr_x(x + 1 + ex * ux[J]), scr_y(y + ex * uy[J]), bw, c, c);
                }
                if (y + 1 < N && alive[I + P]) {
                    const int J = I + P;
                    const SDL_Color c = colour_for(uy[J] - uy[I], Sb, true);
                    geo_line(x0, y0, scr_x(x + ex * ux[J]), scr_y(y + 1 + ex * uy[J]), bw, c, c);
                }
            }
    }
    const double rbase = (style == ST_BONDS ? 0.17 : 0.34) * cell;
    for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
            const int I = ID(x, y);
            if (!alive[I]) continue;
            double r = rbase * pow(ms[I], 0.25);
            r = clampd(r, 0.6, 0.49 * cell);
            const SDL_Color c = colour_for(fbuf[y * N + x], S, sgn);
            const float cx = scr_x(x + ex * ux[I]), cy = scr_y(y + ex * uy[I]);
            geo_rect(cx - (float)r, cy - (float)r, (float)(2 * r), (float)(2 * r), c);
        }
    geo_flush();
}

static void draw_lattice_overlays(void)
{
    const double cell = cellpx();
    SDL_RenderSetClipRect(ren, &(SDL_Rect){ LAT_X, LAT_Y, LAT_PX, LAT_PX });

    if (thermo == TH_WALLS) {
        const int w = bath_width();
        const int c0 = (bc == BC_PERIODIC) ? N / 2 : N - w;
        set_rgba(COL_HOT.r, COL_HOT.g, COL_HOT.b, 45);
        fill_rect(LAT_X, LAT_Y, (int)(w * cell), LAT_PX);
        set_rgba(COL_COLD.r, COL_COLD.g, COL_COLD.b, 45);
        fill_rect(LAT_X + (int)(c0 * cell), LAT_Y, (int)(w * cell), LAT_PX);
    }
    if (bc == BC_ABSORB) {
        const int w = (int)(sponge_width() * cell);
        set_rgba(120, 120, 140, 70);
        outline_rect(LAT_X + w, LAT_Y + w, LAT_PX - 2 * w, LAT_PX - 2 * w);
    }
    if (driver_I >= 0) {
        const int x = driver_I % P - 1, y = driver_I / P - 1;
        const float cx = scr_x(x + prm.exag * ux[driver_I]);
        const float cy = scr_y(y + prm.exag * uy[driver_I]);
        const int r = (int)fmax(5.0, 0.7 * cell);
        set_col(COL_MARKER);
        outline_rect((int)cx - r, (int)cy - r, 2 * r, 2 * r);
        if (drive_pol == 0) SDL_RenderDrawLine(ren, (int)cx - 2 * r, (int)cy, (int)cx + 2 * r, (int)cy);
        else                SDL_RenderDrawLine(ren, (int)cx, (int)cy - 2 * r, (int)cx, (int)cy + 2 * r);
    }
    if (pk_track) {
        const double t = sim_time - pk_t0;
        const double vg = hypot(pk_vgx, pk_vgy);
        if (t * vg < 1.5 * N) {
            double px = pk_x0 + pk_vgx * t, py = pk_y0 + pk_vgy * t;
            if (bc == BC_PERIODIC) {
                px = fmod(px, N); if (px < 0) px += N;
                py = fmod(py, N); if (py < 0) py += N;
            }
            const int cx = (int)scr_x(px), cy = (int)scr_y(py);
            set_rgba(COL_MARKER.r, COL_MARKER.g, COL_MARKER.b, 200);
            SDL_RenderDrawLine(ren, cx - 12, cy, cx - 4, cy);
            SDL_RenderDrawLine(ren, cx + 4, cy, cx + 12, cy);
            SDL_RenderDrawLine(ren, cx, cy - 12, cx, cy - 4);
            SDL_RenderDrawLine(ren, cx, cy + 4, cx, cy + 12);
        }
    }
    SDL_RenderSetClipRect(ren, NULL);

    if (recording) {
        set_col(COL_REC);
        fill_rect(LAT_X + 8, LAT_Y + 8, 14, 14);
        textf(font_mono, COL_REC, LAT_X + 28, LAT_Y + 6, "REC  %s", video_name);
    }
    if (paused)
        text_draw(font_title, COL_MARKER, LAT_X + LAT_PX - 100, LAT_Y + 6, "PAUSED");
}

/* ===================================================================
 * Dispersion panel (+ S(k, omega) underlay)
 * =================================================================== */

static int disp_px_x(double s) { return DISP_X + (int)lround(s / 3.0 * DISP_W); }
static int disp_px_y(double w) { return DISP_Y + DISP_H - (int)lround(w / omega_plot * DISP_H); }

static void s_texture_update(void)
{
    const double dw = s_domega();
    int nb = (int)ceil(omega_plot / dw) + 1;
    if (nb > SW_LEN / 2) nb = SW_LEN / 2;
    if (nb < 2) nb = 2;
    if (!s_tex || s_tex_w != s_np || s_tex_h != nb) {
        if (s_tex) SDL_DestroyTexture(s_tex);
        s_tex = SDL_CreateTexture(ren, SDL_PIXELFORMAT_ARGB8888,
                                  SDL_TEXTUREACCESS_STREAMING, s_np, nb);
        s_tex_w = s_np; s_tex_h = nb;
        s_tex_dirty = true;
    }
    if (!s_tex || !s_tex_dirty) return;
    void *pix; int pitch;
    if (SDL_LockTexture(s_tex, NULL, &pix, &pitch) != 0) return;
    const int H = SW_LEN / 2;
    for (int p = 0; p < s_np; ++p) {
        const double *acc = &s_acc[(size_t)p * H];
        double cmax = 0.0;
        for (int j = 1; j < nb; ++j) if (acc[j] > cmax) cmax = acc[j];
        const bool gamma_col = (p == 0 || p == s_np - 1);
        for (int j = 0; j < nb; ++j) {
            uint32_t *row = (uint32_t *)((uint8_t *)pix + (size_t)(nb - 1 - j) * pitch);
            uint32_t v = 0xFF000000u;
            if (!gamma_col && cmax > 0.0 && s_nacc > 0 && j > 0) {
                const double r = acc[j] / cmax;
                const double t = clampd(1.0 + log10(fmax(r, 1e-4)) / 3.0, 0.0, 1.0);
                const SDL_Color c = lut_s[(int)lround(t * 255.0)];
                v |= ((uint32_t)c.r << 16) | ((uint32_t)c.g << 8) | c.b;
            }
            row[p] = v;
        }
    }
    SDL_UnlockTexture(s_tex);
    s_tex_dirty = false;
}

static void draw_dispersion(void)
{
    text_draw(font_title, COL_TITLE, PANEL_X + 20, 12,
              s_show ? "Dispersion  \xCF\x89(k)  over measured S(k,\xCF\x89)"
                     : "Dispersion  \xCF\x89(k)");
    SDL_Rect plot = { DISP_X, DISP_Y, DISP_W, DISP_H };
    set_rgba(10, 10, 14, 255);
    SDL_RenderFillRect(ren, &plot);

    if (s_show) {
        s_texture_update();
        if (s_tex && s_nacc > 0) {
            const double dw = s_domega();
            const double half = 0.5 * DISP_W / (s_np - 1);
            const double top_w = (s_tex_h - 0.5) * dw;
            SDL_Rect d;
            d.x = (int)lround(DISP_X - half);
            d.w = (int)lround(DISP_W + 2.0 * half);
            d.y = disp_px_y(top_w);
            d.h = (int)lround(s_tex_h * dw / omega_plot * DISP_H);
            SDL_RenderSetClipRect(ren, &plot);
            SDL_RenderCopy(ren, s_tex, NULL, &d);
            SDL_RenderSetClipRect(ren, NULL);
        }
        if (s_nacc == 0)
            textf(font_mono, COL_DIM, DISP_X + 8, DISP_Y + DISP_H - 20,
                  "S(k,\xCF\x89): collecting %d%%  (needs motion: T, poke, excite)",
                  100 * s_fill / SW_LEN);
        if (omega_plot > s_nyquist())
            textf(font_mono, COL_REC, DISP_X + 8, DISP_Y + 22,
                  "\xCF\x89 above Nyquist %.1f is aliased", s_nyquist());
    }

    /* grid + symmetry lines */
    set_rgba(70, 70, 80, 255);
    for (int k = 1; k < 3; ++k) {
        const int xx = disp_px_x(k);
        SDL_RenderDrawLine(ren, xx, DISP_Y, xx, DISP_Y + DISP_H);
    }
    set_rgba(170, 170, 175, 255);
    SDL_RenderDrawLine(ren, DISP_X, DISP_Y + DISP_H, DISP_X + DISP_W, DISP_Y + DISP_H);
    SDL_RenderDrawLine(ren, DISP_X, DISP_Y, DISP_X, DISP_Y + DISP_H);

    /* analytic branches */
    SDL_Point pts[DISP_PTS];
    for (int b = 0; b < disp_nb; ++b) {
        for (int p = 0; p < DISP_PTS; ++p) {
            pts[p].x = DISP_X + (int)lround(p * (double)DISP_W / (DISP_PTS - 1));
            pts[p].y = disp_px_y(disp_w[b][p]);
        }
        set_col(COL_BRANCH[b]);
        SDL_RenderDrawLines(ren, pts, DISP_PTS);
        if (s_show) {   /* thicken over the heat map */
            for (int p = 0; p < DISP_PTS; ++p) pts[p].y -= 1;
            SDL_RenderDrawLines(ren, pts, DISP_PTS);
        }
    }

    /* driver frequency */
    if (driver_I >= 0) {
        const int yy = disp_px_y(prm.drive_w);
        set_col(COL_MARKER);
        dashed_hline(DISP_X, DISP_X + DISP_W, yy, 5);
        textf(font_mono, COL_MARKER, DISP_X + DISP_W - 70, yy - 15, "\xCF\x89_drive");
    }

    /* selected k marker, group-velocity tangent */
    double s;
    if (k_path_pos(sel_nx, sel_ny, &s)) {
        const int xm = disp_px_x(s);
        set_rgba(COL_MARKER.r, COL_MARKER.g, COL_MARKER.b, 160);
        dotted_vline(xm, DISP_Y, DISP_Y + DISP_H);
        double kx, ky;
        path_k(s, &kx, &ky);
        const double w0 = omega_at(kx, ky, sel_b);
        const double ds = 0.004;
        double s1 = s - ds, s2 = s + ds;
        if (s1 < 0) s1 = 0;
        if (s2 > 3) s2 = 3;
        if (floor(s1) != floor(s) && s > 0.0 && fmod(s, 1.0) == 0.0) s1 = s;
        double k1x, k1y, k2x, k2y;
        path_k(s1, &k1x, &k1y); path_k(s2, &k2x, &k2y);
        const double slope = (omega_at(k2x, k2y, sel_b) - omega_at(k1x, k1y, sel_b)) / (s2 - s1);
        const double L = 0.14;
        set_col(COL_MARKER);
        SDL_RenderSetClipRect(ren, &plot);
        SDL_RenderDrawLine(ren, disp_px_x(s - L), disp_px_y(w0 - slope * L),
                                disp_px_x(s + L), disp_px_y(w0 + slope * L));
        fill_rect(xm - 3, disp_px_y(w0) - 3, 7, 7);
        SDL_RenderSetClipRect(ren, NULL);
    }

    /* labels */
    static const char *lab[] = { "\xCE\x93", "X", "M", "\xCE\x93" };
    for (int k = 0; k < 4; ++k)
        text_draw(font_mono, COL_DIM, disp_px_x(k) - 4, DISP_Y + DISP_H + 4, lab[k]);
    for (int k = 0; k <= 2; ++k) {
        const double w = omega_plot * k / 2.0;
        textf(font_mono, COL_DIM, DISP_X - 42, disp_px_y(w) - 8, "%5.2f", w);
    }
    text_draw(font_mono, COL_DIM, DISP_X - 42, DISP_Y + DISP_H + 4, "  \xCF\x89");
    for (int b = 0; b < disp_nb; ++b)
        textf(font_mono, COL_BRANCH[b], DISP_X + DISP_W - 40 * (disp_nb - b), DISP_Y + 4, "b%d", b + 1);
    textf(font_mono, COL_DIM, DISP_X + 6, DISP_Y + 4, "\xCF\x89_max = %.3f", omega_top);
}

/* ===================================================================
 * Brillouin-zone mode-energy map
 * =================================================================== */

static void draw_bz(void)
{
    text_draw(font_title, COL_TITLE, BZ_X, 12, "Mode energy E(k)");
    if (!bz_tex || bz_tex_n != N) {
        if (bz_tex) SDL_DestroyTexture(bz_tex);
        SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "nearest");
        bz_tex = SDL_CreateTexture(ren, SDL_PIXELFORMAT_ARGB8888,
                                   SDL_TEXTUREACCESS_STREAMING, N, N);
        bz_tex_n = N;
#if SDL_VERSION_ATLEAST(2, 0, 12)
        if (bz_tex) SDL_SetTextureScaleMode(bz_tex, SDL_ScaleModeNearest);
#endif
    }
    if (bz_tex) {
        double emax = 0.0;
        for (int k = 1; k < NN; ++k) if (emap[k] > emax) emax = emap[k];
        void *pix; int pitch;
        if (SDL_LockTexture(bz_tex, NULL, &pix, &pitch) == 0) {
            for (int r = 0; r < N; ++r) {
                uint32_t *row = (uint32_t *)((uint8_t *)pix + (size_t)r * pitch);
                const int ny = imod(N / 2 - 1 - r, N);
                for (int c = 0; c < N; ++c) {
                    const int nx = imod(c - N / 2, N);
                    uint32_t v = 0xFF000000u;
                    if (emax > 0.0 && (nx || ny)) {
                        const double t = clampd(1.0 + log10(fmax(emap[ny * N + nx] / emax, 1e-5)) / 4.0, 0.0, 1.0);
                        const SDL_Color col = lut_s[(int)lround(t * 255.0)];
                        v |= ((uint32_t)col.r << 16) | ((uint32_t)col.g << 8) | col.b;
                    }
                    row[c] = v;
                }
            }
            SDL_UnlockTexture(bz_tex);
        }
        SDL_Rect d = { BZ_X, BZ_Y, BZ_S, BZ_S };
        SDL_RenderCopy(ren, bz_tex, NULL, &d);
    }
    const double cs = (double)BZ_S / N;
    const int gx = BZ_X + (int)lround((N / 2 + 0.5) * cs);
    const int gy = BZ_Y + (int)lround((N / 2 - 0.5) * cs);
    set_rgba(90, 90, 100, 255);
    outline_rect(BZ_X, BZ_Y, BZ_S, BZ_S);
    set_rgba(110, 110, 125, 150);
    SDL_RenderDrawLine(ren, gx, gy, BZ_X + BZ_S, gy);                 /* Gamma-X */
    SDL_RenderDrawLine(ren, BZ_X + BZ_S - 1, gy, BZ_X + BZ_S - 1, BZ_Y);  /* X-M     */
    SDL_RenderDrawLine(ren, gx, gy, BZ_X + BZ_S, BZ_Y);               /* M-Gamma */
    if (is_diatomic()) {    /* reduced zone of the checkerboard */
        set_rgba(120, 210, 110, 140);
        SDL_RenderDrawLine(ren, gx, BZ_Y, BZ_X + BZ_S, gy);
        SDL_RenderDrawLine(ren, BZ_X + BZ_S, gy, gx, BZ_Y + BZ_S);
        SDL_RenderDrawLine(ren, gx, BZ_Y + BZ_S, BZ_X, gy);
        SDL_RenderDrawLine(ren, BZ_X, gy, gx, BZ_Y);
    }
    const int sc = imod(sel_nx + N / 2, N), sr = imod(N / 2 - 1 - sel_ny, N);
    const int bx = BZ_X + (int)(sc * cs), by = BZ_Y + (int)(sr * cs);
    const int bs = (int)fmax(4.0, cs);
    set_col(COL_MARKER);
    outline_rect(bx - (bs > 4 ? 0 : 1), by - (bs > 4 ? 0 : 1), bs + 1, bs + 1);

    double w[4], ev[4][4];
    const int nb = phonons_at(2.0 * M_PI * sel_nx / N, 2.0 * M_PI * sel_ny / N, w, ev);
    textf(font_mono, COL_FG, BZ_X, BZ_Y + BZ_S + 6, "k=(%d,%d)\xC2\xB7" "2\xCF\x80/N  b%d",
          signed_n(sel_nx), signed_n(sel_ny), sel_b + 1);
    char buf[96]; int o = snprintf(buf, sizeof buf, "\xCF\x89 =");
    for (int b = 0; b < nb && o < (int)sizeof buf - 8; ++b) o += snprintf(buf + o, sizeof buf - o, " %.2f", w[b]);
    text_draw(font_mono, COL_DIM, BZ_X, BZ_Y + BZ_S + 22, buf);
}

/* ===================================================================
 * Tabs + sliders
 * =================================================================== */

typedef struct {
    const char *label; double *v; double lo, hi; int tab;
    bool is_int, is_log; int dec; bool snap; double snapv;
} Slider;

static Slider sliders[] = {
    { "K_L    nearest-neighbour longitudinal spring", &prm.KL, 0.10, 4.0, TAB_LATTICE, false, false, 2, false, 0 },
    { "K_T    nearest-neighbour transverse spring",   &prm.KT, 0.00, 2.0, TAB_LATTICE, false, false, 2, false, 0 },
    { "K_D    diagonal (next-nearest) central spring", &prm.KD, 0.00, 2.0, TAB_LATTICE, false, false, 2, false, 0 },
    { "\xCE\xB1      FPU cubic anharmonicity (NN bonds)",  &prm.alpha, -2.0, 2.0, TAB_LATTICE, false, false, 2, true, 0.0 },
    { "\xCE\xB2      FPU quartic anharmonicity (NN bonds)", &prm.beta, 0.0, 10.0, TAB_LATTICE, false, false, 2, false, 0 },
    { "m_B/m_A   checkerboard two-atom basis",       &prm.mratio, 0.25, 4.0, TAB_LATTICE, false, true, 2, true, 1.0 },
    { "disorder   random mass spread \xC2\xB1",           &prm.disorder, 0.0, 0.8, TAB_LATTICE, false, false, 2, false, 0 },

    { "amp    excitation / poke amplitude",          &prm.amp, 0.0, 2.0, TAB_EXCITE, false, false, 2, false, 0 },
    { "\xCF\x83_packet   wave-packet width (cells)",       &prm.pk_sigma, 1.0, 32.0, TAB_EXCITE, false, true, 1, false, 0 },
    { "\xCF\x83_poke   poke radius (cells)",               &prm.poke_sigma, 0.6, 8.0, TAB_EXCITE, false, true, 2, false, 0 },
    { "\xCF\x89_drive   driver frequency",                 &prm.drive_w, 0.0, 8.0, TAB_EXCITE, false, false, 3, false, 0 },
    { "F_drive   driver force amplitude",            &prm.drive_F, 0.0, 2.0, TAB_EXCITE, false, false, 2, false, 0 },

    { "kT    target (T key, global bath)",           &prm.kT, 0.0, 2.0, TAB_THERMAL, false, false, 3, false, 0 },
    { "kT_hot    hot wall",                           &prm.kT_hot, 0.0, 2.0, TAB_THERMAL, false, false, 2, false, 0 },
    { "kT_cold   cold wall",                          &prm.kT_cold, 0.0, 2.0, TAB_THERMAL, false, false, 2, false, 0 },
    { "\xCE\xB3      bath friction",                        &prm.gamma, 0.01, 2.0, TAB_THERMAL, false, true, 3, false, 0 },

    { "exaggerate    displacement drawing factor",   &prm.exag, 0.5, 40.0, TAB_VIEW, false, true, 1, false, 0 },
    { "colour gain",                                  &prm.gain, 0.1, 10.0, TAB_VIEW, false, true, 2, true, 1.0 },
    { "steps / frame    simulation speed",            &prm.speed, 1.0, 40.0, TAB_VIEW, true, false, 0, false, 0 },
};
#define N_SLIDERS (int)(sizeof sliders / sizeof sliders[0])

static int slider_row(int i)
{
    int r = 0;
    for (int j = 0; j < i; ++j) if (sliders[j].tab == sliders[i].tab) ++r;
    return r;
}

static int slider_y(int i) { return SL_Y0 + slider_row(i) * SL_DY; }

static double slider_t(const Slider *s)
{
    double t = s->is_log ? log(*s->v / s->lo) / log(s->hi / s->lo)
                         : (*s->v - s->lo) / (s->hi - s->lo);
    return clampd(t, 0.0, 1.0);
}

static void slider_set_t(Slider *s, double t)
{
    t = clampd(t, 0.0, 1.0);
    double v = s->is_log ? s->lo * pow(s->hi / s->lo, t) : s->lo + t * (s->hi - s->lo);
    if (s->is_int) v = floor(v + 0.5);
    if (s->snap) {
        const double tol = 0.012 * (s->is_log ? 1.0 : (s->hi - s->lo));
        const double d = s->is_log ? fabs(log(v / s->snapv)) * (1.0 / log(s->hi / s->lo)) * 1.0
                                   : fabs(v - s->snapv);
        if (d < tol) v = s->snapv;
    }
    *s->v = v;
}

static bool slider_hit(int i, int mx, int my)
{
    const int y = slider_y(i);
    return sliders[i].tab == cur_tab && mx >= SL_X - 6 && mx <= SL_X + SL_W + 6
        && my >= y - 6 && my <= y + SL_H + 6;
}

static void draw_tabs_sliders(void)
{
    for (int t = 0; t < TAB_COUNT; ++t) {
        const int x = PANEL_X + 20 + t * (TAB_W + 6);
        if (t == cur_tab) set_rgba(52, 66, 92, 255); else set_rgba(32, 32, 40, 255);
        fill_rect(x, TAB_Y, TAB_W, TAB_H);
        set_rgba(80, 80, 95, 255);
        outline_rect(x, TAB_Y, TAB_W, TAB_H);
        const int tw = text_width(font_lbl, tab_names[t]);
        text_draw(font_lbl, t == cur_tab ? COL_TITLE : COL_DIM,
                  x + (TAB_W - tw) / 2, TAB_Y + 4, tab_names[t]);
    }
    for (int i = 0; i < N_SLIDERS; ++i) {
        const Slider *s = &sliders[i];
        if (s->tab != cur_tab) continue;
        const int y = slider_y(i);
        text_draw(font_lbl, COL_FG, SL_X, y - 18, s->label);
        char val[32];
        if (s->is_int) snprintf(val, sizeof val, "%d", (int)*s->v);
        else           snprintf(val, sizeof val, "%.*f", s->dec, *s->v);
        text_draw(font_mono, COL_KEY, SL_X + SL_W - text_width(font_mono, val), y - 17, val);
        const double t = slider_t(s);
        set_rgba(48, 48, 58, 255);
        fill_rect(SL_X, y + SL_H / 2 - 3, SL_W, 6);
        set_rgba(95, 155, 215, 255);
        fill_rect(SL_X, y + SL_H / 2 - 3, (int)(t * SL_W), 6);
        set_rgba(240, 240, 240, 255);
        fill_rect(SL_X + (int)(t * SL_W) - 4, y + SL_H / 2 - 8, 8, 16);
    }
}

/* ===================================================================
 * Lower panel: time series / T(x) / C_v(T)
 * =================================================================== */

static void plot_frame(int x, int y, int w, int h)
{
    set_rgba(10, 10, 14, 255);
    fill_rect(x, y, w, h);
    set_rgba(90, 90, 100, 255);
    outline_rect(x, y, w, h);
}

static void draw_series(const float *ring, int x, int y, int w, int h,
                        double lo, double hi, SDL_Color c)
{
    if (ts_n < 2 || hi <= lo) return;
    static SDL_Point pts[TS_LEN];
    const int start = (ts_head - ts_n + TS_LEN) % TS_LEN;
    for (int i = 0; i < ts_n; ++i) {
        const double v = ring[(start + i) % TS_LEN];
        pts[i].x = x + (int)lround((double)i * (w - 1) / (TS_LEN - 1));
        pts[i].y = y + h - 1 - (int)lround(clampd((v - lo) / (hi - lo), 0.0, 1.0) * (h - 2));
    }
    set_col(c);
    SDL_RenderDrawLines(ren, pts, ts_n);
}

static void draw_lower(void)
{
    textf(font_title, COL_TITLE, PANEL_X + 20, DIAG_Y, "%s", low_names[low_mode]);
    text_draw(font_mono, COL_DIM, LOW_X + LOW_W - 150, DIAG_Y + 4, "[L / click: next]");

    if (low_mode == LOW_TS) {
        const int h1 = 78, h2 = LOW_H - h1 - 12;
        const int y1 = LOW_Y, y2 = LOW_Y + h1 + 12;
        plot_frame(LOW_X, y1, LOW_W, h1);
        plot_frame(LOW_X, y2, LOW_W, h2);
        double emax = 1e-12;
        const int start = (ts_head - ts_n + TS_LEN) % TS_LEN;
        for (int i = 0; i < ts_n; ++i) {
            const int j = (start + i) % TS_LEN;
            emax = fmax(emax, fmax(ts_E[j], fmax(ts_KE[j], ts_PE[j])));
        }
        static const SDL_Color cK = { 90, 200, 240, 255 }, cP = { 240, 140, 70, 255 };
        draw_series(ts_E,  LOW_X, y1, LOW_W, h1, 0.0, 1.1 * emax, COL_FG);
        draw_series(ts_KE, LOW_X, y1, LOW_W, h1, 0.0, 1.1 * emax, cK);
        draw_series(ts_PE, LOW_X, y1, LOW_W, h1, 0.0, 1.1 * emax, cP);
        set_rgba(10, 10, 14, 210);
        fill_rect(LOW_X + 2, y1 + 2, 72, 16);
        fill_rect(LOW_X + LOW_W - 114, y1 + 2, 112, 16);
        text_draw(font_mono, COL_FG, LOW_X + 6, y1 + 3, "E");
        text_draw(font_mono, cK, LOW_X + 22, y1 + 3, "KE");
        text_draw(font_mono, cP, LOW_X + 46, y1 + 3, "PE");
        textf(font_mono, COL_DIM, LOW_X + LOW_W - 110, y1 + 3, "max %.3g", 1.1 * emax);

        /* mode energies E(m k) / E(t) */
        static float frac[4][TS_LEN];
        for (int i = 0; i < TS_LEN; ++i) {
            const double e = ts_E[i] > 1e-30f ? ts_E[i] : 1e-30;
            for (int m = 0; m < 4; ++m) frac[m][i] = (float)(ts_m[m][i] / e);
        }
        for (int m = 3; m >= 0; --m)
            draw_series(frac[m], LOW_X, y2, LOW_W, h2, 0.0, 1.05, COL_MODE[m]);
        static const char *ml[4] = { "E(k)", "E(2k)", "E(3k)", "E(4k)" };
        set_rgba(10, 10, 14, 210);
        fill_rect(LOW_X + 2, y2 + 2, 206, 16);
        for (int m = 0; m < 4; ++m)
            text_draw(font_mono, COL_MODE[m], LOW_X + 6 + 52 * m, y2 + 3, ml[m]);
        text_draw(font_mono, COL_DIM, LOW_X + LOW_W - 118, y2 + 3, "fraction of E");
        textf(font_mono, COL_DIM, LOW_X, LOW_Y + LOW_H + 4,
              "last %.0f time units; k = selected k", TS_LEN * prm.speed * DT);
    } else if (low_mode == LOW_PROFILE) {
        plot_frame(LOW_X, LOW_Y, LOW_W, LOW_H);
        double tmax = fmax(prm.kT_hot, prm.kT_cold);
        for (int x = 0; x < N; ++x) tmax = fmax(tmax, prof_T[x]);
        tmax = 1.15 * fmax(tmax, 1e-3);
        const double cw = (double)LOW_W / N;
        if (thermo == TH_WALLS) {
            for (int x = 0; x < N; ++x) {
                const int b = bath_of_col(x);
                if (!b) continue;
                if (b == 1) set_rgba(COL_HOT.r, COL_HOT.g, COL_HOT.b, 50);
                else        set_rgba(COL_COLD.r, COL_COLD.g, COL_COLD.b, 50);
                fill_rect(LOW_X + (int)(x * cw), LOW_Y + 1, (int)ceil(cw), LOW_H - 2);
            }
            set_col(COL_HOT);
            dashed_hline(LOW_X, LOW_X + LOW_W, LOW_Y + LOW_H - (int)(prm.kT_hot / tmax * LOW_H), 4);
            set_col(COL_COLD);
            dashed_hline(LOW_X, LOW_X + LOW_W, LOW_Y + LOW_H - (int)(prm.kT_cold / tmax * LOW_H), 4);
        }
        SDL_Point pts[NMAX];
        for (int x = 0; x < N; ++x) {
            pts[x].x = LOW_X + (int)((x + 0.5) * cw);
            pts[x].y = LOW_Y + LOW_H - 1 - (int)(clampd(prof_T[x] / tmax, 0.0, 1.0) * (LOW_H - 2));
        }
        set_col(COL_FG);
        SDL_RenderDrawLines(ren, pts, N);
        if (N <= 128) for (int x = 0; x < N; ++x) fill_rect(pts[x].x - 1, pts[x].y - 1, 3, 3);
        textf(font_mono, COL_DIM, LOW_X + 6, LOW_Y + 3, "max %.3f", tmax);
        text_draw(font_mono, COL_DIM, LOW_X, LOW_Y + LOW_H + 4,
                  thermo == TH_WALLS ? "x \xE2\x86\x92   (column-averaged kT, slow EMA)"
                                     : "x \xE2\x86\x92   (H twice: hot/cold walls)");
    } else {
        plot_frame(LOW_X, LOW_Y, LOW_W, LOW_H);
        if (cv_dirty) {
            text_draw(font_mono, COL_DIM, LOW_X + 10, LOW_Y + 10, "recomputing\xE2\x80\xA6");
            return;
        }
        const double lx0 = log10(cv_T[0]), lx1 = log10(cv_T[CV_PTS - 1]);
        const double ly0 = -4.0, ly1 = 0.25;
#define CVX(lt) (LOW_X + (int)lround(((lt) - lx0) / (lx1 - lx0) * (LOW_W - 1)))
#define CVY(lc) (LOW_Y + LOW_H - 1 - (int)lround(clampd(((lc) - ly0) / (ly1 - ly0), 0.0, 1.0) * (LOW_H - 2)))
        set_rgba(55, 55, 65, 255);
        for (int d = (int)ceil(lx0); d <= (int)floor(lx1); ++d)
            SDL_RenderDrawLine(ren, CVX(d), LOW_Y, CVX(d), LOW_Y + LOW_H - 1);
        for (int d = -4; d <= 0; ++d)
            SDL_RenderDrawLine(ren, LOW_X, CVY(d), LOW_X + LOW_W - 1, CVY(d));
        set_rgba(150, 150, 160, 255);
        dashed_hline(LOW_X, LOW_X + LOW_W, CVY(0.0), 5);
        /* T^2 reference through the curve at T = 0.03 omega_top */
        const double tr = 0.03 * omega_top;
        const double cr = cv_at(tr) / cv_ndp;
        if (cr > 0.0) {
            const double lt = log10(tr), lc = log10(cr);
            const double a0 = lx0, a1 = lt + 0.6;
            for (double a = a0; a < a1; a += 0.04)
                SDL_RenderDrawPoint(ren, CVX(a), CVY(lc + 2.0 * (a - lt)));
            text_draw(font_mono, COL_DIM, CVX(lt) - 46, CVY(lc + 2.0 * 0.0) - 2, "T\xC2\xB2");
        }
        SDL_Point pts[CV_PTS];
        int np = 0;
        for (int j = 0; j < CV_PTS; ++j) {
            if (cv_C[j] <= 0.0) continue;
            pts[np].x = CVX(log10(cv_T[j]));
            pts[np].y = CVY(log10(cv_C[j]));
            ++np;
        }
        set_col(COL_BRANCH[1]);
        if (np > 1) SDL_RenderDrawLines(ren, pts, np);
        if (prm.kT > 0.0) {
            const double lt = log10(prm.kT);
            if (lt > lx0 && lt < lx1) {
                set_col(COL_MARKER);
                dotted_vline(CVX(lt), LOW_Y, LOW_Y + LOW_H);
                const double c = cv_at(prm.kT) / cv_ndp;
                if (c > 0) fill_rect(CVX(lt) - 3, CVY(log10(c)) - 3, 7, 7);
            }
        }
        text_draw(font_mono, COL_DIM, LOW_X + 6, CVY(0.0) + 3, "Dulong\xE2\x80\x93Petit (classical MD)");
        textf(font_mono, COL_DIM, LOW_X, LOW_Y + LOW_H + 4,
              "T = 10^%.0f \xE2\x80\xA6 10^%.0f;  C/C_DP = 10^-4 \xE2\x80\xA6 1;  marker = kT",
              lx0, lx1);
#undef CVX
#undef CVY
    }
}

/* ===================================================================
 * Diagnostics, status bar, toast, help
 * =================================================================== */

static void draw_diag(void)
{
    text_draw(font_title, COL_TITLE, DIAG_X, DIAG_Y, "Diagnostics");
    int y = DIAG_Y + 26;
    const double E = d_KE + d_PE;
#define DL(col, ...) do { textf(font_mono, col, DIAG_X, y, __VA_ARGS__); y += LINE_DY; } while (0)
    DL(COL_FG, "t        %10.2f", sim_time);
    DL(COL_FG, "E_tot    %10.4f", E);
    if (system_is_open()) DL(COL_DIM, "\xCE\x94" "E/E0    open system");
    else DL(COL_FG, "\xCE\x94" "E/E0    %10.2e", fabs(E_ref) > 1e-12 ? (E - E_ref) / fabs(E_ref) : 0.0);
    DL(COL_FG, "KE / PE  %10.4f", d_PE > 1e-12 ? d_KE / d_PE : 0.0);
    DL(COL_FG, "kT_eff   %10.4f", d_kTeff);
    if (!cv_dirty) DL(COL_FG, "C_v(kT)  %8.3f C_DP", cv_ndp > 0 ? d_cv / cv_ndp : 0.0);
    else           DL(COL_DIM, "C_v(kT)  \xE2\x80\xA6");
    {
        double vgx, vgy;
        group_velocity(2.0 * M_PI * sel_nx / N, 2.0 * M_PI * sel_ny / N, sel_b, &vgx, &vgy);
        DL(COL_FG, "v_g(b%d)  (%+.2f,%+.2f)", sel_b + 1, vgx, vgy);
    }
    if (thermo == TH_WALLS) {
        DL(COL_HOT,  "J_in     %+10.4f", J_in);
        DL(COL_COLD, "J_out    %+10.4f", J_out);
    } else {
        DL(COL_FG, "max|u|   %10.4f", d_maxu);
    }
    if (s_nacc > 0) DL(COL_FG, "S(k,\xCF\x89)   %3d windows", s_nacc);
    else            DL(COL_DIM, "S(k,\xCF\x89)   fill %3d%%", 100 * s_fill / SW_LEN);
    DL(COL_DIM, "atoms    %10d", d_nalive);
    DL(COL_DIM, "fps %4.0f  %6.0f steps/s", fps, fps * prm.speed);
#undef DL
}

static void draw_status(void)
{
    static const char *th_short[] = { "off", "global", "walls" };
    static const char *sh_short[] = { "standing", "travelling", "packet" };
    char a[256], b[256];
    snprintf(a, sizeof a,
             "tool [Tab]: %s   field [D]: %s   style [G]: %s   colour [C]: %s",
             tool_names[tool], field_names[field], style_names[style], cs_names[cscale]);
    snprintf(b, sizeof b,
             "BC [B]: %s   bath [H]: %s   excite [E]: %s%s   N [N]: %d   F1 help  F2\xE2\x80\x93" "F10 demos",
             bc_names[bc], th_short[thermo], sh_short[shape], additive ? "+add" : "", N);
    set_rgba(60, 60, 70, 255);
    SDL_RenderDrawLine(ren, PANEL_X, STAT_Y - 8, PANEL_R, STAT_Y - 8);
    text_draw(font_mono, COL_FG, PANEL_X + 4, STAT_Y, a);
    text_draw(font_mono, COL_FG, PANEL_X + 4, STAT_Y + 18, b);
}

static void draw_toast(void)
{
    if (!toast_msg[0] || now_ms > toast_until) return;
    char tmp[sizeof toast_msg];
    snprintf(tmp, sizeof tmp, "%s", toast_msg);
    char *l2 = strchr(tmp, '\n');
    if (l2) *l2++ = '\0';
    const int w1 = text_width(font_lbl, tmp), w2 = l2 ? text_width(font_lbl, l2) : 0;
    const int w = (w1 > w2 ? w1 : w2) + 24, h = l2 ? 46 : 28;
    const int x = LAT_X + (LAT_PX - w) / 2, y = LAT_Y + LAT_PX - h - 12;
    set_rgba(0, 0, 0, 190);
    fill_rect(x, y, w, h);
    set_rgba(110, 110, 125, 255);
    outline_rect(x, y, w, h);
    text_draw(font_lbl, COL_TITLE, x + 12, y + 6, tmp);
    if (l2) text_draw(font_lbl, COL_TITLE, x + 12, y + 24, l2);
}

static void draw_help(void)
{
    static const char *keys[][2] = {
        { "Space / .",   "pause-resume / single step while paused" },
        { "1 2 3 4",     "excite branch b at the selected k (shape: E)" },
        { "Enter",       "excite the selected branch again" },
        { "Arrows",      "move the selected k in the Brillouin zone" },
        { "E",           "shape: standing / travelling wave / wave packet" },
        { "A",           "additive excitations on/off" },
        { "T",           "thermalise: Maxwell\xE2\x80\x93" "Boltzmann velocities at kT" },
        { "H",           "bath: off / global Langevin / hot-cold walls" },
        { "B",           "boundary: periodic / fixed / free / absorbing" },
        { "D, Shift+D",  "colour field: |u| u_x u_y |v| KE div curl mass" },
        { "G",           "style: atoms / bonds (strain) / heatmap" },
        { "C",           "colour scale: auto (slow) / locked / fixed" },
        { "S, Shift+S",  "measured S(k,\xCF\x89) underlay on/off / clear it" },
        { "L",           "lower plot: time series / T(x) / C_v(T)" },
        { "Tab",         "mouse tool: poke heavy light vacancy driver erase" },
        { "P",           "driver polarisation x / y" },
        { "N, Shift+N",  "lattice size up / down (32 \xE2\x80\xA6 256)" },
        { "R, Shift+R",  "reset motion / reset everything" },
        { "V   I",       "record mp4 (ffmpeg) / save BMP screenshot" },
        { "F1  ?",       "toggle this help        Q, Esc: quit" },
    };
    static const char *mouse[][2] = {
        { "lattice L",    "tool; poke: click = radial pulse, drag = push" },
        { "lattice R",    "poke, whatever the tool" },
        { "dispersion L", "excite nearest branch at that k" },
        { "dispersion R", "set driver \xCF\x89 (places a driver if none)" },
        { "BZ map L / R", "select k and excite / select only" },
        { "sliders",      "drag, or mouse wheel to nudge" },
    };
    static const char *demos[] = {
        "F2 packet", "F3 focusing", "F4 fput", "F5 breathers", "F6 impurity",
        "F7 disorder", "F8 heat", "F9 diatomic", "F10 driven"
    };
    set_rgba(0, 0, 0, 215);
    fill_rect(0, 0, WIN_W, WIN_H);
    const int x0 = 120, x1 = 300;
    int y = 60;
    text_draw(font_title, COL_TITLE, x0, y, "crystal_phonons " PHONONS_VERSION "  \xE2\x80\x94  keys");
    y += 34;
    for (size_t i = 0; i < sizeof keys / sizeof keys[0]; ++i, y += 22) {
        text_draw(font_mono, COL_KEY, x0, y, keys[i][0]);
        text_draw(font_lbl, COL_FG, x1, y - 1, keys[i][1]);
    }
    const int x2 = 860, x3 = 1000;
    y = 60;
    text_draw(font_title, COL_TITLE, x2, y, "Mouse");
    y += 34;
    for (size_t i = 0; i < sizeof mouse / sizeof mouse[0]; ++i, y += 22) {
        text_draw(font_mono, COL_KEY, x2, y, mouse[i][0]);
        text_draw(font_lbl, COL_FG, x3, y - 1, mouse[i][1]);
    }
    y += 24;
    text_draw(font_title, COL_TITLE, x2, y, "Demos");
    y += 34;
    for (size_t i = 0; i < sizeof demos / sizeof demos[0]; ++i, y += 22)
        text_draw(font_mono, COL_KEY, x2, y, demos[i]);
    text_draw(font_lbl, COL_DIM, x0, WIN_H - 60, "Press F1, ? or click to close.");
}

static void render_all(void)
{
    set_col(COL_BG);
    SDL_RenderClear(ren);
    draw_lattice();
    draw_lattice_overlays();
    set_rgba(60, 60, 70, 255);
    SDL_RenderDrawLine(ren, PANEL_X - 10, 10, PANEL_X - 10, WIN_H - 10);
    draw_dispersion();
    draw_bz();
    draw_tabs_sliders();
    draw_lower();
    draw_diag();
    draw_status();
    draw_toast();
    if (show_help) draw_help();
}

/* ===================================================================
 * Video capture and screenshots
 * =================================================================== */

static void timestamp_name(char *buf, size_t n, const char *ext)
{
    time_t t = time(NULL);
    struct tm *lt = localtime(&t);
    char stamp[64] = "now";
    if (lt) strftime(stamp, sizeof stamp, "%Y%m%d_%H%M%S", lt);
    snprintf(buf, n, "%s/phonons_%s.%s", out_dir, stamp, ext);
}

#ifndef __EMSCRIPTEN__
static void start_recording(void)
{
    if (recording) return;
    if (SDL_GetRendererOutputSize(ren, &rec_out_w, &rec_out_h) != 0) return;
    rec_w = rec_out_w & ~1; rec_h = rec_out_h & ~1;
    timestamp_name(video_name, sizeof video_name, "mp4");
    char cmd[1024];
    snprintf(cmd, sizeof cmd,
             "ffmpeg -y -f rawvideo -pix_fmt rgba -s %dx%d -r 60 -i - "
             "-c:v libx264 -preset veryfast -pix_fmt yuv420p "
             "-loglevel warning '%s' 2>/dev/null",
             rec_w, rec_h, video_name);
    video_pipe = popen(cmd, "w");
    if (!video_pipe) { toast("Recording failed: could not start ffmpeg"); return; }
    video_buf = xrealloc(video_buf, (size_t)rec_out_w * (size_t)rec_out_h * 4u);
    recording = true;
    toast("Recording to %s", video_name);
}

static void stop_recording(void)
{
    if (!recording) return;
    if (video_pipe) { pclose(video_pipe); video_pipe = NULL; }
    recording = false;
    toast("Saved %s", video_name);
}

/* Must run before SDL_RenderPresent: the back buffer is undefined after. */
static void capture_frame(void)
{
    if (!recording || !video_pipe) return;
    int w, h;
    if (SDL_GetRendererOutputSize(ren, &w, &h) != 0 || w != rec_out_w || h != rec_out_h) {
        stop_recording();
        toast("Recording stopped: window size changed");
        return;
    }
    if (SDL_RenderReadPixels(ren, NULL, SDL_PIXELFORMAT_RGBA32, video_buf, rec_out_w * 4) != 0) {
        stop_recording();
        return;
    }
    for (int r = 0; r < rec_h; ++r)
        if (fwrite(video_buf + (size_t)r * rec_out_w * 4, 1, (size_t)rec_w * 4, video_pipe)
            != (size_t)rec_w * 4) {
            stop_recording();
            return;
        }
}

static bool save_screenshot(const char *path)
{
    int w, h;
    if (SDL_GetRendererOutputSize(ren, &w, &h) != 0) return false;
    SDL_Surface *s = SDL_CreateRGBSurfaceWithFormat(0, w, h, 32, SDL_PIXELFORMAT_ARGB8888);
    if (!s) return false;
    bool ok = SDL_RenderReadPixels(ren, NULL, SDL_PIXELFORMAT_ARGB8888, s->pixels, s->pitch) == 0
           && SDL_SaveBMP(s, path) == 0;
    SDL_FreeSurface(s);
    return ok;
}
#else
static void start_recording(void) { toast("Video capture is not available in the browser build"); }
static void stop_recording(void) {}
static void capture_frame(void) {}
static bool save_screenshot(const char *path) { (void)path; return false; }
#endif

/* ===================================================================
 * Demos
 * =================================================================== */

typedef struct { const char *name; const char *desc; } DemoInfo;
static const DemoInfo demos[] = {
    { "packet",    "LA wave packet: crests (phase velocity) slide through an envelope moving at v_g.\n"
                   "The crosshair moves at the predicted group velocity d\xCF\x89/dk." },
    { "focusing",  "Point pulse: crossed elliptical fronts (LA/TA speeds depend on direction)\n"
                   "plus lattice caustics at short wavelengths." },
    { "fput",      "FPUT (\xCE\xB1 = 1): energy leaves mode k for 2k, 3k \xE2\x80\xA6 and flows back (lower plot)." },
    { "breathers", "\xCE\xB2-FPU zone-boundary mode: modulational instability breaks it\n"
                   "into localized breathers (bright spots in KE)." },
    { "impurity",  "Light impurity (m/4): its local mode lies above \xCF\x89_max,\n"
                   "so it cannot radiate and keeps ringing after the rest has gone." },
    { "disorder",  "Mass disorder 0.8: high-frequency waves stay near the driver.\n"
                   "Drag disorder (Lattice tab) to 0 to compare." },
    { "heat",      "Harmonic crystal between baths: flat T(x) (ballistic transport).\n"
                   "Raise \xCE\xB2 on the Lattice tab to get a gradient (Fourier's law)." },
    { "diatomic",  "m_B/m_A = 3: optical branches and a gap. Thermal noise fills in\n"
                   "the measured S(k,\xCF\x89) under the curves within ~10 s." },
    { "driven",    "Driver inside the band radiates. Right-click the dispersion plot\n"
                   "above \xCF\x89_max: the radiation stops (evanescent)." },
};
#define N_DEMOS (int)(sizeof demos / sizeof demos[0])

static void apply_demo(int d)
{
    if (d < 0 || d >= N_DEMOS) return;
    prm = prm_default;
    thermo = TH_OFF; bc = BC_PERIODIC; additive = false; s_show = false;
    low_mode = LOW_TS; field = F_ABSU; style = ST_ATOMS; cscale = CS_AUTO;
    shape = SH_PACKET; drive_pol = 0; paused = false;
    int n = 64;
    switch (d) {
    case 0: n = 128; break;
    case 1: n = 256; break;
    case 2: n = 32;  break;
    case 3: n = 128; break;
    case 5: n = 128; break;
    case 8: n = 128; break;
    default: break;
    }
    set_lattice_size(n);
    const int c = N / 2;
    switch (d) {
    case 0:   /* packet */
        field = F_UX; style = ST_HEAT; prm.amp = 0.3; prm.pk_sigma = 8.0;
        apply_settings(); shape = SH_PACKET;
        excite(N * 5 / 16, 0, 1);
        break;
    case 1:   /* focusing */
        prm.KL = 2.0; prm.KT = 0.15; bc = BC_ABSORB; field = F_ABSV; style = ST_HEAT;
        prm.amp = 0.5; prm.poke_sigma = 1.0;
        apply_settings();
        poke_radial(c, c);
        break;
    case 2:   /* fput */
        prm.alpha = 1.0; prm.amp = 1.0; prm.speed = 40; field = F_UX; style = ST_HEAT;
        apply_settings(); shape = SH_STANDING;
        excite(1, 0, 1);
        break;
    case 3:   /* breathers */
        prm.beta = 4.0; prm.amp = 0.12; prm.speed = 10; field = F_KE; style = ST_HEAT;
        apply_settings(); shape = SH_STANDING;
        excite(N / 2, 0, 1);
        add_disp_noise(1e-4);
        break;
    case 4:   /* impurity */
        bc = BC_ABSORB; field = F_KE; style = ST_ATOMS; prm.exag = 3.0;
        defm[ID(c, c)] = 0.25;
        apply_settings();
        ux[ID(c, c)] = 0.3; uy[ID(c, c)] = 0.2;
        compute_forces();
        break;
    case 5:   /* disorder */
        prm.disorder = 0.8; bc = BC_ABSORB; field = F_KE; style = ST_HEAT;
        prm.drive_F = 0.3;
        driver_I = ID(c, c);
        apply_settings();
        disp_update();
        prm.drive_w = 0.85 * omega_top;
        prm_last = prm;
        disp_dirty = true;
        break;
    case 6:   /* heat */
        bc = BC_FIXED; thermo = TH_WALLS; prm.kT_hot = 1.0; prm.kT_cold = 0.1;
        prm.gamma = 0.5; prm.speed = 20; low_mode = LOW_PROFILE;
        field = F_KE; style = ST_HEAT;
        apply_settings();
        thermalize(0.55);
        break;
    case 7:   /* diatomic */
        prm.mratio = 3.0; prm.speed = 20; s_show = true; field = F_ABSV; prm.exag = 0.5;
        apply_settings();
        thermalize(0.06);
        break;
    case 8:   /* driven */
        bc = BC_ABSORB; field = F_UX; style = ST_HEAT; prm.drive_w = 1.0; prm.drive_F = 0.4;
        driver_I = ID(c, c);
        apply_settings();
        break;
    default: break;
    }
    toast("Demo %d \xE2\x80\x94 %s:  %s", d + 2, demos[d].name, demos[d].desc);
}

static int demo_by_name(const char *s)
{
    for (int i = 0; i < N_DEMOS; ++i) if (!strcmp(s, demos[i].name)) return i;
    return -1;
}

/* ===================================================================
 * Input
 * =================================================================== */

static void reset_everything(void)
{
    prm = prm_default;
    thermo = TH_OFF; bc = BC_PERIODIC; additive = false; s_show = false;
    clear_defects();
    zero_motion();
    step_no = 0; sim_time = 0.0;
    ts_reset();
    apply_settings();
    toast("Reset everything to defaults");
}

static void cycle_lattice_size(int dir)
{
    static const int sizes[] = { 32, 64, 128, 256 };
    int i = 0;
    while (i < 4 && sizes[i] < N) ++i;
    if (i >= 4) i = 3;
    if (sizes[i] != N && dir < 0) { /* N not in list: go to nearest below */ }
    else i += dir;
    if (i < 0) i = 0;
    if (i > 3) i = 3;
    if (sizes[i] == N) return;
    set_lattice_size(sizes[i]);
    toast("Lattice %d \xC3\x97 %d (motion and defects cleared)", N, N);
}

static void dispersion_click(int mx, int my, bool right)
{
    const double s = clampd(3.0 * (mx - DISP_X) / (double)DISP_W, 0.0, 3.0);
    const double wc = clampd((DISP_Y + DISP_H - my) / (double)DISP_H * omega_plot, 0.0, 1e9);
    if (right) {
        prm.drive_w = clampd(wc, 0.0, 8.0);
        if (driver_I < 0) {
            driver_I = ID(N / 2, N / 2);
            alive[driver_I] = 1;
            fill_alive_ghosts();
        }
        disp_dirty = true;
        toast("Driver \xCF\x89 = %.3f  (%s band, \xCF\x89_max = %.3f)", prm.drive_w,
              prm.drive_w > omega_top ? "above the" : "inside the", omega_top);
        return;
    }
    int nx, ny;
    path_snap(s, &nx, &ny);
    double w[4], ev[4][4];
    const int nb = phonons_at(2.0 * M_PI * nx / N, 2.0 * M_PI * ny / N, w, ev);
    int best = 0;
    for (int b = 1; b < nb; ++b) if (fabs(w[b] - wc) < fabs(w[best] - wc)) best = b;
    excite(nx, ny, best);
}

static void bz_click(int mx, int my, bool right)
{
    const int c = (int)((mx - BZ_X) * N / (double)BZ_S);
    const int r = (int)((my - BZ_Y) * N / (double)BZ_S);
    sel_nx = imod(c - N / 2, N);
    sel_ny = imod(N / 2 - 1 - r, N);
    if (sel_b >= n_branches()) sel_b = n_branches() - 1;
    if (!right) excite(sel_nx, sel_ny, sel_b);
}

static void handle_key(const SDL_KeyboardEvent *k)
{
    const bool shift = (k->keysym.mod & KMOD_SHIFT) != 0;
    const SDL_Keycode sym = k->keysym.sym;
    if (show_help && sym != SDLK_q) { show_help = false; if (sym == SDLK_F1 || sym == SDLK_ESCAPE) return; }
    if (sym >= SDLK_F2 && sym <= SDLK_F10) { apply_demo((int)(sym - SDLK_F2)); return; }
    switch (sym) {
    case SDLK_q: case SDLK_ESCAPE: running = false; break;
    case SDLK_F1: show_help = !show_help; break;
    case SDLK_SLASH: if (shift) show_help = !show_help; break;
    case SDLK_QUESTION: show_help = !show_help; break;
    case SDLK_SPACE: paused = !paused; break;
    case SDLK_PERIOD: if (paused) step_once = true; break;
    case SDLK_r:
        if (shift) reset_everything();
        else { zero_motion(); step_no = 0; sim_time = 0.0; ts_reset(); s_reset(); toast("Motion reset"); }
        break;
    case SDLK_1: case SDLK_2: case SDLK_3: case SDLK_4: {
        const int b = (int)(sym - SDLK_1);
        if (b < n_branches()) excite(sel_nx, sel_ny, b);
        else toast("Only %d branches (set m_B/m_A \xE2\x89\xA0 1 for 4)", n_branches());
        break;
    }
    case SDLK_RETURN: case SDLK_KP_ENTER: excite(sel_nx, sel_ny, sel_b); break;
    case SDLK_LEFT:  sel_nx = imod(sel_nx - 1, N); break;
    case SDLK_RIGHT: sel_nx = imod(sel_nx + 1, N); break;
    case SDLK_UP:    sel_ny = imod(sel_ny + 1, N); break;
    case SDLK_DOWN:  sel_ny = imod(sel_ny - 1, N); break;
    case SDLK_e: shape = (shape + 1) % SH_COUNT; toast("Excitation shape: %s", sh_names[shape]); break;
    case SDLK_a: additive = !additive; toast("Additive excitations %s", additive ? "on" : "off"); break;
    case SDLK_t: thermalize(prm.kT); toast("Thermalised: Maxwell\xE2\x80\x93" "Boltzmann at kT = %.3f", prm.kT); break;
    case SDLK_h:
        thermo = (thermo + 1) % TH_COUNT;
        J_in = J_out = 0.0;
        if (thermo == TH_WALLS) { low_mode = LOW_PROFILE; prof_init = false; }
        toast("Bath: %s%s", th_names[thermo],
              thermo == TH_WALLS ? (bc == BC_PERIODIC ? "  (periodic: cold strip at N/2)" : "") : "");
        break;
    case SDLK_b:
        bc = (bc + 1) % BC_COUNT;
        update_sponge(); fill_alive_ghosts(); compute_forces(); E_ref_set = false;
        toast("Boundary: %s", bc_names[bc]);
        break;
    case SDLK_d: field = (field + (shift ? F_COUNT - 1 : 1)) % F_COUNT; break;
    case SDLK_g: style = (style + 1) % ST_COUNT; break;
    case SDLK_c:
        cscale = (cscale + 1) % CS_COUNT;
        if (cscale == CS_LOCK) memcpy(cs_lock, cs_ema, sizeof cs_lock);
        toast("Colour scale: %s", cs_names[cscale]);
        break;
    case SDLK_s:
        if (shift) { s_reset(); toast("S(k,\xCF\x89) cleared"); }
        else { s_show = !s_show; toast("S(k,\xCF\x89) underlay %s", s_show ? "on" : "off"); }
        break;
    case SDLK_l: low_mode = (low_mode + 1) % LOW_COUNT; break;
    case SDLK_TAB:
        tool = (tool + (shift ? TOOL_COUNT - 1 : 1)) % TOOL_COUNT;
        toast("Mouse tool: %s", tool_names[tool]);
        break;
    case SDLK_p:
        drive_pol ^= 1;
        toast("Driver polarisation: %c", drive_pol ? 'y' : 'x');
        break;
    case SDLK_n: cycle_lattice_size(shift ? -1 : 1); break;
    case SDLK_v: if (recording) stop_recording(); else start_recording(); break;
    case SDLK_i: {
        char name[512];
        timestamp_name(name, sizeof name, "bmp");
        render_all();
        if (save_screenshot(name)) toast("Saved %s", name); else toast("Screenshot failed");
        break;
    }
    default: break;
    }
}

static void handle_mouse(const SDL_Event *ev)
{
    if (ev->type == SDL_MOUSEBUTTONDOWN) {
        const int mx = ev->button.x, my = ev->button.y;
        const bool right = ev->button.button == SDL_BUTTON_RIGHT;
        if (show_help) { show_help = false; return; }
        if (!right && ev->button.button != SDL_BUTTON_LEFT) return;
        if (!right) {
            for (int t = 0; t < TAB_COUNT; ++t)
                if (in_rect(mx, my, PANEL_X + 20 + t * (TAB_W + 6), TAB_Y, TAB_W, TAB_H)) {
                    cur_tab = t; return;
                }
            for (int i = 0; i < N_SLIDERS; ++i)
                if (slider_hit(i, mx, my)) {
                    drag_slider = i;
                    slider_set_t(&sliders[i], (mx - SL_X) / (double)SL_W);
                    return;
                }
            if (in_rect(mx, my, LOW_X, LOW_Y, LOW_W, LOW_H)) { low_mode = (low_mode + 1) % LOW_COUNT; return; }
        }
        if (in_rect(mx, my, DISP_X, DISP_Y, DISP_W, DISP_H)) { dispersion_click(mx, my, right); return; }
        if (in_rect(mx, my, BZ_X, BZ_Y, BZ_S, BZ_S)) { bz_click(mx, my, right); return; }
        if (in_rect(mx, my, LAT_X, LAT_Y, LAT_PX, LAT_PX)) {
            double cx, cy;
            mouse_to_lattice(mx, my, &cx, &cy);
            last_cx = cx; last_cy = cy;
            if (right || tool == TOOL_POKE) {
                poke_radial(cx, cy);
                lat_drag = 1;
            } else {
                const int x = (int)lround(cx), y = (int)lround(cy);
                apply_tool_at(x, y);
                last_site = is_real(x, y) ? ID(x, y) : -1;
                lat_drag = (tool == TOOL_DRIVER) ? 0 : 2;
            }
        }
    } else if (ev->type == SDL_MOUSEMOTION) {
        const int mx = ev->motion.x, my = ev->motion.y;
        if (drag_slider >= 0) {
            slider_set_t(&sliders[drag_slider], (mx - SL_X) / (double)SL_W);
        } else if (lat_drag) {
            double cx, cy;
            mouse_to_lattice(mx, my, &cx, &cy);
            if (lat_drag == 1) {
                poke_impulse(cx, cy, 0.3 * (cx - last_cx) * 10.0 / fmax(1.0, N / 16.0),
                                     0.3 * (cy - last_cy) * 10.0 / fmax(1.0, N / 16.0));
            } else {
                const int x = (int)lround(cx), y = (int)lround(cy);
                if (is_real(x, y) && ID(x, y) != last_site) {
                    apply_tool_at(x, y);
                    last_site = ID(x, y);
                }
            }
            last_cx = cx; last_cy = cy;
        }
    } else if (ev->type == SDL_MOUSEBUTTONUP) {
        drag_slider = -1;
        lat_drag = 0;
        last_site = -1;
    } else if (ev->type == SDL_MOUSEWHEEL) {
        int mx, my;
        SDL_GetMouseState(&mx, &my);
        float lx = (float)mx, ly = (float)my;
#if SDL_VERSION_ATLEAST(2, 0, 18)
        SDL_RenderWindowToLogical(ren, mx, my, &lx, &ly);
#endif
        for (int i = 0; i < N_SLIDERS; ++i)
            if (slider_hit(i, (int)lx, (int)ly)) {
                Slider *s = &sliders[i];
                double t = slider_t(s);
                if (s->is_int) *s->v = clampd(*s->v + (ev->wheel.y > 0 ? 1 : -1), s->lo, s->hi);
                else slider_set_t(s, t + (ev->wheel.y > 0 ? 0.005 : -0.005));
                break;
            }
    }
}

static void handle_event(const SDL_Event *ev)
{
    switch (ev->type) {
    case SDL_QUIT: running = false; break;
    case SDL_KEYDOWN: handle_key(&ev->key); break;
    case SDL_MOUSEBUTTONDOWN: case SDL_MOUSEBUTTONUP:
    case SDL_MOUSEMOTION: case SDL_MOUSEWHEEL:
        handle_mouse(ev); break;
    default: break;
    }
}

/* ===================================================================
 * Frame
 * =================================================================== */

static void shutdown_all(void);

static void frame(void)
{
    now_ms = SDL_GetTicks();
    ++frame_no;

    SDL_Event ev;
    while (SDL_PollEvent(&ev)) handle_event(&ev);

    check_param_changes();
    if (disp_dirty) disp_update();
    if (cv_dirty && (headless || now_ms - cv_dirty_since > 150)) cv_rebuild();

    const double tb = sim_time;
    if (!paused) advance((int)prm.speed);
    else if (step_once) { advance(1); step_once = false; }
    const double dts = sim_time - tb;

    diagnostics_update(dts);
    const int every = N <= 128 ? 1 : (N <= 256 ? 2 : 4);
    if (frame_no % (unsigned)every == 0 || dts > 0.0) spectra_update();
    if (dts > 0.0) { ts_push(); profile_update(); }

    render_all();
    if (recording) capture_frame();
    if (shot_file && (int)frame_no >= shot_frames) {
        if (!save_screenshot(shot_file)) fprintf(stderr, "crystal_phonons: screenshot failed\n");
        running = false;
    }
    SDL_RenderPresent(ren);

    static Uint32 fps_t = 0; static int fps_n = 0;
    ++fps_n;
    if (now_ms - fps_t >= 500) { fps = fps_n * 1000.0 / (double)(now_ms - fps_t + 1e-9); fps_n = 0; fps_t = now_ms; }

#ifdef __EMSCRIPTEN__
    if (!running) { emscripten_cancel_main_loop(); shutdown_all(); }
#endif
}

/* ===================================================================
 * Fonts, init, shutdown
 * =================================================================== */

static const char *find_font(const char *env, const char *const *paths)
{
    const char *e = getenv(env);
    if (e && *e) return e;
    for (int i = 0; paths[i]; ++i) {
        FILE *f = fopen(paths[i], "rb");
        if (f) { fclose(f); return paths[i]; }
    }
    return NULL;
}

static int load_fonts(void)
{
    static const char *sans[] = {
#ifdef __EMSCRIPTEN__
        "/fonts/DejaVuSans.ttf",
#endif
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/local/share/fonts/dejavu/DejaVuSans.ttf",
        "/opt/homebrew/share/fonts/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        NULL
    };
    static const char *mono[] = {
#ifdef __EMSCRIPTEN__
        "/fonts/DejaVuSansMono.ttf",
#endif
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf",
        "/usr/local/share/fonts/dejavu/DejaVuSansMono.ttf",
        "/opt/homebrew/share/fonts/DejaVuSansMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
        NULL
    };
    const char *fs = find_font("PHONONS_FONT", sans);
    const char *fm = find_font("PHONONS_FONT_MONO", mono);
    if (!fs || !fm) {
        fprintf(stderr, "crystal_phonons: no usable font found.  Install DejaVu "
                        "(apt install fonts-dejavu) or set PHONONS_FONT and "
                        "PHONONS_FONT_MONO to .ttf files.\n");
        return -1;
    }
    font_lbl   = TTF_OpenFont(fs, 13);
    font_title = TTF_OpenFont(fs, 16);
    font_mono  = TTF_OpenFont(fm, 12);
    if (!font_lbl || !font_title || !font_mono) {
        fprintf(stderr, "crystal_phonons: TTF_OpenFont: %s\n", TTF_GetError());
        return -1;
    }
    return 0;
}

static void shutdown_all(void)
{
    if (recording) stop_recording();
    free(video_buf); video_buf = NULL;
    tcache_clear();
    destroy_textures();
    if (font_lbl)   TTF_CloseFont(font_lbl);
    if (font_title) TTF_CloseFont(font_title);
    if (font_mono)  TTF_CloseFont(font_mono);
    font_lbl = font_title = font_mono = NULL;
    TTF_Quit();
    if (ren) SDL_DestroyRenderer(ren);
    if (win) SDL_DestroyWindow(win);
    ren = NULL; win = NULL;
    SDL_Quit();
}

/* ===================================================================
 * Self-test
 * =================================================================== */

static int st_fail = 0;

static void st_report(const char *name, bool ok, const char *fmt, ...)
{
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    printf("%s  %-36s %s\n", ok ? "PASS" : "FAIL", name, buf);
    if (!ok) ++st_fail;
}

static void st_setup(int n)
{
    prm = prm_default;
    bc = BC_PERIODIC; thermo = TH_OFF; additive = false; shape = SH_STANDING;
    set_lattice_size(n);
    driver_I = -1;
}

static void st_project(int nx, int ny, int b, double *re, double *im)
{
    const double kx = 2.0 * M_PI * nx / N, ky = 2.0 * M_PI * ny / N;
    double w[4], ev[4][4];
    const int nb = phonons_at(kx, ky, w, ev);
    const int nsub = nb / 2;
    double r = 0.0, i = 0.0;
    for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
            const int I = ID(x, y);
            const int s = (nsub == 2) ? ((x + y) & 1) : 0;
            const double pj = sqrt(nominal_mass(x, y)) * (ev[b][2 * s] * ux[I] + ev[b][2 * s + 1] * uy[I]);
            const double ph = kx * x + ky * y;
            r += pj * cos(ph); i -= pj * sin(ph);
        }
    *re = r; *im = i;
}

/* Measure the frequency of a travelling mode by the phase slope of its
 * projection; returns |omega| and the leakage into the other branches. */
static double st_mode_freq(int nx, int ny, int b, int nsteps, double *leak)
{
    double re, im;
    st_project(nx, ny, b, &re, &im);
    double ph0 = atan2(im, re), prev = ph0, unw = 0.0;
    const double a0 = hypot(re, im);
    for (int s = 0; s < nsteps; ++s) {
        verlet_step();
        st_project(nx, ny, b, &re, &im);
        const double ph = atan2(im, re);
        double d = ph - prev;
        while (d > M_PI) d -= 2.0 * M_PI;
        while (d < -M_PI) d += 2.0 * M_PI;
        unw += d; prev = ph;
    }
    double lk = 0.0;
    const int nb = n_branches();
    for (int o = 0; o < nb; ++o) {
        if (o == b) continue;
        st_project(nx, ny, o, &re, &im);
        lk = fmax(lk, hypot(re, im) / a0);
    }
    *leak = lk;
    return fabs(unw) / (nsteps * DT);
}

static int run_selftest(void)
{
    headless = true;
    rng_seed(12345);
    printf("crystal_phonons %s self-test\n\n", PHONONS_VERSION);

    /* 1. Jacobi eigensolver vs closed form (monatomic with K_D coupling) */
    {
        st_setup(64);
        prm.KD = 0.7; prm.KT = 0.2;
        double maxerr = 0.0;
        for (int t = 0; t < 2000; ++t) {
            const double kx = (2.0 * rng_u01() - 1.0) * M_PI, ky = (2.0 * rng_u01() - 1.0) * M_PI;
            double dxx, dyy, dxy, w[4], ev[4][4];
            force_const_mono(kx, ky, &dxx, &dyy, &dxy);
            const double tr = 0.5 * (dxx + dyy), dd = sqrt(0.25 * (dxx - dyy) * (dxx - dyy) + dxy * dxy);
            const double w0 = sqrt(fmax(0.0, tr - dd)), w1 = sqrt(tr + dd);
            phonons_at(kx, ky, w, ev);
            maxerr = fmax(maxerr, fmax(fabs(w[0] - w0), fabs(w[1] - w1)));
        }
        st_report("eigensolver vs closed form", maxerr < 1e-10, "max |dw| = %.2e", maxerr);
    }

    /* 2. Standing LA wave period */
    {
        st_setup(64);
        apply_settings();
        prm.amp = 0.05; shape = SH_STANDING;
        excite(4, 0, 1);
        const double wa = omega_at(2.0 * M_PI * 4 / 64, 0.0, 1);
        double prev = ux[ID(0, 0)], tfirst = -1, tlast = -1;
        int ncross = 0;
        for (int s = 0; s < 6000; ++s) {
            verlet_step();
            const double u = ux[ID(0, 0)];
            if (prev < 0.0 && u >= 0.0) {
                const double tc = sim_time - DT * u / (u - prev);
                if (tfirst < 0) tfirst = tc; else { tlast = tc; ++ncross; }
            }
            prev = u;
        }
        const double Tm = (tlast - tfirst) / ncross, Ta = 2.0 * M_PI / wa;
        st_report("standing LA period", fabs(Tm / Ta - 1.0) < 2e-3,
                  "measured %.4f  analytic %.4f  (%+.2e)", Tm, Ta, Tm / Ta - 1.0);
    }

    /* 3. Oblique travelling wave with K_D coupling: frequency + purity */
    {
        st_setup(64);
        prm.KD = 0.5; apply_settings();
        prm.amp = 0.05; shape = SH_TRAVEL;
        for (int b = 0; b < 2; ++b) {
            excite(5, 3, b);
            double leak;
            const double wm = st_mode_freq(5, 3, b, 3000, &leak);
            const double wa = omega_at(2.0 * M_PI * 5 / 64, 2.0 * M_PI * 3 / 64, b);
            char nm[64]; snprintf(nm, sizeof nm, "K_D mode k=(5,3) branch %d", b + 1);
            st_report(nm, fabs(wm / wa - 1.0) < 1e-3 && leak < 1e-6,
                      "w %.5f vs %.5f, leak %.1e", wm, wa, leak);
        }
    }

    /* 4. Diatomic optical branch */
    {
        st_setup(64);
        prm.mratio = 3.0; prm.KD = 0.3; apply_settings();
        prm.amp = 0.05; shape = SH_TRAVEL;
        for (int b = 2; b < 4; ++b) {
            excite(3, 1, b);
            double leak;
            const double wm = st_mode_freq(3, 1, b, 3000, &leak);
            const double wa = omega_at(2.0 * M_PI * 3 / 64, 2.0 * M_PI * 1 / 64, b);
            char nm[64]; snprintf(nm, sizeof nm, "diatomic optical k=(3,1) b%d", b + 1);
            st_report(nm, fabs(wm / wa - 1.0) < 2e-3 && leak < 1e-5,
                      "w %.5f vs %.5f, leak %.1e", wm, wa, leak);
        }
    }

    /* 5. Forces are -grad(PE) for every BC, with K_D, alpha, beta, vacancies */
    for (int b = 0; b < BC_COUNT; ++b) {
        st_setup(12);
        bc = b;
        prm.KD = 0.4; prm.alpha = 0.3; prm.beta = 0.8; prm.KT = 0.25;
        alive[ID(3, 4)] = 0; alive[ID(0, 7)] = 0; alive[ID(11, 11)] = 0;
        defm[ID(5, 5)] = 4.0;
        apply_settings();
        for (int y = 0; y < N; ++y)
            for (int x = 0; x < N; ++x)
                if (alive[ID(x, y)]) { ux[ID(x, y)] = 0.2 * randn(); uy[ID(x, y)] = 0.2 * randn(); }
        compute_forces();
        double maxerr = 0.0;
        static const int pts[][2] = { {0, 0}, {11, 0}, {0, 11}, {11, 10}, {5, 5}, {3, 3},
                                      {6, 0}, {0, 6}, {2, 4}, {4, 4}, {10, 11}, {1, 7} };
        for (size_t p = 0; p < sizeof pts / sizeof pts[0]; ++p) {
            const int I = ID(pts[p][0], pts[p][1]);
            if (!alive[I]) continue;
            for (int c = 0; c < 2; ++c) {
                double *u = c ? uy : ux;
                const double f = c ? fy[I] : fx[I];
                const double h = 1e-5, u0 = u[I];
                double ke, pp, pm;
                u[I] = u0 + h; compute_energy(&ke, &pp);
                u[I] = u0 - h; compute_energy(&ke, &pm);
                u[I] = u0;
                maxerr = fmax(maxerr, fabs(-(pp - pm) / (2.0 * h) - f));
            }
        }
        compute_forces();
        char nm[64]; snprintf(nm, sizeof nm, "F = -grad PE (%s BC)", bc_names[b]);
        st_report(nm, maxerr < 1e-6, "max error %.1e", maxerr);
    }

    /* 6. Energy conservation: anharmonic, fixed BC, vacancies + defects */
    {
        st_setup(32);
        bc = BC_FIXED; prm.KD = 0.3; prm.beta = 1.0; prm.alpha = 0.2;
        alive[ID(10, 10)] = 0; defm[ID(20, 12)] = 0.25; defm[ID(5, 25)] = 4.0;
        apply_settings();
        thermalize(0.1);
        double ke, pe;
        compute_energy(&ke, &pe);
        const double E0 = ke + pe;
        double dmax = 0.0;
        for (int s = 0; s < 5000; ++s) {
            verlet_step();
            if (s % 50 == 0) { compute_energy(&ke, &pe); dmax = fmax(dmax, fabs(ke + pe - E0)); }
        }
        st_report("energy conservation (anharmonic)", dmax / E0 < 2e-3,
                  "max |dE|/E = %.1e over 5000 steps", dmax / E0);
    }

    /* 7. Wave packet moves at the group velocity */
    {
        st_setup(128);
        apply_settings();
        prm.amp = 0.05; prm.pk_sigma = 8.0; shape = SH_PACKET;
        excite(16, 0, 1);
        double vgx, vgy;
        group_velocity(2.0 * M_PI * 16 / 128, 0.0, 1, &vgx, &vgy);
        double c0 = 0.0;
        for (int pass = 0; pass < 2; ++pass) {
            double sx = 0.0, cxs = 0.0;
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x) {
                    const int I = ID(x, y);
                    const double e = vx[I] * vx[I] + vy[I] * vy[I] + prm.KL * (ux[I] * ux[I] + uy[I] * uy[I]) * 0.0;
                    const double th = 2.0 * M_PI * x / N;
                    sx += e * sin(th); cxs += e * cos(th);
                }
            const double c = atan2(sx, cxs) / (2.0 * M_PI) * N;
            if (pass == 0) { c0 = c; for (int s = 0; s < 1000; ++s) verlet_step(); }
            else {
                double d = c - c0;
                if (d < 0) d += N;
                const double vm = d / (1000 * DT);
                st_report("packet moves at v_g", fabs(vm / vgx - 1.0) < 0.03,
                          "measured %.4f  v_g %.4f", vm, vgx);
            }
        }
    }

    /* 8. Langevin thermostat reaches the target temperature */
    {
        st_setup(32);
        thermo = TH_GLOBAL; prm.kT = 0.3; prm.gamma = 0.5;
        apply_settings();
        double acc = 0.0; int n = 0;
        for (int s = 0; s < 24000; ++s) {
            verlet_step();
            if (s > 6000 && s % 10 == 0) { double ke, pe; compute_energy(&ke, &pe); acc += ke / NN; ++n; }
        }
        const double T = acc / n;
        st_report("Langevin kT_eff", fabs(T / 0.3 - 1.0) < 0.03, "%.4f vs 0.3000", T);
    }

    /* 9. Quantum C_v limits */
    {
        st_setup(64);
        apply_settings();
        cv_rebuild();
        const double hi = cv_at(1000.0 * omega_top) / cv_ndp;
        st_report("C_v -> Dulong-Petit at high T", fabs(hi - 1.0) < 1e-4, "C/C_DP = %.6f", hi);
        const double T1 = 0.05 * omega_top;
        const double r = cv_at(2.0 * T1) / cv_at(T1);
        st_report("C_v ~ T^2 at low T (2D Debye)", r > 3.2 && r < 4.4, "C(2T)/C(T) = %.3f", r);
    }

    /* 10. Measured S(k,omega) peaks on the analytic branches */
    {
        st_setup(32);
        apply_settings();
        thermalize(0.1);
        while (s_nacc < 6) advance(S_STRIDE);
        const int H = SW_LEN / 2;
        const double dw = s_domega();
        bool ok = true;
        double worst = 0.0;
        for (int p = 2; p <= N / 2; p += 3) {
            const double *acc = &s_acc[(size_t)p * H];
            int jmax = 1;
            for (int j = 1; j < H; ++j) if (acc[j] > acc[jmax]) jmax = j;
            const double wp = jmax * dw;
            const double kx = 2.0 * M_PI * p / N;
            const double d = fmin(fabs(wp - omega_at(kx, 0.0, 0)), fabs(wp - omega_at(kx, 0.0, 1)));
            worst = fmax(worst, d / dw);
            if (d > 1.5 * dw) ok = false;
        }
        st_report("S(k,w) peaks on dispersion", ok, "worst offset %.2f bins (dw = %.4f)", worst, dw);
    }

    printf("\n%s: %d failure%s\n", st_fail ? "FAILED" : "OK", st_fail, st_fail == 1 ? "" : "s");
    free_lattice();
    return st_fail ? 1 : 0;
}

/* ===================================================================
 * main
 * =================================================================== */

static void usage(FILE *f)
{
    fprintf(f,
        "usage: crystal_phonons [options]\n"
        "  -n N            lattice side, power of two, %d..%d (default 64)\n"
        "  -d, --demo NAME start a demo: packet focusing fput breathers impurity\n"
        "                  disorder heat diatomic driven\n"
        "  -s SEED         random seed (default 42)\n"
        "  -o DIR          directory for videos and screenshots (default .)\n"
        "  --shot FILE     render, save a BMP screenshot to FILE and exit\n"
        "  --frames K      frames to run before --shot (default 120)\n"
        "  --selftest      run the numerical validation suite and exit\n"
        "  -h, --help      this help;  -V, --version  print version\n"
        "Press F1 in the window for the key reference.\n", NMIN, NMAX);
}

int main(int argc, char **argv)
{
    int n = 64, demo = -1;
    uint64_t seed = 42;
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        const bool more = i + 1 < argc;
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(stdout); return 0; }
        else if (!strcmp(a, "-V") || !strcmp(a, "--version")) { printf("crystal_phonons %s\n", PHONONS_VERSION); return 0; }
        else if (!strcmp(a, "--selftest")) selftest = true;
        else if (!strcmp(a, "-n") && more) n = atoi(argv[++i]);
        else if (!strcmp(a, "-s") && more) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(a, "-o") && more) out_dir = argv[++i];
        else if ((!strcmp(a, "-d") || !strcmp(a, "--demo")) && more) {
            demo = demo_by_name(argv[++i]);
            if (demo < 0) { fprintf(stderr, "crystal_phonons: unknown demo '%s'\n", argv[i]); usage(stderr); return 2; }
        }
        else if (!strcmp(a, "--shot") && more) shot_file = argv[++i];
        else if (!strcmp(a, "--frames") && more) shot_frames = atoi(argv[++i]);
        else { fprintf(stderr, "crystal_phonons: bad argument '%s'\n", a); usage(stderr); return 2; }
    }
    if (selftest) return run_selftest();
    if (n < NMIN || n > NMAX || (n & (n - 1))) {
        fprintf(stderr, "crystal_phonons: -n must be a power of two in %d..%d\n", NMIN, NMAX);
        return 2;
    }

    rng_seed(seed);
    build_luts();
    prm = prm_default;
#ifdef SIGPIPE
    signal(SIGPIPE, SIG_IGN);
#endif

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "crystal_phonons: SDL_Init: %s\n", SDL_GetError());
        return 1;
    }
    if (TTF_Init() != 0) {
        fprintf(stderr, "crystal_phonons: TTF_Init: %s\n", TTF_GetError());
        SDL_Quit(); return 1;
    }
    int ww = WIN_W, wh = WIN_H;
    SDL_Rect ub;
    if (SDL_GetDisplayUsableBounds(0, &ub) == 0) {
        const double sc = fmin(1.0, fmin((ub.w - 40) / (double)WIN_W, (ub.h - 60) / (double)WIN_H));
        if (sc > 0.3) { ww = (int)(WIN_W * sc); wh = (int)(WIN_H * sc); }
    }
    win = SDL_CreateWindow("crystal_phonons", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                           ww, wh, SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
    if (!win) {
        fprintf(stderr, "crystal_phonons: SDL_CreateWindow: %s\n", SDL_GetError());
        TTF_Quit(); SDL_Quit(); return 1;
    }
    ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!ren) ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_SOFTWARE);
    if (!ren) {
        fprintf(stderr, "crystal_phonons: SDL_CreateRenderer: %s\n", SDL_GetError());
        SDL_DestroyWindow(win); TTF_Quit(); SDL_Quit(); return 1;
    }
    SDL_RenderSetLogicalSize(ren, WIN_W, WIN_H);
    SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_BLEND);
    if (load_fonts() != 0) { shutdown_all(); return 1; }

    set_lattice_size(n);
    if (demo >= 0) apply_demo(demo);
    else {
        /* start with something moving: an LA packet along x */
        excite(N / 8, 0, 1);
        toast("Click the dispersion plot or the k-map to launch waves; click the lattice to poke it.\n"
              "F1: help   F2\xE2\x80\x93" "F10: demos   (F4 FPUT, F9 diatomic S(k,\xCF\x89), F10 driven)");
    }

#ifdef __EMSCRIPTEN__
    emscripten_set_main_loop(frame, 0, 1);
#else
    while (running) frame();
    shutdown_all();
    free_lattice();
#endif
    return 0;
}
