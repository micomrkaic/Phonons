/*
 * crystal_phonons.c -- 2D phonon simulator with SDL2 + SDL_ttf.
 *
 * Layout (1340 x 760 window):
 *   left   lattice animation (720 x 720)
 *   right  control panel: dispersion plot, parameter sliders, key list,
 *          live diagnostics; everything renders in-window, nothing is
 *          printed to stderr during the run.
 *
 * Video capture:
 *   Press V to toggle. Frames are piped raw to ffmpeg, encoded with
 *   libx264 to phonons_YYYYMMDD_HHMMSS.mp4 in the current directory.
 *
 * See README.md for the physics and validation notes.
 */

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>

#include <math.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

/* ============== compile-time parameters ============================= */
#define N            48
#define MASS         1.0
#define DT           0.04
#define SUBSTEPS     3

#define WIN_W        1340
#define WIN_H        760
#define LAT_PAD      20
#define LAT_PX       720

/* right-hand control panel */
#define PANEL_X      760
#define PANEL_W      560

/* dispersion plot */
#define DISP_TITLE_Y 15
#define DISP_X       (PANEL_X + 20)
#define DISP_Y       50
#define DISP_W       (PANEL_W - 40)
#define DISP_H       180
#define DISP_PTS     300

/* parameter sliders */
#define PARAM_TITLE_Y 268
#define SLIDER_Y0     314
#define SLIDER_DY     42
#define SLIDER_H      14
#define N_SLIDERS     5

/* keybindings + diagnostics */
#define KEYS_TITLE_Y 540
#define KEYS_LIST_Y  566
#define KEYS_LINE_DY 18
#define READOUTS_X   (PANEL_X + 280)

/* ============== runtime parameters (mutable via sliders) ============ */
static double g_KL        = 1.0;
static double g_KT        = 0.30;
static double g_kT_target = 0.05;
static double g_amp       = 0.25;
static double g_kn        = 4.0;     /* mode index, slider stores as double */

/* ============== state =============================================== */
static double ux[N][N], uy[N][N];
static double vx[N][N], vy[N][N];
static double fx[N][N], fy[N][N];

static bool   paused      = false;
static int    color_mode  = 0;       /* 0:|u|  1:|v|  2:local KE */
static long   step_no     = 0;
static double sim_time    = 0.0;

/* Live diagnostics, refreshed every few frames */
static double diag_KE = 0.0, diag_PE = 0.0, diag_kT_eff = 0.0, diag_cv = 0.0;

/* ============== SDL/TTF resources =================================== */
static SDL_Window   *win        = NULL;
static SDL_Renderer *ren        = NULL;
static TTF_Font     *font_title = NULL;
static TTF_Font     *font_lbl   = NULL;
static TTF_Font     *font_mono  = NULL;

/* ============== video capture ======================================= */
static FILE    *video_pipe   = NULL;
static uint8_t *video_buffer = NULL;
static bool     recording    = false;
static char     video_filename[160] = "";

/* ============== colours ============================================= */
static const SDL_Color COL_FG       = { 220, 220, 220, 255 };
static const SDL_Color COL_DIM      = { 140, 140, 145, 255 };
static const SDL_Color COL_TITLE    = { 240, 240, 240, 255 };
static const SDL_Color COL_KEY      = { 110, 200, 240, 255 };
static const SDL_Color COL_X_POL    = {  80, 180, 230, 255 };
static const SDL_Color COL_Y_POL    = { 240, 130,  60, 255 };
static const SDL_Color COL_REC      = { 230,  60,  60, 255 };
static const SDL_Color COL_MARKER   = { 240, 220,  90, 255 };

/* ===================================================================
 * Physics
 * =================================================================== */

static void zero_state(void)
{
    memset(ux, 0, sizeof ux); memset(uy, 0, sizeof uy);
    memset(vx, 0, sizeof vx); memset(vy, 0, sizeof vy);
    memset(fx, 0, sizeof fx); memset(fy, 0, sizeof fy);
    step_no = 0;
    sim_time = 0.0;
}

static void compute_forces(void)
{
    for (int i = 0; i < N; ++i) {
        const int ip = (i + 1) % N, im = (i + N - 1) % N;
        for (int j = 0; j < N; ++j) {
            const int jp = (j + 1) % N, jm = (j + N - 1) % N;
            fx[i][j] = g_KL * (ux[ip][j] + ux[im][j] - 2.0 * ux[i][j])
                     + g_KT * (ux[i][jp] + ux[i][jm] - 2.0 * ux[i][j]);
            fy[i][j] = g_KT * (uy[ip][j] + uy[im][j] - 2.0 * uy[i][j])
                     + g_KL * (uy[i][jp] + uy[i][jm] - 2.0 * uy[i][j]);
        }
    }
}

static void verlet_step(void)
{
    const double h = 0.5 * DT;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            vx[i][j] += h * fx[i][j] / MASS;
            vy[i][j] += h * fy[i][j] / MASS;
            ux[i][j] += DT * vx[i][j];
            uy[i][j] += DT * vy[i][j];
        }
    compute_forces();
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            vx[i][j] += h * fx[i][j] / MASS;
            vy[i][j] += h * fy[i][j] / MASS;
        }
    ++step_no;
    sim_time += DT;
}

static void compute_energy(double *KE, double *PE)
{
    double ke = 0.0, pe = 0.0;
    for (int i = 0; i < N; ++i) {
        const int ip = (i + 1) % N;
        for (int j = 0; j < N; ++j) {
            const int jp = (j + 1) % N;
            ke += 0.5 * MASS * (vx[i][j]*vx[i][j] + vy[i][j]*vy[i][j]);
            const double dux_x = ux[ip][j] - ux[i][j];
            const double duy_x = uy[ip][j] - uy[i][j];
            pe += 0.5 * g_KL * dux_x * dux_x + 0.5 * g_KT * duy_x * duy_x;
            const double dux_y = ux[i][jp] - ux[i][j];
            const double duy_y = uy[i][jp] - uy[i][j];
            pe += 0.5 * g_KT * dux_y * dux_y + 0.5 * g_KL * duy_y * duy_y;
        }
    }
    *KE = ke; *PE = pe;
}

static void excite_plane_wave(int nx, int ny, double ex, double ey, double amp)
{
    const double dkx = 2.0 * M_PI * nx / (double)N;
    const double dky = 2.0 * M_PI * ny / (double)N;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            const double c = cos(dkx * i + dky * j);
            ux[i][j] += amp * ex * c;
            uy[i][j] += amp * ey * c;
        }
}

static double randn(void)
{
    const double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
    const double u2 =  rand()        / (RAND_MAX + 1.0);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static void thermalize(double T)
{
    if (T <= 0.0) {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) { vx[i][j] = 0.0; vy[i][j] = 0.0; }
        return;
    }
    const double sigma = sqrt(T / MASS);
    double sxv = 0.0, syv = 0.0;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            vx[i][j] = sigma * randn();
            vy[i][j] = sigma * randn();
            sxv += vx[i][j]; syv += vy[i][j];
        }
    sxv /= (double)(N * N); syv /= (double)(N * N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            vx[i][j] -= sxv; vy[i][j] -= syv;
        }
}

static double omega_xpol(double kx, double ky)
{
    const double sx = sin(0.5 * kx), sy = sin(0.5 * ky);
    return sqrt((4.0 * g_KL / MASS) * sx * sx + (4.0 * g_KT / MASS) * sy * sy);
}
static double omega_ypol(double kx, double ky)
{
    const double sx = sin(0.5 * kx), sy = sin(0.5 * ky);
    return sqrt((4.0 * g_KT / MASS) * sx * sx + (4.0 * g_KL / MASS) * sy * sy);
}

static double cv_quantum(double T)
{
    if (T <= 0.0) return 0.0;
    double C = 0.0;
    for (int nx = 0; nx < N; ++nx)
        for (int ny = 0; ny < N; ++ny) {
            if (nx == 0 && ny == 0) continue;
            const double kx = 2.0 * M_PI * nx / (double)N;
            const double ky = 2.0 * M_PI * ny / (double)N;
            const double w[2] = { omega_xpol(kx, ky), omega_ypol(kx, ky) };
            for (int b = 0; b < 2; ++b) {
                const double x = w[b] / T;
                if (x < 1e-8) { C += 1.0; continue; }
                const double ex = exp(x);
                C += x * x * ex / ((ex - 1.0) * (ex - 1.0));
            }
        }
    return C;
}

/* ===================================================================
 * Text rendering helper
 * =================================================================== */

static void render_text(TTF_Font *font, SDL_Color color,
                        const char *text, int x, int y)
{
    if (!text || !*text || !font) return;
    SDL_Surface *surf = TTF_RenderUTF8_Blended(font, text, color);
    if (!surf) return;
    SDL_Texture *tex = SDL_CreateTextureFromSurface(ren, surf);
    if (tex) {
        SDL_Rect dst = { x, y, surf->w, surf->h };
        SDL_RenderCopy(ren, tex, NULL, &dst);
        SDL_DestroyTexture(tex);
    }
    SDL_FreeSurface(surf);
}

/* ===================================================================
 * Lattice rendering
 * =================================================================== */

static void hsv_to_rgb(double h, double s, double v,
                       uint8_t *r, uint8_t *g, uint8_t *b)
{
    const double c  = v * s;
    const double hh = fmod(h, 360.0) / 60.0;
    const double x  = c * (1.0 - fabs(fmod(hh, 2.0) - 1.0));
    double rr = 0.0, gg = 0.0, bb = 0.0;
    if      (hh < 1.0) { rr = c; gg = x; }
    else if (hh < 2.0) { rr = x; gg = c; }
    else if (hh < 3.0) { gg = c; bb = x; }
    else if (hh < 4.0) { gg = x; bb = c; }
    else if (hh < 5.0) { rr = x; bb = c; }
    else               { rr = c; bb = x; }
    const double m = v - c;
    *r = (uint8_t)(255.0 * (rr + m));
    *g = (uint8_t)(255.0 * (gg + m));
    *b = (uint8_t)(255.0 * (bb + m));
}

static double scalar_at(int i, int j)
{
    if (color_mode == 0) return sqrt(ux[i][j]*ux[i][j] + uy[i][j]*uy[i][j]);
    if (color_mode == 1) return sqrt(vx[i][j]*vx[i][j] + vy[i][j]*vy[i][j]);
    return 0.5 * MASS * (vx[i][j]*vx[i][j] + vy[i][j]*vy[i][j]);
}

static void draw_lattice(void)
{
    double smax = 1e-12;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            const double s = scalar_at(i, j);
            if (s > smax) smax = s;
        }
    double umax = 1e-12;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            const double m = fmax(fabs(ux[i][j]), fabs(uy[i][j]));
            if (m > umax) umax = m;
        }
    const double cell = (double)LAT_PX / (double)N;
    /* draw_scale converts lattice-spacing units to pixels.
     * Default 1 unit -> 1 cell. Cap at 0.4 cell so atoms don't overlap. */
    double draw_scale = cell;
    if (umax * draw_scale > 0.4 * cell) draw_scale = 0.4 * cell / umax;
    int rad = (int)(0.30 * cell);
    if (rad < 2) rad = 2;

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            const double cx = LAT_PAD + (i + 0.5) * cell + draw_scale * ux[i][j];
            const double cy = LAT_PAD + (j + 0.5) * cell + draw_scale * uy[i][j];
            const double t  = scalar_at(i, j) / smax;
            uint8_t r, g, b;
            hsv_to_rgb(240.0 * (1.0 - t), 0.85, 0.40 + 0.60 * t, &r, &g, &b);
            SDL_SetRenderDrawColor(ren, r, g, b, 255);
            SDL_Rect rect = { (int)(cx - rad), (int)(cy - rad), 2*rad, 2*rad };
            SDL_RenderFillRect(ren, &rect);
        }
}

/* ===================================================================
 * Dispersion plot (Gamma -> X -> M -> Gamma)
 * =================================================================== */

static void draw_dispersion(void)
{
    render_text(font_title, COL_TITLE,
                "Dispersion   \xCE\x93 - X - M - \xCE\x93",
                PANEL_X + 20, DISP_TITLE_Y);

    const double omega_max = sqrt(4.0 * (g_KL + g_KT) / MASS);

    /* axes */
    SDL_SetRenderDrawColor(ren, 180, 180, 180, 255);
    SDL_RenderDrawLine(ren, DISP_X,          DISP_Y + DISP_H,
                            DISP_X + DISP_W, DISP_Y + DISP_H);
    SDL_RenderDrawLine(ren, DISP_X, DISP_Y, DISP_X, DISP_Y + DISP_H);

    /* high-symmetry markers */
    SDL_SetRenderDrawColor(ren, 80, 80, 90, 255);
    for (int k = 1; k < 3; ++k) {
        const int xx = DISP_X + (int)(k * DISP_W / 3);
        SDL_RenderDrawLine(ren, xx, DISP_Y, xx, DISP_Y + DISP_H);
    }

    /* both branches */
    SDL_Point ptsx[DISP_PTS], ptsy[DISP_PTS];
    for (int p = 0; p < DISP_PTS; ++p) {
        const double s = 3.0 * p / (double)(DISP_PTS - 1);
        double kx = 0.0, ky = 0.0;
        if (s < 1.0)      { kx = M_PI * s;     ky = 0.0;             }
        else if (s < 2.0) { kx = M_PI;         ky = M_PI * (s - 1.0); }
        else              { const double u = s - 2.0;
                            kx = M_PI * (1.0 - u);
                            ky = M_PI * (1.0 - u); }
        const double wx = omega_xpol(kx, ky);
        const double wy = omega_ypol(kx, ky);
        const int X = DISP_X + (int)(p * (double)DISP_W / (double)(DISP_PTS - 1));
        ptsx[p].x = X; ptsx[p].y = DISP_Y + DISP_H - (int)((wx / omega_max) * DISP_H);
        ptsy[p].x = X; ptsy[p].y = DISP_Y + DISP_H - (int)((wy / omega_max) * DISP_H);
    }
    SDL_SetRenderDrawColor(ren, COL_X_POL.r, COL_X_POL.g, COL_X_POL.b, 255);
    SDL_RenderDrawLines(ren, ptsx, DISP_PTS);
    SDL_SetRenderDrawColor(ren, COL_Y_POL.r, COL_Y_POL.g, COL_Y_POL.b, 255);
    SDL_RenderDrawLines(ren, ptsy, DISP_PTS);

    /* marker for currently selected k_n along Gamma->X */
    const int kn = (int)g_kn;
    const double s_kn = 2.0 * kn / (double)N;        /* in [0, 1] along G->X */
    if (s_kn <= 1.0) {
        const int xm = DISP_X + (int)(s_kn / 3.0 * DISP_W);
        SDL_SetRenderDrawColor(ren, COL_MARKER.r, COL_MARKER.g, COL_MARKER.b, 200);
        for (int yy = DISP_Y; yy < DISP_Y + DISP_H; yy += 4)
            SDL_RenderDrawPoint(ren, xm, yy);
    }

    /* axis labels */
    const char *labels[] = { "\xCE\x93", "X", "M", "\xCE\x93" };
    for (int k = 0; k < 4; ++k) {
        const int xx = DISP_X + (int)(k * DISP_W / 3);
        render_text(font_mono, COL_DIM, labels[k], xx - 4, DISP_Y + DISP_H + 4);
    }
    /* legend */
    render_text(font_mono, COL_X_POL, "x-pol",
                DISP_X + DISP_W - 80, DISP_Y + 4);
    render_text(font_mono, COL_Y_POL, "y-pol",
                DISP_X + DISP_W - 80, DISP_Y + 22);
    /* omega_max */
    char buf[64];
    snprintf(buf, sizeof buf, "\xCF\x89_max = %.3f", omega_max);
    render_text(font_mono, COL_DIM, buf, DISP_X + 6, DISP_Y + 4);
}

/* ===================================================================
 * Sliders
 * =================================================================== */

typedef struct {
    const char *label;
    double *value;
    double min_val, max_val;
    int x, y, w;
    bool is_int;
    int decimals;
    bool dragging;
} Slider;

static Slider sliders[N_SLIDERS];

static void init_sliders(void)
{
    const int sx  = PANEL_X + 20;
    const int sw  = PANEL_W - 40;
    const int sy0 = SLIDER_Y0;
    const int dy  = SLIDER_DY;

    sliders[0] = (Slider){ "K_L  longitudinal stiffness",
                           &g_KL,        0.10, 4.00,
                           sx, sy0 + 0*dy, sw, false, 2, false };
    sliders[1] = (Slider){ "K_T  transverse stiffness",
                           &g_KT,        0.05, 2.00,
                           sx, sy0 + 1*dy, sw, false, 2, false };
    sliders[2] = (Slider){ "kT_target  (press T to apply)",
                           &g_kT_target, 0.00, 2.00,
                           sx, sy0 + 2*dy, sw, false, 3, false };
    sliders[3] = (Slider){ "amp  excitation amplitude",
                           &g_amp,       0.00, 0.50,
                           sx, sy0 + 3*dy, sw, false, 2, false };
    sliders[4] = (Slider){ "k_n  mode index, k = 2\xCF\x80\xC2\xB7k_n/N",
                           &g_kn,        1.00, (double)(N/2),
                           sx, sy0 + 4*dy, sw, true, 0, false };
}

static void slider_set_from_x(Slider *s, int mx)
{
    double t = (double)(mx - s->x) / (double)s->w;
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;
    double v = s->min_val + t * (s->max_val - s->min_val);
    if (s->is_int) v = floor(v + 0.5);
    *s->value = v;
}

static bool slider_hit(Slider *s, int mx, int my)
{
    return mx >= s->x && mx <= s->x + s->w &&
           my >= s->y - 4 && my <= s->y + SLIDER_H + 4;
}

static void handle_mouse_event(SDL_Event *ev)
{
    if (ev->type == SDL_MOUSEBUTTONDOWN && ev->button.button == SDL_BUTTON_LEFT) {
        for (int i = 0; i < N_SLIDERS; ++i)
            if (slider_hit(&sliders[i], ev->button.x, ev->button.y)) {
                sliders[i].dragging = true;
                slider_set_from_x(&sliders[i], ev->button.x);
            }
    } else if (ev->type == SDL_MOUSEMOTION) {
        for (int i = 0; i < N_SLIDERS; ++i)
            if (sliders[i].dragging)
                slider_set_from_x(&sliders[i], ev->motion.x);
    } else if (ev->type == SDL_MOUSEBUTTONUP && ev->button.button == SDL_BUTTON_LEFT) {
        for (int i = 0; i < N_SLIDERS; ++i) sliders[i].dragging = false;
    }
}

static void draw_slider(Slider *s)
{
    char buf[160];
    if (s->is_int)
        snprintf(buf, sizeof buf, "%s:  %d", s->label, (int)*s->value);
    else
        snprintf(buf, sizeof buf, "%s:  %.*f",
                 s->label, s->decimals, *s->value);
    render_text(font_lbl, COL_FG, buf, s->x, s->y - 22);

    SDL_SetRenderDrawColor(ren, 50, 50, 60, 255);
    SDL_Rect bg = { s->x, s->y + (SLIDER_H/2) - 3, s->w, 6 };
    SDL_RenderFillRect(ren, &bg);

    double t = (*s->value - s->min_val) / (s->max_val - s->min_val);
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;
    SDL_SetRenderDrawColor(ren, 100, 160, 220, 255);
    SDL_Rect fill = { s->x, s->y + (SLIDER_H/2) - 3, (int)(t * s->w), 6 };
    SDL_RenderFillRect(ren, &fill);

    SDL_SetRenderDrawColor(ren, 240, 240, 240, 255);
    int tx = s->x + (int)(t * s->w);
    SDL_Rect thumb = { tx - 4, s->y + (SLIDER_H/2) - 8, 8, 16 };
    SDL_RenderFillRect(ren, &thumb);
}

static void draw_sliders(void)
{
    render_text(font_title, COL_TITLE, "Parameters",
                PANEL_X + 20, PARAM_TITLE_Y);
    for (int i = 0; i < N_SLIDERS; ++i) draw_slider(&sliders[i]);
}

/* ===================================================================
 * Keybindings panel
 * =================================================================== */

static const struct { const char *key; const char *desc; } kbinds[] = {
    { "1",     "TA mode at k_n along x (eps = y)" },
    { "2",     "LA mode at k_n along x (eps = x)" },
    { "3",     "M-point mode (k = pi, pi)"        },
    { "T",     "thermalise to kT_target"          },
    { "D",     "cycle colour: |u|, |v|, KE"       },
    { "R",     "reset displacements + velocities" },
    { "V",     "start / stop video capture"       },
    { "Space", "pause / resume"                   },
    { "Q",     "quit"                             },
};
#define N_KBINDS (int)(sizeof kbinds / sizeof kbinds[0])

static void draw_keybindings(void)
{
    const int x = PANEL_X + 20;
    render_text(font_title, COL_TITLE, "Keys", x, KEYS_TITLE_Y);
    int y = KEYS_LIST_Y;
    for (int i = 0; i < N_KBINDS; ++i) {
        render_text(font_mono, COL_KEY, kbinds[i].key, x, y);
        render_text(font_mono, COL_FG,  kbinds[i].desc, x + 60, y);
        y += KEYS_LINE_DY;
    }
}

/* ===================================================================
 * Live diagnostics panel
 * =================================================================== */

static void draw_readouts(void)
{
    render_text(font_title, COL_TITLE, "Diagnostics",
                READOUTS_X, KEYS_TITLE_Y);
    int y = KEYS_LIST_Y;
    char buf[96];

    snprintf(buf, sizeof buf, "t        = %9.2f", sim_time);
    render_text(font_mono, COL_FG, buf, READOUTS_X, y); y += KEYS_LINE_DY;
    snprintf(buf, sizeof buf, "KE       = %9.4f", diag_KE);
    render_text(font_mono, COL_FG, buf, READOUTS_X, y); y += KEYS_LINE_DY;
    snprintf(buf, sizeof buf, "PE       = %9.4f", diag_PE);
    render_text(font_mono, COL_FG, buf, READOUTS_X, y); y += KEYS_LINE_DY;
    snprintf(buf, sizeof buf, "E_total  = %9.4f", diag_KE + diag_PE);
    render_text(font_mono, COL_FG, buf, READOUTS_X, y); y += KEYS_LINE_DY;
    snprintf(buf, sizeof buf, "kT_eff   = %9.4f", diag_kT_eff);
    render_text(font_mono, COL_FG, buf, READOUTS_X, y); y += KEYS_LINE_DY;
    const double ratio = (diag_PE > 1e-12) ? diag_KE / diag_PE : 0.0;
    snprintf(buf, sizeof buf, "KE / PE  = %9.4f", ratio);
    render_text(font_mono, COL_FG, buf, READOUTS_X, y); y += KEYS_LINE_DY;
    snprintf(buf, sizeof buf, "C_v(kT)  = %8.1f k_B", diag_cv);
    render_text(font_mono, COL_FG, buf, READOUTS_X, y); y += KEYS_LINE_DY;
    snprintf(buf, sizeof buf, "step #   = %9ld", step_no);
    render_text(font_mono, COL_DIM, buf, READOUTS_X, y); y += KEYS_LINE_DY;
    if (paused)
        render_text(font_mono, COL_MARKER, "[ PAUSED ]", READOUTS_X, y);
}

/* ===================================================================
 * Recording badge
 * =================================================================== */

static void draw_recording_badge(void)
{
    if (!recording) return;
    SDL_SetRenderDrawColor(ren, COL_REC.r, COL_REC.g, COL_REC.b, 255);
    SDL_Rect dot = { LAT_PAD + 8, LAT_PAD + 8, 14, 14 };
    SDL_RenderFillRect(ren, &dot);
    render_text(font_mono, COL_REC, "REC", LAT_PAD + 28, LAT_PAD + 5);
    if (video_filename[0])
        render_text(font_mono, COL_REC, video_filename,
                    LAT_PAD + 70, LAT_PAD + 5);
}

/* ===================================================================
 * Video capture (raw RGBA -> ffmpeg via popen)
 * =================================================================== */

static void start_recording(void)
{
    if (recording) return;
    time_t t = time(NULL);
    struct tm *lt = localtime(&t);
    if (lt) strftime(video_filename, sizeof video_filename,
                     "phonons_%Y%m%d_%H%M%S.mp4", lt);
    else    snprintf(video_filename, sizeof video_filename, "phonons.mp4");

    char cmd[640];
    snprintf(cmd, sizeof cmd,
        "ffmpeg -y -f rawvideo -pix_fmt rgba -s %dx%d -r 60 -i - "
        "-c:v libx264 -preset veryfast -pix_fmt yuv420p "
        "-loglevel warning '%s' 2>/dev/null",
        WIN_W, WIN_H, video_filename);

    video_pipe = popen(cmd, "w");
    if (!video_pipe) {
        fprintf(stderr, "start_recording: popen failed\n");
        return;
    }
    if (!video_buffer)
        video_buffer = malloc((size_t)WIN_W * (size_t)WIN_H * 4u);
    if (!video_buffer) {
        pclose(video_pipe); video_pipe = NULL;
        fprintf(stderr, "start_recording: out of memory\n");
        return;
    }
    recording = true;
}

static void stop_recording(void)
{
    if (!recording) return;
    if (video_pipe) { pclose(video_pipe); video_pipe = NULL; }
    recording = false;
}

static void capture_frame(void)
{
    if (!recording || !video_pipe || !video_buffer) return;
    if (SDL_RenderReadPixels(ren, NULL, SDL_PIXELFORMAT_RGBA32,
                             video_buffer, WIN_W * 4) != 0) {
        stop_recording();
        return;
    }
    const size_t need = (size_t)WIN_W * (size_t)WIN_H * 4u;
    if (fwrite(video_buffer, 1, need, video_pipe) != need)
        stop_recording();   /* ffmpeg likely died */
}

/* ===================================================================
 * Font loading
 * =================================================================== */

static const char *find_existing_path(const char *const *paths)
{
    for (int i = 0; paths[i]; ++i) {
        FILE *f = fopen(paths[i], "rb");
        if (f) { fclose(f); return paths[i]; }
    }
    return NULL;
}

static int load_fonts(void)
{
    static const char *sans_paths[] = {
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        NULL,
    };
    static const char *mono_paths[] = {
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
        "/Library/Fonts/Courier New.ttf",
        NULL,
    };
    const char *sans = find_existing_path(sans_paths);
    const char *mono = find_existing_path(mono_paths);
    if (!sans || !mono) {
        fprintf(stderr,
            "Could not locate DejaVu fonts. On Debian/Ubuntu:\n"
            "    sudo apt install fonts-dejavu fonts-dejavu-core\n");
        return -1;
    }
    font_lbl   = TTF_OpenFont(sans, 14);
    font_title = TTF_OpenFont(sans, 17);
    font_mono  = TTF_OpenFont(mono, 13);
    if (!font_lbl || !font_title || !font_mono) {
        fprintf(stderr, "TTF_OpenFont: %s\n", TTF_GetError());
        return -1;
    }
    return 0;
}

/* ===================================================================
 * Main
 * =================================================================== */

int main(int argc, char **argv)
{
    (void)argc; (void)argv;
    srand(42);

    /* Don't crash if ffmpeg dies mid-stream. */
    signal(SIGPIPE, SIG_IGN);

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
        return 1;
    }
    if (TTF_Init() != 0) {
        fprintf(stderr, "TTF_Init: %s\n", TTF_GetError());
        SDL_Quit(); return 1;
    }
    win = SDL_CreateWindow("crystal_phonons",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        WIN_W, WIN_H, SDL_WINDOW_SHOWN);
    if (!win) {
        fprintf(stderr, "SDL_CreateWindow: %s\n", SDL_GetError());
        TTF_Quit(); SDL_Quit(); return 1;
    }
    ren = SDL_CreateRenderer(win, -1,
        SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!ren) {
        fprintf(stderr, "SDL_CreateRenderer: %s\n", SDL_GetError());
        SDL_DestroyWindow(win); TTF_Quit(); SDL_Quit(); return 1;
    }
    SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_BLEND);

    if (load_fonts() != 0) {
        SDL_DestroyRenderer(ren); SDL_DestroyWindow(win);
        TTF_Quit(); SDL_Quit(); return 1;
    }

    init_sliders();
    zero_state();
    compute_forces();

    bool running    = true;
    long diag_step  = 0;

    while (running) {
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT) {
                running = false;
            } else if (ev.type == SDL_KEYDOWN) {
                const SDL_Keycode k = ev.key.keysym.sym;
                switch (k) {
                case SDLK_q: case SDLK_ESCAPE: running = false; break;
                case SDLK_SPACE: paused = !paused; break;
                case SDLK_r:     zero_state(); compute_forces(); break;
                case SDLK_d:     color_mode = (color_mode + 1) % 3; break;
                case SDLK_t:     thermalize(g_kT_target); break;
                case SDLK_v:
                    if (recording) stop_recording(); else start_recording();
                    break;
                case SDLK_1:
                    zero_state();
                    excite_plane_wave((int)g_kn, 0, 0.0, 1.0, g_amp);
                    compute_forces();
                    break;
                case SDLK_2:
                    zero_state();
                    excite_plane_wave((int)g_kn, 0, 1.0, 0.0, g_amp);
                    compute_forces();
                    break;
                case SDLK_3:
                    zero_state();
                    excite_plane_wave(N/2, N/2, 1.0, 1.0, g_amp);
                    compute_forces();
                    break;
                default: break;
                }
            } else {
                handle_mouse_event(&ev);
            }
        }

        if (!paused)
            for (int s = 0; s < SUBSTEPS; ++s) verlet_step();

        /* refresh diagnostics every few steps to keep readouts steady */
        if (step_no - diag_step >= 5 || paused) {
            compute_energy(&diag_KE, &diag_PE);
            diag_kT_eff = diag_KE / (double)(N * N);
            diag_cv     = cv_quantum(g_kT_target);
            diag_step   = step_no;
        }

        SDL_SetRenderDrawColor(ren, 18, 18, 22, 255);
        SDL_RenderClear(ren);

        /* vertical separator between lattice and panel */
        SDL_SetRenderDrawColor(ren, 60, 60, 70, 255);
        SDL_RenderDrawLine(ren, PANEL_X - 10, 10, PANEL_X - 10, WIN_H - 10);

        draw_lattice();
        draw_dispersion();
        draw_sliders();
        draw_keybindings();
        draw_readouts();
        draw_recording_badge();

        SDL_RenderPresent(ren);
        capture_frame();
    }

    if (recording) stop_recording();
    free(video_buffer);

    if (font_lbl)   TTF_CloseFont(font_lbl);
    if (font_title) TTF_CloseFont(font_title);
    if (font_mono)  TTF_CloseFont(font_mono);
    TTF_Quit();
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
