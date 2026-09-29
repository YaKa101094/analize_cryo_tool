"""
analysis_krueger_ics.py – ICS-Auswertung (Methode B) für Analyse-Krüger
=========================================================================
Enthält die Klasse KruegerICSDialog. Portiert die Spotzahl-Pipeline aus
ics_auswertung.ipynb (Willi_Analyse_Tool/Software Willi) in den ImageViewer,
angewendet auf die dort bereits geladenen Datensätze (kein eigener
.img-Parser nötig).

Methode (siehe Notebook für die vollständige Herleitung):
  1. Räumliche Autokorrelationsfunktion g(xi, eta) des Bildes via FFT
     (Wiener-Khinchin), unbiased normiert über die Anzahl überlappender
     Pixel je Verschiebung.
  2. 2D-Gauß-Fit von g im Fenster um den Zero-Lag (g(0,0) immer aus dem
     Fit ausgeschlossen — Schrotrausch-Spike), in zwei Varianten: freies a
     und auf die physikalische PSF-Breite fixiertes a.
  3. Robuste Untergrundschätzung direkt aus dem Bild (mehrere wählbare
     Verfahren, siehe BACKGROUND_METHOD).
  4. Verdünnungskorrektur q² und daraus Spotdichte/-anzahl je Bild.
  5. Plausibilitätsprüfung (Konvergenz, endliche/positive g0, a im
     erwarteten Bereich, Spotdichte < RHO_MAX_PER_PX2) statt reiner
     numerischer Konvergenz.
  6. Optionale cv-Korrektur der Helligkeitsheterogenität
     (N_korrigiert = (1+cv²)·N, siehe Notebook Kapitel 5).

Nicht übernommen (im Notebook Zusatzabschnitte, hier bewusst weggelassen,
da sie einen eigenständigen Mehr-Proben-Workflow mit Probenmetadaten
voraussetzen, der beim Arbeiten mit den im ImageViewer bereits geladenen
Datensätzen keine Entsprechung hat): probenweise Sammel-CSV über mehrere
Messreihen hinweg (Abschnitt 7, "methodeB_proben_summary.csv") und der
Systematikvergleich der Untergrundschätzer (Abschnitt 8). Der Export hier
liefert stattdessen eine einzelne Ergebnistabelle für die aktuell
geladenen Bilder.

Ursprüngliche Analyse-Methode (ICS-Pipeline, Untergrundschätzer,
cv-Korrektur) entwickelt von Wilhelm Krüger.

Autor: Yannik Kasprzak, Institut für Physik, Universität zu Lübeck
"""

import csv
import os
import tkinter as tk
from tkinter import ttk

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve

from image_processing import get_channel_array, timestamped_filename, unique_output_path

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    _matplotlib_ok = True
except Exception as _e:
    print(f"[matplotlib import failed] {_e}")
    _matplotlib_ok = False

# Feste, im Notebook nicht als Kernparameter behandelte Einstellungen — siehe
# ics_auswertung.ipynb, Zelle 3, für die Herleitung.
A_PLAUSIBLE_MIN_FACTOR = 0.3
A_PLAUSIBLE_MAX_FACTOR = 3.0
A_FIT_BOUND_MIN = 0.1
A_FIT_BOUND_MAX_FACTOR = 5.0
G0_MIN = 1e-6
RHO_MAX_PER_PX2 = 1.0
Q2_MIN_THRESHOLD = 0.01
BACKGROUND_MARGINAL_TOLERANCE = 1.0

CHANNEL_OPTIONS = [
    ("Summe (APD1+APD2)", "sum"),
    ("APD1 (detector0)", "detector0"),
    ("APD2 (detector1)", "detector1"),
]

BACKGROUND_METHODS = [
    ("Median", "median"),
    ("Perzentil", "percentile"),
    ("Modus", "mode"),
    ("Dunkle Bereiche", "dark_regions"),
    ("Fester Wert", "fixed"),
]


# ── ICS-Pipeline (portiert aus dem Notebook) ────────────────────────────────

def _estimate_background_dark_regions(image, sigma_px, frac):
    smoothed = gaussian_filter(image.astype(np.float64), sigma=sigma_px)
    threshold = np.percentile(smoothed, frac * 100.0)
    mask = smoothed <= threshold
    vals = image[mask]
    if vals.size < 2:
        return np.nan
    return float(vals.mean())


def _estimate_background_mean(image, method, percentile, dark_frac, fixed_value, sigma_px):
    if method == "median":
        return float(np.median(image))
    if method == "percentile":
        return float(np.percentile(image, percentile))
    if method == "mode":
        vals, counts = np.unique(image, return_counts=True)
        return float(vals[counts.argmax()])
    if method == "dark_regions":
        return _estimate_background_dark_regions(image, sigma_px, dark_frac)
    if method == "fixed":
        return float(fixed_value)
    raise ValueError(f"Unbekannte Untergrundmethode: {method!r}")


def _compute_acf(image):
    """Berechnet die räumliche Autokorrelationsfunktion g(xi, eta)."""
    ny, nx = image.shape
    mean_I = image.mean()
    dI = image - mean_I
    raw = fftconvolve(dI, dI[::-1, ::-1], mode="full")

    eta_lags = np.arange(-(ny - 1), ny)
    xi_lags = np.arange(-(nx - 1), nx)
    count_y = ny - np.abs(eta_lags)
    count_x = nx - np.abs(xi_lags)
    counts = np.outer(count_y, count_x)

    g = (raw / counts) / mean_I ** 2
    return g, xi_lags, eta_lags


def _gaussian_2d(xy, g0, a, ginf):
    xi, eta = xy
    return g0 * np.exp(-(xi ** 2 + eta ** 2) / a ** 2) + ginf


def _gaussian_1d(r, g0, a, ginf):
    return g0 * np.exp(-(r ** 2) / a ** 2) + ginf


def _extract_fit_window(g, xi_lags, eta_lags, fit_radius):
    cy = np.where(eta_lags == 0)[0][0]
    cx = np.where(xi_lags == 0)[0][0]
    r = fit_radius
    sub_g = g[cy - r:cy + r + 1, cx - r:cx + r + 1]
    sub_xi = xi_lags[cx - r:cx + r + 1]
    sub_eta = eta_lags[cy - r:cy + r + 1]
    return sub_g, sub_xi, sub_eta


def _radial_profile(sub_g, sub_xi, sub_eta):
    XI, ETA = np.meshgrid(sub_xi, sub_eta)
    R = np.sqrt(XI ** 2 + ETA ** 2)
    r_flat, g_flat = R.ravel(), sub_g.ravel()
    r_bins = np.round(r_flat).astype(int)
    max_r = r_bins.max()
    r_vals = np.arange(0, max_r + 1)
    g_means = np.array([g_flat[r_bins == rv].mean() for rv in r_vals])
    return r_vals, g_means


def _check_free_fit_plausibility(g0, a, converged, w_value_px):
    if not converged:
        return False, "not_converged"
    if not (np.isfinite(g0) and np.isfinite(a)):
        return False, "non_finite"
    if g0 <= G0_MIN:
        return False, "g0_non_positive"
    a_min = A_PLAUSIBLE_MIN_FACTOR * w_value_px
    a_max = A_PLAUSIBLE_MAX_FACTOR * w_value_px
    if not (a_min <= abs(a) <= a_max):
        return False, "a_out_of_range"
    return True, None


def _check_fixed_fit_plausibility(g0, converged):
    if not converged:
        return False, "not_converged"
    if not np.isfinite(g0):
        return False, "non_finite"
    if g0 <= G0_MIN:
        return False, "g0_non_positive"
    return True, None


def _check_rho_plausibility(rho):
    if not np.isfinite(rho) or rho <= 0:
        return False, "rho_non_finite"
    if rho > RHO_MAX_PER_PX2:
        return False, "rho_implausible"
    return True, None


def _fit_acf_gaussian(g, xi_lags, eta_lags, w_value_px, fit_radius, fixed_a=None):
    """Fittet g(xi,eta) im Fenster um den Zero-Lag, g(0,0) immer ausgeschlossen."""
    sub_g, sub_xi, sub_eta = _extract_fit_window(g, xi_lags, eta_lags, fit_radius)
    XI, ETA = np.meshgrid(sub_xi, sub_eta)
    xi_flat, eta_flat, g_flat = XI.ravel(), ETA.ravel(), sub_g.ravel()

    mask = ~((xi_flat == 0) & (eta_flat == 0))
    xi_flat, eta_flat, g_flat = xi_flat[mask], eta_flat[mask], g_flat[mask]

    g0_guess = g_flat.max() - g_flat.min()
    ginf_guess = np.median(g_flat)

    if fixed_a is not None:
        def _model(xy, g0, ginf):
            return _gaussian_2d(xy, g0, fixed_a, ginf)

        try:
            popt, _ = curve_fit(_model, (xi_flat, eta_flat), g_flat,
                                p0=[g0_guess, ginf_guess], maxfev=10000)
            g0, ginf = popt
            a = fixed_a
            converged = True
        except RuntimeError:
            g0, ginf = (np.nan, np.nan)
            a = fixed_a
            converged = False
        success, fail_reason = _check_fixed_fit_plausibility(g0, converged)
    else:
        a_bound_max = A_FIT_BOUND_MAX_FACTOR * fit_radius
        bounds = ([-np.inf, A_FIT_BOUND_MIN, -np.inf], [np.inf, a_bound_max, np.inf])
        try:
            popt, _ = curve_fit(
                _gaussian_2d, (xi_flat, eta_flat), g_flat,
                p0=[g0_guess, w_value_px, ginf_guess], bounds=bounds, maxfev=10000,
            )
            g0, a, ginf = popt
            converged = True
        except RuntimeError:
            g0, a, ginf = (np.nan, np.nan, np.nan)
            converged = False
        success, fail_reason = _check_free_fit_plausibility(g0, a, converged, w_value_px)

    return {
        "g0": g0, "a": abs(a) if np.isfinite(a) else a, "ginf": ginf,
        "fit_success": success, "fail_reason": fail_reason,
    }


def _process_image_ics(img, pixel_size_m, config):
    """Führt die komplette ICS-Pipeline für ein bereits geladenes Bild-Array aus."""
    w_value_px = config["w0_m"] / pixel_size_m
    sigma_psf_px = w_value_px / 2.0
    mean_I = img.mean()

    bg_mean = _estimate_background_mean(
        img, config["bg_method"], config["bg_percentile"],
        config["bg_dark_frac"], config["bg_fixed_value"], sigma_psf_px,
    )
    background_marginal = abs(mean_I - bg_mean) < BACKGROUND_MARGINAL_TOLERANCE

    g, xi_lags, eta_lags = _compute_acf(img)
    fit_free = _fit_acf_gaussian(g, xi_lags, eta_lags, w_value_px, config["fit_radius"])
    fit_fixed = _fit_acf_gaussian(g, xi_lags, eta_lags, w_value_px, config["fit_radius"],
                                  fixed_a=w_value_px)

    q2 = ((mean_I - bg_mean) / mean_I) ** 2 if mean_I > 0 else np.nan
    q2_ok = np.isfinite(q2) and q2 >= Q2_MIN_THRESHOLD
    area_px2 = img.shape[0] * img.shape[1]

    def _finish(fit):
        success = fit["fit_success"]
        fail_reason = fit["fail_reason"]
        if success and np.isfinite(q2) and q2 > 0:
            g0_corr = fit["g0"] / q2
            rho = 1.0 / (g0_corr * np.pi * fit["a"] ** 2)
            n_total = rho * area_px2
            rho_ok, rho_reason = _check_rho_plausibility(rho)
            if not rho_ok:
                success = False
                fail_reason = rho_reason
        else:
            if success:
                success = False
                fail_reason = "q2_invalid"
            g0_corr, rho, n_total = np.nan, np.nan, np.nan
        return g0_corr, rho, n_total, success, fail_reason

    g0_corr_free, rho_free, n_free, success_free, reason_free = _finish(fit_free)
    g0_corr_fixed, rho_fixed, n_fixed, success_fixed, reason_fixed = _finish(fit_fixed)

    cv_factor = 1.0
    if config["apply_cv_correction"]:
        cv_factor = 1.0 + config["cv_assumed"] ** 2

    return {
        "image": img, "pixel_size_m": pixel_size_m, "w_value_px": w_value_px,
        "sigma_psf_px": sigma_psf_px, "mean_intensity": mean_I,
        "background_mean": bg_mean, "background_marginal": background_marginal,
        "q2": q2, "q2_ok": q2_ok,
        "g": g, "xi_lags": xi_lags, "eta_lags": eta_lags,
        "fit_free": fit_free, "fit_fixed": fit_fixed,
        "g0_corrected_free": g0_corr_free, "rho_free_px2": rho_free,
        "n_free": n_free, "n_free_cv": n_free * cv_factor if np.isfinite(n_free) else np.nan,
        "fit_success_free": success_free, "fail_reason_free": reason_free,
        "g0_corrected_fixed": g0_corr_fixed, "rho_fixed_px2": rho_fixed,
        "n_fixed": n_fixed, "n_fixed_cv": n_fixed * cv_factor if np.isfinite(n_fixed) else np.nan,
        "fit_success_fixed": success_fixed, "fail_reason_fixed": reason_fixed,
        "cv_factor": cv_factor,
    }


# ── Dialog / Tab ─────────────────────────────────────────────────────────────

class KruegerICSDialog:
    """ICS-Auswertung (Methode B) – Spotzahl aus der Autokorrelation.

    Parameters
    ----------
    parent            : tk.Frame  Tab-Frame im Haupt-Notebook.
    datasets          : list[dict]  Geladene Bilddatensätze.
    get_output_dir    : callable    Gibt den Ausgabepfad zurück (str).
    get_bg_correction : callable    Liefert bool: perzentilbasierte
                                    Hintergrundkorrektur des Hauptfensters an/aus.
    on_close          : callable | None  Wird vom "Schließen"-Button aufgerufen.
    """

    def __init__(self, parent, datasets, get_output_dir, get_bg_correction, on_close=None):
        self._datasets = datasets
        self._get_output_dir = get_output_dir
        self._get_bg_correction = get_bg_correction

        self._results = {}
        self._index = 0
        self._plot = {"fig": None, "canvas": None}

        self.win = parent
        close_cmd = on_close if on_close is not None else self.win.destroy

        if not _matplotlib_ok:
            tk.Label(self.win, text="matplotlib konnte nicht importiert werden.",
                     font=("Arial", 11, "bold")).pack(pady=(22, 4))
            tk.Button(self.win, text="Schließen", command=close_cmd, width=12).pack(pady=14)
            return

        tk.Label(self.win, text="ICS-Auswertung (Methode B) – Spotzahl aus Autokorrelation",
                 font=("Arial", 12, "bold")).pack(pady=(10, 4))

        self._build_param_frame()

        nav_frame = tk.Frame(self.win)
        nav_frame.pack(pady=(4, 2))
        self._prev_button = tk.Button(nav_frame, text="< Zurück", command=self._on_prev, width=10)
        self._prev_button.pack(side="left", padx=4)
        self._progress_label = tk.Label(nav_frame, text="Kein Bild analysiert", width=16)
        self._progress_label.pack(side="left", padx=4)
        self._next_button = tk.Button(nav_frame, text="Weiter >", command=self._on_next, width=10)
        self._next_button.pack(side="left", padx=4)

        self._info_label = tk.Label(self.win, text="", justify="left", font=("Courier", 9))
        self._info_label.pack(pady=(0, 4))

        self._plot_frame = tk.Frame(self.win, bg="#1a1a1a")
        self._plot_frame.pack(fill="both", expand=True, padx=12, pady=(0, 4))

        btn_frame = tk.Frame(self.win)
        btn_frame.pack(pady=6)
        tk.Button(btn_frame, text="Analyse starten", command=self._run_analysis,
                  width=16).pack(side="left", padx=6)
        tk.Button(btn_frame, text="Ergebnisse exportieren (CSV)", command=self._save_results_csv,
                  width=22).pack(side="left", padx=6)
        tk.Button(btn_frame, text="Bild speichern", command=self._save_plot_png,
                  width=14).pack(side="left", padx=6)
        tk.Button(btn_frame, text="Schließen", command=close_cmd,
                  width=12).pack(side="left", padx=6)

    # ── UI-Aufbau ─────────────────────────────────────────────────────────────

    def _build_param_frame(self):
        frame = tk.LabelFrame(self.win, text="Parameter", padx=8, pady=4)
        frame.pack(fill="x", padx=12, pady=4)

        self._channel_var = tk.StringVar(value=CHANNEL_OPTIONS[0][0])
        self._w0_var = tk.StringVar(value="244.6")
        self._scan_um_var = tk.StringVar(value="20.0")
        self._fit_radius_var = tk.StringVar(value="15")
        self._bg_method_var = tk.StringVar(value=BACKGROUND_METHODS[3][0])
        self._bg_percentile_var = tk.StringVar(value="1.0")
        self._bg_dark_frac_var = tk.StringVar(value="0.05")
        self._bg_fixed_var = tk.StringVar(value="7.0")
        self._cv_correction_var = tk.BooleanVar(value=True)
        self._cv_assumed_var = tk.StringVar(value="0.89")

        input_frame = tk.Frame(frame)
        input_frame.pack(side="left", anchor="n")

        tk.Label(input_frame, text="Kanal:", anchor="e").grid(
            row=0, column=0, sticky="e", padx=6, pady=2)
        ttk.Combobox(input_frame, textvariable=self._channel_var, state="readonly", width=18,
                    values=[label for label, _ in CHANNEL_OPTIONS]).grid(
            row=0, column=1, sticky="w", padx=6, pady=2)

        row_defs = [
            ("PSF-Strahltaille w₀ (nm):", self._w0_var),
            ("Scanbereich (µm, quadratisch):", self._scan_um_var),
            ("Fit-Radius (px):", self._fit_radius_var),
        ]
        for idx, (lbl_text, var) in enumerate(row_defs, start=1):
            tk.Label(input_frame, text=lbl_text, anchor="e").grid(
                row=idx, column=0, sticky="e", padx=6, pady=2)
            tk.Entry(input_frame, textvariable=var, width=10).grid(
                row=idx, column=1, sticky="w", padx=6, pady=2)

        r = len(row_defs) + 1
        tk.Label(input_frame, text="Untergrundmethode:", anchor="e").grid(
            row=r, column=0, sticky="e", padx=6, pady=2)
        ttk.Combobox(input_frame, textvariable=self._bg_method_var, state="readonly", width=18,
                    values=[label for label, _ in BACKGROUND_METHODS]).grid(
            row=r, column=1, sticky="w", padx=6, pady=2)

        bg_row_defs = [
            ("Perzentil (nur 'Perzentil'):", self._bg_percentile_var),
            ("Flächenanteil (nur 'Dunkle Bereiche'):", self._bg_dark_frac_var),
            ("Fester Wert (nur 'Fester Wert'):", self._bg_fixed_var),
        ]
        for i, (lbl_text, var) in enumerate(bg_row_defs, start=r + 1):
            tk.Label(input_frame, text=lbl_text, anchor="e").grid(
                row=i, column=0, sticky="e", padx=6, pady=2)
            tk.Entry(input_frame, textvariable=var, width=10).grid(
                row=i, column=1, sticky="w", padx=6, pady=2)

        cv_row = r + len(bg_row_defs) + 1
        tk.Checkbutton(input_frame, text="cv-Korrektur der Helligkeitsheterogenität",
                       variable=self._cv_correction_var).grid(
            row=cv_row, column=0, columnspan=2, sticky="w", padx=4, pady=(6, 0))
        tk.Label(input_frame, text="cv (angenommen):", anchor="e").grid(
            row=cv_row + 1, column=0, sticky="e", padx=6, pady=2)
        tk.Entry(input_frame, textvariable=self._cv_assumed_var, width=10).grid(
            row=cv_row + 1, column=1, sticky="w", padx=6, pady=2)

        tk.Frame(frame, width=1, bg="#444444").pack(
            side="left", fill="y", padx=(12, 10), pady=2)

        desc_text = (
            "g(ξ,η) = g₀·exp(-(ξ²+η²)/a²) + g∞, gefittet an die räumliche\n"
            "Autokorrelation. g(0,0) wird immer ausgeschlossen (Schrotrausch-\n"
            "Spike). Zwei Varianten: a frei, oder a fest = w₀ in Pixeln.\n\n"
            "Untergrund   Direkt aus dem Bild geschätzt. 'Dunkle Bereiche'\n"
            "               ist robust bei dichter Belegung, wo der Median\n"
            "               bereits im Signal liegt (siehe Notebook Kap. 6).\n\n"
            "cv-Korrektur  N_korrigiert = (1+cv²)·N kompensiert die durch\n"
            "               Helligkeits-Heterogenität der Moleküle bedingte\n"
            "               Unterschätzung. cv=0,89 = isotrop orientierte,\n"
            "               eingefrorene Dipole (theoretisch hergeleitet).\n\n"
            "fit_success prüft mehr als Konvergenz: g₀>0, a im plausiblen\n"
            "Bereich um w₀ (nur freies a), Spotdichte < 1/px² (Entartung)."
        )
        tk.Label(frame, text=desc_text, justify="left", anchor="nw",
                 font=("Arial", 8), fg="#aaaaaa").pack(
            side="left", anchor="n", pady=4)

    # ── Analyse ───────────────────────────────────────────────────────────────

    def _channel_key(self):
        label = self._channel_var.get()
        for lbl, key in CHANNEL_OPTIONS:
            if lbl == label:
                return key
        return "sum"

    def _bg_method_key(self):
        label = self._bg_method_var.get()
        for lbl, key in BACKGROUND_METHODS:
            if lbl == label:
                return key
        return "dark_regions"

    def _run_analysis(self):
        if not self._datasets:
            self._info_label.config(text="Keine Bilder geladen.")
            return
        try:
            w0_m = float(self._w0_var.get().replace(",", ".")) * 1e-9
            scan_m = float(self._scan_um_var.get().replace(",", ".")) * 1e-6
            fit_radius = int(self._fit_radius_var.get())
            bg_percentile = float(self._bg_percentile_var.get().replace(",", "."))
            bg_dark_frac = float(self._bg_dark_frac_var.get().replace(",", "."))
            bg_fixed = float(self._bg_fixed_var.get().replace(",", "."))
            cv_assumed = float(self._cv_assumed_var.get().replace(",", "."))
        except ValueError:
            self._info_label.config(text="Ungültige Eingabe in den Parametern.")
            return
        if w0_m <= 0 or scan_m <= 0 or fit_radius < 1:
            self._info_label.config(
                text="w₀ und Scanbereich müssen > 0 sein; Fit-Radius muss >= 1 sein.")
            return

        config = {
            "w0_m": w0_m, "fit_radius": fit_radius,
            "bg_method": self._bg_method_key(), "bg_percentile": bg_percentile,
            "bg_dark_frac": bg_dark_frac, "bg_fixed_value": bg_fixed,
            "apply_cv_correction": self._cv_correction_var.get(),
            "cv_assumed": cv_assumed,
        }
        channel = self._channel_key()
        bg_correction = self._get_bg_correction()

        self._results = {}
        for i, ds in enumerate(self._datasets):
            img = get_channel_array(ds, channel, bg_correction=bg_correction).astype(np.float64)
            pixel_size_m = scan_m / img.shape[1] if img.shape[1] else scan_m
            result = _process_image_ics(img, pixel_size_m, config)
            result["filename"] = ds["name"]
            self._results[i] = result
        self._index = 0
        self._redraw()

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_prev(self):
        if self._index > 0:
            self._index -= 1
            self._redraw()

    def _on_next(self):
        if self._index < len(self._results) - 1:
            self._index += 1
            self._redraw()

    # ── Darstellung ───────────────────────────────────────────────────────────

    def _redraw(self):
        if not self._results:
            self._progress_label.config(text="Kein Bild analysiert")
            self._info_label.config(
                text="Noch keine Analyse gestartet — Parameter prüfen und \"Analyse starten\" klicken.")
            return

        r = self._results[self._index]

        if self._plot["fig"] is None:
            fig = Figure(figsize=(11, 3.4), dpi=90, facecolor="#1a1a1a")
            canvas = FigureCanvasTkAgg(fig, master=self._plot_frame)
            canvas.get_tk_widget().pack(fill="both", expand=True)
            self._plot["fig"] = fig
            self._plot["canvas"] = canvas
        fig = self._plot["fig"]
        fig.clear()

        axes = fig.subplots(1, 4)
        for ax in axes:
            ax.set_facecolor("#1a1a1a")
            ax.tick_params(colors="white", labelsize=7)

        n_free_txt = f"{r['n_free']:.1f}" if np.isfinite(r["n_free"]) else "–"
        n_fixed_txt = f"{r['n_fixed']:.1f}" if np.isfinite(r["n_fixed"]) else "–"
        axes[0].imshow(r["image"], cmap="inferno", origin="upper")
        axes[0].set_title(
            f"Bild (mean={r['mean_intensity']:.2f})\nN(frei)={n_free_txt}  N(fix)={n_fixed_txt}",
            color="white", fontsize=8)
        axes[0].set_xlabel("x (px)", color="white", fontsize=7)
        axes[0].set_ylabel("y (px)", color="white", fontsize=7)

        sub_g, sub_xi, sub_eta = _extract_fit_window(
            r["g"], r["xi_lags"], r["eta_lags"], self._current_fit_radius())
        XI, ETA = np.meshgrid(sub_xi, sub_eta)

        for ax, fit, label in zip(axes[1:3], [r["fit_free"], r["fit_fixed"]], ["a frei", "a fixiert"]):
            model = _gaussian_2d((XI, ETA), fit["g0"], fit["a"], fit["ginf"])
            ax.imshow(sub_g, cmap="viridis", origin="lower",
                      extent=[sub_xi.min(), sub_xi.max(), sub_eta.min(), sub_eta.max()])
            ax.contour(XI, ETA, model, levels=6, colors="red", linewidths=1)
            status = "OK" if fit["fit_success"] else f"FAIL ({fit.get('fail_reason', '?')})"
            ax.set_title(
                f"2D: {label} [{status}]\ng0={fit['g0']:.3f}, a={fit['a']:.2f}",
                color="white", fontsize=8)
            ax.set_xlabel("ξ (px)", color="white", fontsize=7)
            ax.set_ylabel("η (px)", color="white", fontsize=7)

        r_vals, g_means = _radial_profile(sub_g, sub_xi, sub_eta)
        r_line = np.linspace(0, r_vals.max(), 200)
        ax = axes[3]
        ax.scatter(r_vals, g_means, s=14, color="white", label="Messdaten")
        ax.plot(r_line, _gaussian_1d(r_line, r["fit_free"]["g0"], r["fit_free"]["a"],
                                     r["fit_free"]["ginf"]), color="tab:blue", label="a frei")
        ax.plot(r_line, _gaussian_1d(r_line, r["fit_fixed"]["g0"], r["fit_fixed"]["a"],
                                     r["fit_fixed"]["ginf"]), color="tab:orange", linestyle="--",
                label="a fixiert")
        ax.set_xlabel("r (px)", color="white", fontsize=7)
        ax.set_ylabel("g(r)", color="white", fontsize=7)
        ax.set_title("1D-Kontrolle (radial gemittelt)", color="white", fontsize=8)
        ax.legend(fontsize=6, labelcolor="white", facecolor="#1a1a1a")

        fig.suptitle(r["filename"], color="white", fontsize=9)
        fig.tight_layout(pad=1.0)
        self._plot["canvas"].draw()

        self._progress_label.config(text=f"Bild {self._index + 1} / {len(self._results)}")
        px_nm = r["pixel_size_m"] * 1e9
        marginal_txt = " ⚠ Untergrund nahe mittl. Intensität" if r["background_marginal"] else ""
        q2_txt = " ⚠ q² unter Schwelle" if not r["q2_ok"] else ""
        self._info_label.config(
            text=(
                f"{r['filename']}  |  Pixelgröße: {px_nm:.1f} nm  |  "
                f"Untergrund: {r['background_mean']:.2f}  |  q²: {r['q2']:.4f}"
                f"{marginal_txt}{q2_txt}"
            )
        )

    def _current_fit_radius(self):
        try:
            return int(self._fit_radius_var.get())
        except ValueError:
            return 15

    # ── Export ────────────────────────────────────────────────────────────────

    def _save_results_csv(self):
        if not self._results:
            return
        output_dir = self._get_output_dir()
        out_path = unique_output_path(output_dir, timestamped_filename("krueger_ics_results", "csv"))

        fieldnames = [
            "filename", "pixel_size_nm", "mean_intensity", "background_mean",
            "background_marginal", "q2", "q2_ok",
            "g0_free", "a_free", "n_free", "n_free_cv", "fit_success_free", "fail_reason_free",
            "g0_fixed", "n_fixed", "n_fixed_cv", "fit_success_fixed", "fail_reason_fixed",
            "cv_factor",
        ]
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(fieldnames)
            for i in sorted(self._results):
                r = self._results[i]
                writer.writerow([
                    r["filename"], r["pixel_size_m"] * 1e9, r["mean_intensity"],
                    r["background_mean"], r["background_marginal"], r["q2"], r["q2_ok"],
                    r["fit_free"]["g0"], r["fit_free"]["a"], r["n_free"], r["n_free_cv"],
                    r["fit_success_free"], r["fail_reason_free"],
                    r["fit_fixed"]["g0"], r["n_fixed"], r["n_fixed_cv"],
                    r["fit_success_fixed"], r["fail_reason_fixed"], r["cv_factor"],
                ])

        self._info_label.config(text=f"Exportiert: {os.path.basename(out_path)}")

    def _save_plot_png(self):
        if self._plot["fig"] is None or self._index not in self._results:
            return
        base = os.path.splitext(self._results[self._index]["filename"])[0]
        out_path = unique_output_path(
            self._get_output_dir(), timestamped_filename(f"krueger_ics_{base}", "png"))
        self._plot["fig"].savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
