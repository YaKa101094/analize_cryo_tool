"""
analysis_spot_zaehler.py – Spot-Zähler für Multi-Molekül-Übersichtsscans
=========================================================================
Enthält die Klasse SpotZaehlerDialog. Portiert die Detektions-Pipeline aus
Spot_Zaehler_v7_patched.ipynb (Willi_Analyse_Tool/spot-zaehler) in den
ImageViewer, angewendet auf die dort bereits geladenen Datensätze (kein
eigener .img-Parser nötig — die Bildkanäle stehen schon als Arrays bereit).

Methode (siehe Notebook für die vollständige Herleitung):
  1. Robuste Hintergrundschätzung (Median + MAD) und Gauß-Glättung mit
     einem an die PSF angepassten matched filter.
  2. Kandidaten-Peaks über lokalen Maxima oberhalb einer robusten Schwelle
     (Median + k·MAD), mit Mindestabstand ~ FWHM der PSF.
  3. 2D-Gauß-Fit pro Kandidat (Subpixel-Position, Helligkeit, Breite, R²).
  4. Automatische Erkennung verschmolzener Doppelspots (zusätzlicher
     Zwei-Peak-Fit fester Breite, wenn der Einzel-Fit deutlich zu breit ist).
  5. Interaktive visuelle Prüfung: Linksklick auf einen Marker schaltet ihn
     akzeptiert/abgelehnt, Linksklick auf leere Stelle fügt einen neuen Spot
     hinzu, Rechtsklick fügt IMMER einen neuen Spot hinzu (auch neben einem
     bestehenden Marker), Bilder können komplett ausgeschlossen werden.
  6. CSV-Export (Zusammenfassung je Bild + Einzel-Spot-Liste).

Nicht übernommen (im Notebook Zusatzabschnitte, hier bewusst weggelassen,
da redundant zu bereits vorhandenen App-Funktionen bzw. nur für den
eigenständigen Notebook-Workflow relevant): eigener .img-Dateileser
(Abschnitt 2 — die App lädt Bilder bereits selbst), Gesamtübersichts-Raster
aller Bilder (Abschnitt 10 — vorhanden im Bildanalyse-Tab) und der
Spotdichte-Vergleichsexport für einen externen Methodenvergleich
(Abschnitt 12).

Ursprüngliche Analyse-Methode (Detektions-Pipeline, Doppelspot-Split,
Review-Logik) entwickelt von Wilhelm Krüger.

Autor: Yannik Kasprzak, Institut für Physik, Universität zu Lübeck
"""

import csv
import os
import tkinter as tk
from tkinter import ttk

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import curve_fit

from image_processing import get_channel_array, timestamped_filename, unique_output_path

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    _matplotlib_ok = True
except Exception as _e:
    print(f"[matplotlib import failed] {_e}")
    _matplotlib_ok = False

# Feste, im Notebook nicht als Kernparameter behandelte Einstellungen —
# siehe Spot_Zaehler_v7_patched.ipynb, Abschnitt 1, für die Herleitung.
FIT_WINDOW_FACTOR = 3.0
SIGMA_FLAG_RANGE = (0.5, 2.0)
SPLIT_ATTEMPT_SIGMA_FACTOR = 1.10
SPLIT_MIN_R2_GAIN = 0.03
SPLIT_SEPARATION_RANGE_FWHM = (0.7, 3.0)
DISPLAY_VMAX_PERCENTILE = 99.5
CLICK_RADIUS_FACTOR = 1.5

CHANNEL_OPTIONS = [
    ("Summe (APD1+APD2)", "sum"),
    ("APD1 (detector0)", "detector0"),
    ("APD2 (detector1)", "detector1"),
]


# ── Detektions-Pipeline (portiert aus dem Notebook) ─────────────────────────

def _sigma_psf_px(w0_m, pixel_size_m_x):
    """Standardabweichung sigma der (Gauss-genaeherten) PSF in Pixeln.

    Fuer ein Gauss-Strahlprofil I(r) = I0 * exp(-2 r^2 / w0^2) gilt:
    sigma = w0 / 2.
    """
    return (w0_m / 2.0) / pixel_size_m_x


def _estimate_background(img):
    """Robuste Hintergrund-/Rauschschaetzung ueber Median und MAD."""
    median = np.median(img)
    mad = np.median(np.abs(img - median))
    return median, 1.4826 * mad  # MAD -> Standardabweichung (Normalverteilung)


def _smooth_matched_filter(img, sigma_px):
    """Gauss-Glaettung mit sigma = PSF-Breite (matched filter)."""
    return ndi.gaussian_filter(img.astype(np.float64), sigma=sigma_px)


def _detect_peaks(img_smoothed, sigma_px, background_median, background_sigma,
                  k_threshold, min_distance_factor):
    """Findet Kandidaten-Peaks im geglaetteten Bild (lokale Maxima oberhalb
    einer robusten Schwelle, mit Mindestabstand ~ FWHM der PSF).

    scipy-eigene Neuimplementierung von skimage.feature.peak_local_max
    (kein zusaetzlicher Abhaengigkeits-Fussabdruck): benachbarte Kandidaten
    mit identischem Maximalwert (Plateaus) werden ueber scipy.ndimage.label
    zu einem einzigen Peak an ihrem Schwerpunkt zusammengefasst.
    """
    fwhm_px = sigma_px * 2.3548
    min_distance = max(1, int(round(fwhm_px * min_distance_factor)))
    threshold_abs = background_median + k_threshold * background_sigma
    window = 2 * min_distance + 1
    local_max = (
        (img_smoothed == ndi.maximum_filter(img_smoothed, size=window))
        & (img_smoothed >= threshold_abs)
    )
    if not np.any(local_max):
        return np.empty((0, 2), dtype=np.intp)
    labeled, n_labels = ndi.label(local_max)
    if n_labels == 0:
        return np.empty((0, 2), dtype=np.intp)
    centroids = ndi.center_of_mass(local_max, labeled, range(1, n_labels + 1))
    return np.round(np.array(centroids)).astype(np.intp)


def _gaussian_2d(coords, amplitude, x0, y0, sigma, offset):
    x, y = coords
    return offset + amplitude * np.exp(
        -((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)
    )


def _fit_spot(img, y0_guess, x0_guess, sigma_guess, window_factor=FIT_WINDOW_FACTOR):
    """2D-Gauss-Fit in einem kleinen Fenster um (x0_guess, y0_guess)."""
    win = max(2, int(round(sigma_guess * window_factor)))
    ny, nx = img.shape
    y_min, y_max = max(0, y0_guess - win), min(ny, y0_guess + win + 1)
    x_min, x_max = max(0, x0_guess - win), min(nx, x0_guess + win + 1)
    sub = img[y_min:y_max, x_min:x_max].astype(np.float64)
    yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
    xdata = np.vstack((xx.ravel(), yy.ravel()))
    ydata = sub.ravel()

    offset0 = np.median(sub)
    amp0 = sub.max() - offset0
    p0 = [max(amp0, 1.0), x0_guess, y0_guess, sigma_guess, offset0]
    try:
        popt, _ = curve_fit(_gaussian_2d, xdata, ydata, p0=p0, maxfev=2000)
        pred = _gaussian_2d(xdata, *popt)
        ss_res = np.sum((ydata - pred) ** 2)
        ss_tot = np.sum((ydata - ydata.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        amplitude, x0, y0, sigma, offset = popt
        return {
            "success": True, "amplitude": amplitude, "x0": x0, "y0": y0,
            "sigma": abs(sigma), "offset": offset, "r2": r2,
        }
    except Exception:
        return {"success": False}


def _fit_all_peaks(img, peaks_yx, sigma_guess, window_factor=FIT_WINDOW_FACTOR):
    return [_fit_spot(img, int(y), int(x), sigma_guess, window_factor) for (y, x) in peaks_yx]


def _fit_double_spot(img, y0_guess, x0_guess, sigma_fixed, window_factor=FIT_WINDOW_FACTOR):
    """Versucht, einen ungewoehnlich breiten Fleck als zwei gleich grosse,
    sich ueberlagernde Molekuele (feste Breite sigma_fixed) zu erklaeren."""
    win = max(3, int(round(sigma_fixed * (window_factor + 1))))
    ny, nx = img.shape
    y_min, y_max = max(0, y0_guess - win), min(ny, y0_guess + win + 1)
    x_min, x_max = max(0, x0_guess - win), min(nx, x0_guess + win + 1)
    sub = img[y_min:y_max, x_min:x_max].astype(np.float64)
    yy, xx = np.mgrid[y_min:y_max, x_min:x_max]

    offset0 = np.median(sub)
    weights = np.clip(sub - offset0, 0, None)
    total_w = weights.sum()
    if total_w <= 0:
        return {"success": False}

    x_mean = (weights * xx).sum() / total_w
    y_mean = (weights * yy).sum() / total_w
    dx, dy = xx - x_mean, yy - y_mean
    cov = np.array([
        [(weights * dx * dx).sum(), (weights * dx * dy).sum()],
        [(weights * dx * dy).sum(), (weights * dy * dy).sum()],
    ]) / total_w
    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = eigvecs[:, -1]  # Hauptachse der staerksten Ausdehnung

    excess = max(eigvals[-1] - sigma_fixed ** 2, (0.3 * sigma_fixed) ** 2)
    d_est = 2.0 * np.sqrt(excess)

    x1_0, y1_0 = x_mean - 0.5 * d_est * direction[0], y_mean - 0.5 * d_est * direction[1]
    x2_0, y2_0 = x_mean + 0.5 * d_est * direction[0], y_mean + 0.5 * d_est * direction[1]
    amp0 = max(sub.max() - offset0, 1.0) / 2.0

    def double_gaussian(coords, amp1, x1, y1, amp2, x2, y2, off):
        x, y = coords
        g1 = amp1 * np.exp(-((x - x1) ** 2 + (y - y1) ** 2) / (2 * sigma_fixed ** 2))
        g2 = amp2 * np.exp(-((x - x2) ** 2 + (y - y2) ** 2) / (2 * sigma_fixed ** 2))
        return off + g1 + g2

    xdata = np.vstack((xx.ravel(), yy.ravel()))
    ydata = sub.ravel()
    p0 = [amp0, x1_0, y1_0, amp0, x2_0, y2_0, offset0]
    try:
        popt, _ = curve_fit(double_gaussian, xdata, ydata, p0=p0, maxfev=4000)
        pred = double_gaussian(xdata, *popt)
        ss_res = np.sum((ydata - pred) ** 2)
        ss_tot = np.sum((ydata - ydata.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        amp1, x1, y1, amp2, x2, y2, off = popt
        return {
            "success": True, "r2": r2,
            "separation_px": float(np.hypot(x1 - x2, y1 - y2)),
            "spots": [
                {"amplitude": amp1, "x0": x1, "y0": y1, "sigma": sigma_fixed, "offset": off},
                {"amplitude": amp2, "x0": x2, "y0": y2, "sigma": sigma_fixed, "offset": off},
            ],
        }
    except Exception:
        return {"success": False}


def _try_split_merged_spot(img, peak_yx, single_fit, sigma_psf_px_val):
    """Prueft, ob ein als 'zu breit' markierter Kandidat plausibel als zwei
    gleich grosse Molekuele erklaerbar ist. Gibt bei Erfolg eine Liste mit
    zwei Spot-Dicts zurueck, sonst None (dann bleibt der Einzel-Fit bestehen)."""
    y, x = peak_yx
    result = _fit_double_spot(img, int(round(y)), int(round(x)), sigma_psf_px_val, FIT_WINDOW_FACTOR)
    if not result["success"]:
        return None

    fwhm_px = sigma_psf_px_val * 2.3548
    sep_min, sep_max = SPLIT_SEPARATION_RANGE_FWHM
    if not (sep_min * fwhm_px <= result["separation_px"] <= sep_max * fwhm_px):
        return None
    if result["r2"] < single_fit.get("r2", 0.0) + SPLIT_MIN_R2_GAIN:
        return None
    return result["spots"]


def _process_image(img, pixel_size_m, config):
    """Fuehrt die komplette Detektions-Pipeline fuer ein bereits geladenes
    Bild-Array aus (siehe Notebook-Abschnitt 6)."""
    sigma_px = _sigma_psf_px(config["w0_m"], pixel_size_m)
    bg_median, bg_sigma = _estimate_background(img)
    smoothed = _smooth_matched_filter(img, sigma_px)
    peaks = _detect_peaks(
        smoothed, sigma_px, bg_median, bg_sigma,
        k_threshold=config["k_threshold"],
        min_distance_factor=config["min_peak_distance_factor"],
    )
    fits = _fit_all_peaks(img, peaks, sigma_px)

    if config["auto_split_merged_spots"]:
        peaks_split, fits_split = [], []
        for (y, x), fit in zip(peaks, fits):
            too_wide = fit["success"] and fit["sigma"] > SPLIT_ATTEMPT_SIGMA_FACTOR * sigma_px
            split_spots = _try_split_merged_spot(img, (y, x), fit, sigma_px) if too_wide else None
            if split_spots is not None:
                for s in split_spots:
                    peaks_split.append((int(round(s["y0"])), int(round(s["x0"]))))
                    fits_split.append({
                        "success": True, "amplitude": s["amplitude"], "x0": s["x0"], "y0": s["y0"],
                        "sigma": s["sigma"], "offset": s["offset"], "r2": fit["r2"], "auto_split": True,
                    })
            else:
                peaks_split.append((y, x))
                fit_copy = dict(fit)
                fit_copy["auto_split"] = False
                fits_split.append(fit_copy)
        peaks, fits = peaks_split, fits_split

    return {
        "image": img,
        "pixel_size_m": pixel_size_m,
        "sigma_psf_px": sigma_px,
        "background_median": bg_median,
        "background_sigma": bg_sigma,
        "peaks_yx": peaks,
        "fits": fits,
    }


# ── Interaktive Prüfsitzung (portiert aus SpotReviewSession) ───────────────

class SpotReviewSession:
    """Verwaltet den Pruefstatus aller Spots EINES Bildes.

    Farbcode/Marker (siehe SpotZaehlerDialog._redraw): gruen = akzeptiert,
    rot = abgelehnt; gestrichelt = verdaechtiger Grenzfall (Fit-sigma weicht
    stark von der PSF-Breite ab); Diamant = manuell hinzugefuegt (erneuter
    Klick entfernt ihn wieder vollstaendig); Quadrat = automatisch als
    Doppelspot erkannt und aufgeteilt.
    """

    def __init__(self, img, sigma_psf_px_val, peaks_yx, fits, filename,
                sigma_flag_range=SIGMA_FLAG_RANGE, click_radius_factor=CLICK_RADIUS_FACTOR):
        self.img = img
        self.sigma_psf_px = sigma_psf_px_val
        self.filename = filename
        self.sigma_flag_range = sigma_flag_range
        self.click_radius_px = sigma_psf_px_val * 2.3548 * click_radius_factor

        self.spots = []
        for (y, x), fit in zip(peaks_yx, fits):
            if fit["success"]:
                sx, sy, sigma, r2, amp = fit["x0"], fit["y0"], fit["sigma"], fit["r2"], fit["amplitude"]
            else:
                sx, sy, sigma, r2, amp = float(x), float(y), sigma_psf_px_val, np.nan, np.nan
            suspicious = not (sigma_flag_range[0] * sigma_psf_px_val <= sigma <= sigma_flag_range[1] * sigma_psf_px_val)
            source = "auto_split" if fit.get("auto_split") else "auto"
            self.spots.append({
                "x": sx, "y": sy, "sigma": sigma, "r2": r2, "amplitude": amp,
                "accepted": True, "source": source, "suspicious": suspicious,
            })
        self.excluded = False

    def _nearest_spot_index(self, x, y):
        if not self.spots:
            return None
        d = [np.hypot(s["x"] - x, s["y"] - y) for s in self.spots]
        i = int(np.argmin(d))
        return i if d[i] <= self.click_radius_px else None

    def handle_click(self, x, y, force_add=False):
        """force_add=True (Rechtsklick) fuegt IMMER einen neuen Spot hinzu,
        auch wenn sich in der Naehe schon ein anderer Marker befindet —
        wichtig fuer eng benachbarte Molekuele."""
        if not force_add:
            idx = self._nearest_spot_index(x, y)
            if idx is not None:
                if self.spots[idx]["source"] == "manual":
                    del self.spots[idx]
                    return ("removed", idx)
                self.spots[idx]["accepted"] = not self.spots[idx]["accepted"]
                return ("toggled", idx)
        fit = _fit_spot(self.img, int(round(y)), int(round(x)), self.sigma_psf_px)
        if fit["success"]:
            sx, sy, sigma, r2, amp = fit["x0"], fit["y0"], fit["sigma"], fit["r2"], fit["amplitude"]
        else:
            sx, sy, sigma, r2, amp = x, y, self.sigma_psf_px, np.nan, np.nan
        suspicious = not (self.sigma_flag_range[0] * self.sigma_psf_px <= sigma <= self.sigma_flag_range[1] * self.sigma_psf_px)
        self.spots.append({
            "x": sx, "y": sy, "sigma": sigma, "r2": r2, "amplitude": amp,
            "accepted": True, "source": "manual", "suspicious": suspicious,
        })
        return ("added", len(self.spots) - 1)

    def confirmed_spots(self):
        return [] if self.excluded else [s for s in self.spots if s["accepted"]]

    def summary(self):
        return {
            "filename": self.filename,
            "excluded": self.excluded,
            "n_auto_detected": sum(1 for s in self.spots if s["source"] in ("auto", "auto_split")),
            "n_auto_split": sum(1 for s in self.spots if s["source"] == "auto_split"),
            "n_manual_added": sum(1 for s in self.spots if s["source"] == "manual"),
            "n_confirmed": len(self.confirmed_spots()),
            "n_suspicious_confirmed": sum(1 for s in self.confirmed_spots() if s["suspicious"]),
        }


# ── Dialog / Tab ─────────────────────────────────────────────────────────────

class SpotZaehlerDialog:
    """Spot-Zähler-Analysefenster für Multi-Molekül-Übersichtsscans.

    Parameters
    ----------
    parent            : tk.Tk | tk.Toplevel | tk.Frame  Elternwidget — entweder
                        ein eigenständiges Fenster oder ein Tab-Frame im
                        Haupt-Notebook.
    datasets          : list[dict]          Geladene Bilddatensätze
    get_output_dir    : callable             Gibt den Ausgabepfad zurück (str)
    get_bg_correction : callable             Liefert bool zurück: ob die perzentilbasierte
                                             Hintergrundkorrektur des Hauptfensters aktiv ist
    on_close          : callable | None      Wird vom "Schließen"-Button aufgerufen,
                                             statt parent.destroy() (siehe
                                             ImageViewer._new_tab).
    """

    def __init__(self, parent, datasets, get_output_dir, get_bg_correction, on_close=None):
        self._datasets = datasets
        self._get_output_dir = get_output_dir
        self._get_bg_correction = get_bg_correction

        self._pipeline_out = {}   # index -> _process_image()-Ergebnis
        self._sessions = {}       # index -> SpotReviewSession
        self._index = 0

        self._plot = {"fig": None, "ax": None, "canvas": None, "cid": None}

        self.win = parent
        close_cmd = on_close if on_close is not None else self.win.destroy

        if not _matplotlib_ok:
            tk.Label(self.win, text="matplotlib konnte nicht importiert werden.",
                     font=("Arial", 11, "bold")).pack(pady=(22, 4))
            tk.Button(self.win, text="Schließen", command=close_cmd, width=12).pack(pady=14)
            return

        tk.Label(self.win, text="Spot-Zähler – Multi-Molekül-Übersichtsscans",
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
        self._exclude_var = tk.BooleanVar(value=False)
        tk.Checkbutton(nav_frame, text="Bild ausschließen", variable=self._exclude_var,
                       command=self._on_exclude_change).pack(side="left", padx=(12, 4))

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
        self._w0_var = tk.StringVar(value="245.0")
        self._scan_um_var = tk.StringVar(value="20.0")
        self._k_threshold_var = tk.StringVar(value="2.0")
        self._min_dist_var = tk.StringVar(value="1.0")
        self._auto_split_var = tk.BooleanVar(value=True)

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
            ("Detektionsschwelle k (× Rauschen):", self._k_threshold_var),
            ("Mindestabstand (× FWHM):", self._min_dist_var),
        ]
        for idx, (lbl_text, var) in enumerate(row_defs, start=1):
            tk.Label(input_frame, text=lbl_text, anchor="e").grid(
                row=idx, column=0, sticky="e", padx=6, pady=2)
            tk.Entry(input_frame, textvariable=var, width=10).grid(
                row=idx, column=1, sticky="w", padx=6, pady=2)

        tk.Checkbutton(input_frame, text="Verschmolzene Doppelspots automatisch aufteilen",
                       variable=self._auto_split_var).grid(
            row=len(row_defs) + 1, column=0, columnspan=2, sticky="w", padx=4, pady=(6, 0))

        tk.Frame(frame, width=1, bg="#444444").pack(
            side="left", fill="y", padx=(12, 10), pady=2)

        desc_text = (
            "w₀        1/e²-Radius der PSF. σ = w₀/2, FWHM = 2,3548·σ.\n\n"
            "Scanbereich  Physikalische Kantenlänge des Bildes — für\n"
            "               Pixelgröße (PSF in Pixeln) und µm-Export.\n\n"
            "k          Schwelle = Hintergrund-Median + k·MAD. Größer =\n"
            "               strenger. Lieber niedrig ansetzen und\n"
            "               falsch-positive Spots im Review ablehnen.\n\n"
            "Mindestabstand  1.0 = Beugungsgrenze (zwei Moleküle näher\n"
            "               als eine FWHM sind nicht als zwei Maxima\n"
            "               unterscheidbar).\n\n"
            "Klick auf einen Marker schaltet ihn an/aus, Klick auf\n"
            "leere Stelle fügt einen Spot hinzu, Rechtsklick fügt\n"
            "immer einen neuen Spot hinzu (auch neben einem Marker)."
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

    def _run_analysis(self):
        """Führt die Detektions-Pipeline für alle aktiven Datensätze aus und
        startet eine frische Prüfsitzung je Bild (bestehende manuelle
        Korrekturen gehen dabei verloren — entspricht einem Neustart mit den
        aktuellen Parametern)."""
        if not self._datasets:
            self._info_label.config(text="Keine Bilder geladen.")
            return
        try:
            w0_m = float(self._w0_var.get().replace(",", ".")) * 1e-9
            scan_m = float(self._scan_um_var.get().replace(",", ".")) * 1e-6
            k_threshold = float(self._k_threshold_var.get().replace(",", "."))
            min_dist_factor = float(self._min_dist_var.get().replace(",", "."))
        except ValueError:
            self._info_label.config(text="Ungültige Eingabe in den Parametern.")
            return
        if w0_m <= 0 or scan_m <= 0 or k_threshold <= 0 or min_dist_factor <= 0:
            self._info_label.config(text="Alle Parameter müssen > 0 sein.")
            return

        config = {
            "w0_m": w0_m,
            "k_threshold": k_threshold,
            "min_peak_distance_factor": min_dist_factor,
            "auto_split_merged_spots": self._auto_split_var.get(),
        }
        channel = self._channel_key()
        bg_correction = self._get_bg_correction()

        self._pipeline_out = {}
        self._sessions = {}
        for i, ds in enumerate(self._datasets):
            img = get_channel_array(ds, channel, bg_correction=bg_correction).astype(np.float64)
            pixel_size_m = scan_m / img.shape[1] if img.shape[1] else scan_m
            result = _process_image(img, pixel_size_m, config)
            self._pipeline_out[i] = result
            self._sessions[i] = SpotReviewSession(
                result["image"], result["sigma_psf_px"], result["peaks_yx"], result["fits"],
                filename=ds["name"],
            )
        self._index = 0
        self._redraw()

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_prev(self):
        if self._index > 0:
            self._index -= 1
            self._redraw()

    def _on_next(self):
        if self._index < len(self._sessions) - 1:
            self._index += 1
            self._redraw()

    def _on_exclude_change(self):
        if self._index in self._sessions:
            self._sessions[self._index].excluded = self._exclude_var.get()
            self._redraw()

    def _on_click(self, event):
        if self._index not in self._sessions or event.inaxes != self._plot["ax"] or event.xdata is None:
            return
        session = self._sessions[self._index]
        force_add = (event.button == 3)
        session.handle_click(event.xdata, event.ydata, force_add=force_add)
        self._redraw()

    # ── Darstellung ───────────────────────────────────────────────────────────

    def _redraw(self):
        if not self._sessions:
            self._progress_label.config(text="Kein Bild analysiert")
            self._info_label.config(text="Noch keine Analyse gestartet — Parameter prüfen und \"Analyse starten\" klicken.")
            return

        result = self._pipeline_out[self._index]
        session = self._sessions[self._index]
        img = result["image"]

        if self._plot["fig"] is None:
            self._plot["fig"] = Figure(figsize=(5.5, 5.5), dpi=90, facecolor="#1a1a1a")
            self._plot["ax"] = self._plot["fig"].add_subplot(111)
            self._plot["canvas"] = FigureCanvasTkAgg(self._plot["fig"], master=self._plot_frame)
            self._plot["canvas"].get_tk_widget().pack(fill="both", expand=True)
            self._plot["cid"] = self._plot["canvas"].mpl_connect("button_press_event", self._on_click)

        ax = self._plot["ax"]
        ax.clear()
        ax.set_facecolor("#1a1a1a")

        vmin = result["background_median"]
        vmax = np.percentile(img, DISPLAY_VMAX_PERCENTILE)
        if vmax <= vmin:
            vmax = float(img.max())
        ax.imshow(img, cmap="inferno", origin="upper", vmin=vmin, vmax=vmax)
        ax.set_xlabel("x (Pixel)", color="white", fontsize=9)
        ax.set_ylabel("y (Pixel)", color="white", fontsize=9)
        ax.tick_params(colors="white", labelsize=8)
        ax.set_title(session.filename, color="white", fontsize=10)

        for s in session.spots:
            edgecolor = "limegreen" if s["accepted"] else "red"
            ls = "--" if s["suspicious"] else "-"
            marker = {"manual": "D", "auto_split": "s"}.get(s["source"], "o")
            ax.scatter(
                [s["x"]], [s["y"]], s=90, facecolors="none",
                edgecolors=edgecolor, linewidths=1.8, linestyles=ls, marker=marker,
            )

        summ = session.summary()
        status = "AUSGESCHLOSSEN" if summ["excluded"] else f"{summ['n_confirmed']} bestätigt"
        ax.text(
            0.02, 0.98,
            f"{status} (auto: {summ['n_auto_detected']}, davon Split: {summ['n_auto_split']}, "
            f"manuell: {summ['n_manual_added']})",
            transform=ax.transAxes, va="top", ha="left", color="white", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
        )
        self._plot["fig"].tight_layout(pad=1.0)
        self._plot["canvas"].draw()

        self._exclude_var.set(session.excluded)
        self._progress_label.config(text=f"Bild {self._index + 1} / {len(self._sessions)}")
        px_nm = result["pixel_size_m"] * 1e9
        self._info_label.config(
            text=(
                f"{session.filename}  |  Pixelgröße: {px_nm:.1f} nm  |  "
                f"σ_PSF: {result['sigma_psf_px']:.2f} px  |  "
                f"Hintergrund: {result['background_median']:.1f} ± {result['background_sigma']:.1f}"
            )
        )

    # ── Export ────────────────────────────────────────────────────────────────

    def _save_results_csv(self):
        """Exportiert Zusammenfassung + Einzel-Spot-Liste als CSV (siehe
        Notebook-Abschnitt 11: spotcount_summary.csv / spotcount_spots.csv)."""
        if not self._sessions:
            return
        output_dir = self._get_output_dir()
        summary_path = unique_output_path(output_dir, timestamped_filename("spotzaehler_summary", "csv"))
        spots_path = unique_output_path(output_dir, timestamped_filename("spotzaehler_spots", "csv"))

        with open(summary_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "excluded", "n_auto_detected", "n_auto_split",
                             "n_manual_added", "n_confirmed", "n_suspicious_confirmed"])
            for i in sorted(self._sessions):
                summ = self._sessions[i].summary()
                writer.writerow([summ["filename"], summ["excluded"], summ["n_auto_detected"],
                                 summ["n_auto_split"], summ["n_manual_added"],
                                 summ["n_confirmed"], summ["n_suspicious_confirmed"]])

        with open(spots_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "x_px", "y_px", "x_um", "y_um",
                             "amplitude", "sigma_px", "r2", "source", "suspicious"])
            for i in sorted(self._sessions):
                session = self._sessions[i]
                px_x = self._pipeline_out[i]["pixel_size_m"]
                for s in session.confirmed_spots():
                    writer.writerow([
                        session.filename, s["x"], s["y"],
                        s["x"] * px_x * 1e6, s["y"] * px_x * 1e6,
                        s["amplitude"], s["sigma"], s["r2"], s["source"], s["suspicious"],
                    ])

        self._info_label.config(
            text=f"Exportiert: {os.path.basename(summary_path)}, {os.path.basename(spots_path)}"
        )

    def _save_plot_png(self):
        """Speichert die aktuelle Bildansicht (mit Spot-Markern) als PNG."""
        if self._plot["fig"] is None or self._index not in self._sessions:
            return
        base = os.path.splitext(self._sessions[self._index].filename)[0]
        out_path = unique_output_path(
            self._get_output_dir(), timestamped_filename(f"spotzaehler_{base}", "png"))
        self._plot["fig"].savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
