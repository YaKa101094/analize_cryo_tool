"""
analysis_krueger_zstack.py – Z-Stack-Analyse (PSF-Charakterisierung) für
Analyse-Krüger
=========================================================================
Enthält die Klasse ZStackDialog. Portiert den Kern der PSF-Charakterisierung
aus z-stack.ipynb (Willi_Analyse_Tool/Software Willi): lokalisiert eine
einzelne Bead (Fluoreszenzkügelchen) in jeder Ebene eines Z-Stacks, fittet
pro Ebene einen 2D-Gauß, baut daraus einen drift-korrigierten ROI-Stack und
bestimmt die axiale PSF-Breite (FWHM_z) aus dem zentralen Intensitätsprofil.

Im Gegensatz zu den anderen Analyse-Krüger-Methoden arbeitet dieses
Werkzeug NICHT auf den im Hauptfenster bereits geladenen Datensätzen,
sondern lädt selbst einen Ordner voller .img-Dateien — je eine pro
Z-Ebene, üblicherweise mit "z_<Schrittnummer>" im Dateinamen (siehe
z_aus_dateiname), sonst in alphabetischer Reihenfolge.

Bewusst nicht übernommen (siehe Nutzerentscheidung "Kernfunktion" statt
"Vollständig" — diese Teile bleiben dem Original-Notebook vorbehalten):
manuelle Zentrums-Korrekturen (Zelle 2c-K), interaktive 3D-WebGL-
Visualisierung (Zelle 2f-3D), Parallelprofile bei lateralem Versatz und
Kreismasken-Profil (Methoden B/C aus Zelle 2g, hier nur Methode A:
Zentralpixel), laterale 2D-Gauß-Fit-Ansicht der Fokusebenen (Zelle 2h) und
CSV/LaTeX-Export für pgfplots (Zelle 2i).

Ursprüngliche Analyse-Methode (Lokalisation, Drift-Korrektur, axiale
Profilfits) entwickelt von Wilhelm Krüger.

Autor: Yannik Kasprzak, Institut für Physik, Universität zu Lübeck
"""

import csv
import os
import re
import tkinter as tk
from tkinter import filedialog

import numpy as np
from scipy.ndimage import gaussian_filter, label, center_of_mass
from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator

from image_processing import load_img_dataset, timestamped_filename, unique_output_path

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    _matplotlib_ok = True
except Exception as _e:
    print(f"[matplotlib import failed] {_e}")
    _matplotlib_ok = False

Z_INTERP_FAKTOR = 8


# ── Lokalisation & 2D-Gauß-Fit pro Ebene (portiert aus dem Notebook) ────────

def _z_aus_dateiname(pfad, fallback_index):
    treffer = re.search(r"z_?(\d+)", os.path.basename(pfad), re.IGNORECASE)
    return int(treffer.group(1)) if treffer else fallback_index


def _adaptive_schwelle(bild, glaett_sigma):
    smooth = gaussian_filter(bild.astype(float), sigma=glaett_sigma)
    finite = smooth[np.isfinite(smooth)]
    if finite.size == 0:
        return None, smooth
    peak = float(np.max(finite))
    bg = float(np.percentile(finite, 25))
    if peak <= bg:
        return None, smooth
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    noise = 1.4826 * mad
    kontrast = peak - bg
    schwelle = max(bg + 3.0 * noise, bg + 0.35 * kontrast)
    schwelle = float(np.clip(schwelle, bg + 0.10 * kontrast, bg + 0.85 * kontrast))
    return schwelle, smooth


def _lokalisiere_bead(bild, glaett_sigma):
    """Findet das Bead-Zentrum in einer einzelnen Ebene über eine
    datengetriebene Schwelle. Gibt (cx_px, cy_px, True) oder (None, None, False)."""
    schwelle, smooth = _adaptive_schwelle(bild, glaett_sigma)
    if schwelle is None:
        return None, None, False

    maske = smooth > schwelle
    if maske.sum() < 3 or maske.sum() > 0.8 * maske.size:
        finite = smooth[np.isfinite(smooth)]
        bg = float(np.percentile(finite, 25))
        peak = float(np.max(finite))
        for anteil in (0.30, 0.25, 0.20, 0.15, 0.10):
            maske = smooth > (bg + anteil * (peak - bg))
            if 3 <= maske.sum() <= 0.8 * maske.size:
                break

    regionen, n_reg = label(maske)
    if n_reg == 0:
        return None, None, False
    beste = max(range(1, n_reg + 1), key=lambda i: bild[regionen == i].sum())
    cy, cx = center_of_mass(bild, regionen, beste)
    return cx, cy, True


def _schaetze_fit_radius_px(bild, cx_px, cy_px, r_min, r_max, halb_fenster):
    x_c, y_c = int(round(cx_px)), int(round(cy_px))
    xlo, xhi = max(0, x_c - halb_fenster), min(bild.shape[1], x_c + halb_fenster + 1)
    ylo, yhi = max(0, y_c - halb_fenster), min(bild.shape[0], y_c + halb_fenster + 1)
    roi = bild[ylo:yhi, xlo:xhi].astype(float)

    bg = float(np.percentile(roi, 25))
    signal = np.clip(roi - bg, 0, None)
    if signal.sum() <= 0:
        return int(np.clip(r_min, r_min, r_max))

    yy, xx = np.indices(roi.shape)
    x_mean = float((signal * xx).sum() / signal.sum())
    y_mean = float((signal * yy).sum() / signal.sum())
    r2 = (xx - x_mean) ** 2 + (yy - y_mean) ** 2
    sigma_est_px = float(np.sqrt((signal * r2).sum() / (2.0 * signal.sum())))
    return max(r_min, int(np.clip(np.ceil(2.0 * sigma_est_px), r_min, r_max)))


def _fit_gauss_2d(bild_roi, x_koord, y_koord, cx_start_nm, cy_start_nm,
                  r_fit_px, px_nm, max_abweichung_px):
    """2D-Gauß-Fit mit radialer Maske und drei Qualitätsfiltern."""
    xx, yy = np.meshgrid(x_koord, y_koord)
    n_y, n_x = bild_roi.shape

    cx_roi_px = (cx_start_nm - x_koord[0]) / px_nm
    cy_roi_px = (cy_start_nm - y_koord[0]) / px_nm
    ppx, ppy = np.meshgrid(np.arange(n_x), np.arange(n_y))
    dist = np.sqrt((ppx - cx_roi_px) ** 2 + (ppy - cy_roi_px) ** 2)
    maske = dist <= r_fit_px
    if maske.sum() < 15:
        return None, False

    x_m, y_m, I_m = xx[maske], yy[maske], bild_roi[maske]

    def gauss(xy, I_bg, I0, x0, y0, sx, sy):
        x, y = xy
        return I_bg + I0 * np.exp(-((x - x0) ** 2 / (2 * sx ** 2) + (y - y0) ** 2 / (2 * sy ** 2)))

    r_nm = r_fit_px * px_nm
    p0 = [np.percentile(I_m, 10), I_m.max() - I_m.min(), cx_start_nm, cy_start_nm, 80, 80]
    lo = [0, 0, cx_start_nm - r_nm, cy_start_nm - r_nm, 20, 20]
    hi = [I_m.max(), I_m.max() * 3, cx_start_nm + r_nm, cy_start_nm + r_nm, 350, 350]

    try:
        popt, _ = curve_fit(gauss, (x_m, y_m), I_m, p0=p0, bounds=(lo, hi), maxfev=10000)
        _, I0, x0, y0, sx, sy = popt
        sx, sy = abs(sx), abs(sy)
        popt[4], popt[5] = sx, sy

        if not (20 < sx < 350 and 20 < sy < 350):
            return None, False
        if (abs(x0 - cx_start_nm) / px_nm > max_abweichung_px or
                abs(y0 - cy_start_nm) / px_nm > max_abweichung_px):
            return None, False
        if I0 < 2 * np.std(I_m):
            return None, False
        return popt, True
    except Exception:
        return None, False


def _gauss_z(z, I_bg, I0, z0, sigma_z):
    return I_bg + I0 * np.exp(-((z - z0) ** 2) / (2 * sigma_z ** 2))


def _lorentz_z(z, I_bg, I0, z0, z_R):
    return I_bg + I0 / (1 + ((z - z0) / z_R) ** 2)


def _fit_axial(z_arr, I_arr):
    res = {"popt_g": None, "popt_l": None, "sigma_z": np.nan, "fwhm_z": np.nan,
          "zR": np.nan, "z0": np.nan}
    if len(z_arr) < 5:
        return res
    z0_s = float(z_arr[np.argmax(I_arr)])
    sig_s = float((z_arr[-1] - z_arr[0]) / 4)
    I0_s = float(I_arr.max() - I_arr.min())
    Ibg_s = float(np.percentile(I_arr, 10))
    bounds = ([0, 0, z_arr.min(), 50], [I_arr.max(), I_arr.max() * 3, z_arr.max(), z_arr[-1] - z_arr[0]])
    for modell, key in [(_gauss_z, "g"), (_lorentz_z, "l")]:
        try:
            popt, _ = curve_fit(modell, z_arr, I_arr, p0=[Ibg_s, I0_s, z0_s, sig_s],
                                bounds=bounds, maxfev=10000)
            res[f"popt_{key}"] = popt
        except Exception:
            pass
    if res["popt_g"] is not None:
        res["sigma_z"] = abs(res["popt_g"][3])
        res["fwhm_z"] = 2.355 * res["sigma_z"]
        res["z0"] = res["popt_g"][2]
    if res["popt_l"] is not None:
        res["zR"] = abs(res["popt_l"][3])
    return res


def _interpolierte_zentren(zeilen, n_ebenen):
    cx_arr = np.full(n_ebenen, np.nan)
    cy_arr = np.full(n_ebenen, np.nan)
    for z in zeilen:
        if z["fit_ok"]:
            cx_arr[z["ebene"]] = z["cx_px"]
            cy_arr[z["ebene"]] = z["cy_px"]

    idx_gueltig = np.where(~np.isnan(cx_arr))[0]
    idx_alle = np.arange(n_ebenen)
    if len(idx_gueltig) < 2:
        return None, None, None
    cx_interp = np.interp(idx_alle, idx_gueltig, cx_arr[idx_gueltig])
    cy_interp = np.interp(idx_alle, idx_gueltig, cy_arr[idx_gueltig])
    gueltig = ~np.isnan(cx_arr)
    return cx_interp, cy_interp, gueltig


def _interp_xz(schnitt, z_nm_arr, faktor):
    gueltige = ~np.isnan(schnitt).any(axis=1)
    z_g, sn_g = z_nm_arr[gueltige], schnitt[gueltige, :]
    if len(z_g) < 2:
        return sn_g, z_g
    n_x = sn_g.shape[1]
    x_idx = np.arange(n_x)
    interpolator = RegularGridInterpolator((z_g, x_idx), sn_g, method="linear",
                                           bounds_error=False, fill_value=np.nan)
    z_fein = np.linspace(z_g[0], z_g[-1], (len(z_g) - 1) * faktor + 1)
    xx, zz = np.meshgrid(x_idx, z_fein)
    result = interpolator(np.stack([zz.ravel(), xx.ravel()], axis=1)).reshape(len(z_fein), n_x)
    return result, z_fein


def _normiere(arr):
    vmax, vmin = np.nanmax(arr), np.nanmin(arr)
    return (arr - vmin) / (vmax - vmin) if vmax > vmin else arr


# ── Dialog / Tab ─────────────────────────────────────────────────────────────

class ZStackDialog:
    """Z-Stack-PSF-Analyse: lädt einen Ordner mit einer .img-Datei je
    Z-Ebene und bestimmt Bead-Zentrum, ROI-Drift-Korrektur, x-z/y-z-Schnitt
    und axiale PSF-Breite (FWHM_z, Zentralpixel-Methode).

    Parameters
    ----------
    parent         : tk.Frame  Tab-Frame im Haupt-Notebook.
    get_output_dir : callable  Gibt den Ausgabepfad zurück (str), für
                               Plot-/CSV-Export.
    on_close       : callable | None  Wird vom "Schließen"-Button aufgerufen.
    """

    def __init__(self, parent, get_output_dir, on_close=None):
        self._get_output_dir = get_output_dir
        self._folder = None
        self._zeilen = None
        self._z_nm = None
        self._roi_stack = None
        self._axial = None
        self._plot = {"fig": None, "canvas": None}

        self.win = parent
        close_cmd = on_close if on_close is not None else self.win.destroy

        if not _matplotlib_ok:
            tk.Label(self.win, text="matplotlib konnte nicht importiert werden.",
                     font=("Arial", 11, "bold")).pack(pady=(22, 4))
            tk.Button(self.win, text="Schließen", command=close_cmd, width=12).pack(pady=14)
            return

        tk.Label(self.win, text="Z-Stack – PSF-Charakterisierung aus Bead-Z-Stack",
                 font=("Arial", 12, "bold")).pack(pady=(10, 4))

        self._build_param_frame()

        self._status_label = tk.Label(self.win, text="", justify="left", font=("Courier", 9))
        self._status_label.pack(pady=(0, 4))

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

        self._folder_var = tk.StringVar(value="(kein Ordner gewählt)")
        self._px_nm_var = tk.StringVar(value="200.0")
        self._z_schritt_var = tk.StringVar(value="200.0")
        self._r_fit_min_var = tk.StringVar(value="3")
        self._r_fit_max_var = tk.StringVar(value="20")
        self._fit_pad_var = tk.StringVar(value="3")
        self._max_abw_var = tk.StringVar(value="5")
        self._glaett_sigma_var = tk.StringVar(value="2.0")

        folder_frame = tk.Frame(frame)
        folder_frame.pack(side="left", anchor="n", fill="y")
        tk.Button(folder_frame, text="Ordner wählen...", command=self._choose_folder,
                  width=16).grid(row=0, column=0, sticky="w", padx=6, pady=(2, 6))
        tk.Label(folder_frame, textvariable=self._folder_var, anchor="w",
                 font=("Arial", 8), wraplength=180, justify="left").grid(
            row=1, column=0, sticky="w", padx=6)

        input_frame = tk.Frame(frame)
        input_frame.pack(side="left", anchor="n", padx=(16, 0))

        row_defs = [
            ("Pixelgröße (nm/px):", self._px_nm_var),
            ("Z-Schrittweite (nm):", self._z_schritt_var),
            ("Fit-Radius min (px):", self._r_fit_min_var),
            ("Fit-Radius max (px):", self._r_fit_max_var),
            ("Fit-Rand-Zugabe (px):", self._fit_pad_var),
            ("Max. Zentrumsabweichung (px):", self._max_abw_var),
            ("Glättung Lokalisation (px, σ):", self._glaett_sigma_var),
        ]
        for idx, (lbl_text, var) in enumerate(row_defs):
            tk.Label(input_frame, text=lbl_text, anchor="e").grid(
                row=idx, column=0, sticky="e", padx=6, pady=2)
            tk.Entry(input_frame, textvariable=var, width=10).grid(
                row=idx, column=1, sticky="w", padx=6, pady=2)

        tk.Frame(frame, width=1, bg="#444444").pack(
            side="left", fill="y", padx=(12, 10), pady=2)

        desc_text = (
            "Ordner mit je einer .img-Datei pro Z-Ebene (Dateiname\n"
            "idealerweise mit \"z_<Nummer>\", sonst alphabetische\n"
            "Reihenfolge). Pro Ebene: Bead lokalisieren (adaptive\n"
            "Schwelle) → 2D-Gauß-Fit → drift-korrigierter ROI-Stack.\n\n"
            "x-z/y-z-Schnitt  Schnitt durch die Stapelmitte, in z\n"
            "               interpoliert.\n\n"
            "Axiales Profil  Intensität am ROI-Zentrum über alle\n"
            "               Ebenen, mit Gauß- und Lorentz-Fit für\n"
            "               FWHM_z bzw. Rayleigh-Länge z_R.\n\n"
            "Kernfunktion: keine manuellen Korrekturen, keine\n"
            "interaktive 3D-Ansicht, keine Parallelprofile/laterale\n"
            "2D-Fits (siehe Original-Notebook für den vollen Umfang)."
        )
        tk.Label(frame, text=desc_text, justify="left", anchor="nw",
                 font=("Arial", 8), fg="#aaaaaa").pack(
            side="left", anchor="n", pady=4)

    def _choose_folder(self):
        folder = filedialog.askdirectory(title="Z-Stack-Ordner wählen")
        if folder:
            self._folder = folder
            self._folder_var.set(folder)

    # ── Analyse ───────────────────────────────────────────────────────────────

    def _run_analysis(self):
        if not self._folder or not os.path.isdir(self._folder):
            self._status_label.config(text="Bitte zuerst einen Ordner wählen.")
            return
        try:
            px_nm = float(self._px_nm_var.get().replace(",", "."))
            z_schritt_nm = float(self._z_schritt_var.get().replace(",", "."))
            r_fit_min = int(self._r_fit_min_var.get())
            r_fit_max = int(self._r_fit_max_var.get())
            fit_pad_px = int(self._fit_pad_var.get())
            max_abweichung_px = float(self._max_abw_var.get().replace(",", "."))
            glaett_sigma = float(self._glaett_sigma_var.get().replace(",", "."))
        except ValueError:
            self._status_label.config(text="Ungültige Eingabe in den Parametern.")
            return
        if px_nm <= 0 or z_schritt_nm <= 0 or r_fit_min < 1 or r_fit_max < r_fit_min:
            self._status_label.config(
                text="Pixelgröße, Z-Schrittweite müssen > 0 sein; Fit-Radius-Grenzen ungültig.")
            return

        paths = [
            os.path.join(self._folder, f) for f in os.listdir(self._folder)
            if f.lower().endswith(".img")
        ]
        if not paths:
            self._status_label.config(text=f"Keine .img-Dateien in {self._folder} gefunden.")
            return
        paths.sort(key=lambda p: _z_aus_dateiname(p, 0))

        bilder, z_werte = [], []
        for i, p in enumerate(paths):
            ds = load_img_dataset(p)
            if ds is None:
                continue
            bilder.append(ds["sum"].astype(float))
            z_werte.append(_z_aus_dateiname(p, i) * z_schritt_nm)

        n_ebenen = len(bilder)
        if n_ebenen == 0:
            self._status_label.config(text="Keine der .img-Dateien konnte gelesen werden.")
            return

        z_nm = np.array(z_werte, dtype=float)
        halb_fenster = r_fit_max + fit_pad_px + 2

        zeilen = []
        for i, bild in enumerate(bilder):
            cx_px, cy_px, gefunden = _lokalisiere_bead(bild, glaett_sigma)
            if not gefunden:
                zeilen.append({"ebene": i, "z_nm": z_nm[i], "gefunden": False, "fit_ok": False,
                              "cx_px": np.nan, "cy_px": np.nan, "cx_nm": np.nan, "cy_nm": np.nan,
                              "sigma_x_nm": np.nan, "sigma_y_nm": np.nan,
                              "fwhm_x_nm": np.nan, "fwhm_y_nm": np.nan})
                continue

            r_fit_px = _schaetze_fit_radius_px(bild, cx_px, cy_px, r_fit_min, r_fit_max, halb_fenster)
            n_py, n_px = bild.shape
            rand_px = int(min(cx_px, cy_px, (n_px - 1) - cx_px, (n_py - 1) - cy_px))
            roi_halb_px = int(min(r_fit_px + fit_pad_px, rand_px))

            if roi_halb_px < 4:
                zeilen.append({"ebene": i, "z_nm": z_nm[i], "gefunden": True, "fit_ok": False,
                              "cx_px": cx_px, "cy_px": cy_px, "cx_nm": cx_px * px_nm, "cy_nm": cy_px * px_nm,
                              "sigma_x_nm": np.nan, "sigma_y_nm": np.nan,
                              "fwhm_x_nm": np.nan, "fwhm_y_nm": np.nan})
                continue

            cx_nm, cy_nm = cx_px * px_nm, cy_px * px_nm
            koordinaten = np.arange(n_px) * px_nm
            koordinaten_y = np.arange(n_py) * px_nm
            cx_i = int(np.clip(round(cx_px), roi_halb_px, n_px - 1 - roi_halb_px))
            cy_i = int(np.clip(round(cy_px), roi_halb_px, n_py - 1 - roi_halb_px))
            xlo, xhi = cx_i - roi_halb_px, cx_i + roi_halb_px
            ylo, yhi = cy_i - roi_halb_px, cy_i + roi_halb_px
            roi = bild[ylo:yhi, xlo:xhi]
            x_roi = koordinaten[xlo:xhi]
            y_roi = koordinaten_y[ylo:yhi]

            popt, fit_ok = _fit_gauss_2d(roi, x_roi, y_roi, cx_nm, cy_nm, r_fit_px, px_nm,
                                         max_abweichung_px)
            if fit_ok:
                _, _, x0, y0, sx, sy = popt
                zeilen.append({"ebene": i, "z_nm": z_nm[i], "gefunden": True, "fit_ok": True,
                              "cx_px": x0 / px_nm, "cy_px": y0 / px_nm, "cx_nm": x0, "cy_nm": y0,
                              "sigma_x_nm": sx, "sigma_y_nm": sy,
                              "fwhm_x_nm": 2.355 * sx, "fwhm_y_nm": 2.355 * sy,
                              "roi_halb_px": roi_halb_px})
            else:
                zeilen.append({"ebene": i, "z_nm": z_nm[i], "gefunden": True, "fit_ok": False,
                              "cx_px": cx_px, "cy_px": cy_px, "cx_nm": cx_nm, "cy_nm": cy_nm,
                              "sigma_x_nm": np.nan, "sigma_y_nm": np.nan,
                              "fwhm_x_nm": np.nan, "fwhm_y_nm": np.nan})

        n_fit_ok = sum(1 for z in zeilen if z["fit_ok"])
        cx_interp, cy_interp, ebene_gueltig = _interpolierte_zentren(zeilen, n_ebenen)
        if cx_interp is None:
            self._status_label.config(
                text=f"{n_fit_ok}/{n_ebenen} Ebenen erfolgreich gefittet — "
                     "weniger als 2 gültige Fits, ROI-Stack nicht möglich.")
            self._zeilen = zeilen
            self._z_nm = z_nm
            self._roi_stack = None
            self._axial = None
            return

        roi_halb_px_global = r_fit_max + fit_pad_px + 5
        roi_size = 2 * roi_halb_px_global
        roi_stack = np.full((n_ebenen, roi_size, roi_size), np.nan)
        for i, bild in enumerate(bilder):
            n_py, n_px = bild.shape
            cx_i, cy_i = int(round(cx_interp[i])), int(round(cy_interp[i]))
            xlo, xhi = cx_i - roi_halb_px_global, cx_i + roi_halb_px_global
            ylo, yhi = cy_i - roi_halb_px_global, cy_i + roi_halb_px_global
            if xlo < 0 or xhi > n_px or ylo < 0 or yhi > n_py:
                continue
            roi_stack[i] = bild[ylo:yhi, xlo:xhi]

        mitte = roi_halb_px_global
        xz_slice = roi_stack[:, mitte, :]
        yz_slice = roi_stack[:, :, mitte]
        xz_interp, z_fein = _interp_xz(xz_slice, z_nm, Z_INTERP_FAKTOR)
        yz_interp, _ = _interp_xz(yz_slice, z_nm, Z_INTERP_FAKTOR)

        I_zentral_roh = np.array([roi_stack[i, mitte, mitte] for i in range(n_ebenen)], dtype=float)
        gueltig_z = ~np.isnan(I_zentral_roh)
        z_zentral = z_nm[gueltig_z]
        I_zentral = I_zentral_roh[gueltig_z]
        fit_axial = _fit_axial(z_zentral, I_zentral)

        self._zeilen = zeilen
        self._z_nm = z_nm
        self._roi_stack = roi_stack
        self._axial = {
            "roi_halb_px": roi_halb_px_global, "px_nm": px_nm,
            "xz_interp": xz_interp, "yz_interp": yz_interp, "z_fein": z_fein,
            "z_zentral": z_zentral, "I_zentral": I_zentral, "fit": fit_axial,
            "n_ebenen": n_ebenen, "n_fit_ok": n_fit_ok,
            "n_interpoliert": int((~ebene_gueltig).sum()),
        }
        self._redraw()

    # ── Darstellung ───────────────────────────────────────────────────────────

    def _redraw(self):
        if self._axial is None:
            return
        a = self._axial

        if self._plot["fig"] is None:
            fig = Figure(figsize=(12, 3.6), dpi=90, facecolor="#1a1a1a")
            canvas = FigureCanvasTkAgg(fig, master=self._plot_frame)
            canvas.get_tk_widget().pack(fill="both", expand=True)
            self._plot["fig"] = fig
            self._plot["canvas"] = canvas
        fig = self._plot["fig"]
        fig.clear()
        axes = fig.subplots(1, 3)
        for ax in axes:
            ax.set_facecolor("#1a1a1a")
            ax.tick_params(colors="white", labelsize=7)

        px_nm = a["px_nm"]
        roi_size = a["xz_interp"].shape[1]
        roi_halb_px = a["roi_halb_px"]
        x_achse_nm = (np.arange(roi_size) - roi_halb_px + 0.5) * px_nm

        xz_norm = _normiere(a["xz_interp"])
        yz_norm = _normiere(a["yz_interp"])
        extent_xz = [x_achse_nm[0], x_achse_nm[-1], a["z_fein"][0], a["z_fein"][-1]]

        axes[0].imshow(xz_norm, origin="lower", aspect="auto", cmap="inferno",
                       extent=extent_xz, vmin=0, vmax=1)
        axes[0].set_title("x-z-Schnitt", color="white", fontsize=9)
        axes[0].set_xlabel("x (nm)", color="white", fontsize=8)
        axes[0].set_ylabel("z (nm)", color="white", fontsize=8)

        axes[1].imshow(yz_norm, origin="lower", aspect="auto", cmap="inferno",
                       extent=extent_xz, vmin=0, vmax=1)
        axes[1].set_title("y-z-Schnitt", color="white", fontsize=9)
        axes[1].set_xlabel("y (nm)", color="white", fontsize=8)
        axes[1].set_ylabel("z (nm)", color="white", fontsize=8)

        ax = axes[2]
        fit = a["fit"]
        z_zentral, I_zentral = a["z_zentral"], a["I_zentral"]
        lo, hi = I_zentral.min(), I_zentral.max()
        I_norm = (I_zentral - lo) / (hi - lo) if hi > lo else I_zentral
        ax.plot(z_zentral, I_norm, "o", color="#E05C2B", ms=4, alpha=0.6, label="Messung")
        if fit["popt_g"] is not None:
            z_kurve = np.linspace(z_zentral.min(), z_zentral.max(), 300)
            I_fit = _gauss_z(z_kurve, *fit["popt_g"])
            I_fit_norm = (I_fit - lo) / (hi - lo) if hi > lo else I_fit
            ax.plot(z_kurve, I_fit_norm, color="#E05C2B", lw=2,
                    label=f"Gauß FWHM_z={fit['fwhm_z']:.0f} nm")
        if fit["popt_l"] is not None:
            z_kurve = np.linspace(z_zentral.min(), z_zentral.max(), 300)
            I_fit = _lorentz_z(z_kurve, *fit["popt_l"])
            I_fit_norm = (I_fit - lo) / (hi - lo) if hi > lo else I_fit
            ax.plot(z_kurve, I_fit_norm, color="#3B8BD4", lw=1.5, ls="--",
                    label=f"Lorentz z_R={fit['zR']:.0f} nm")
        ax.set_title("Axiales Profil (Zentralpixel)", color="white", fontsize=9)
        ax.set_xlabel("z (nm)", color="white", fontsize=8)
        ax.set_ylabel("norm. Intensität", color="white", fontsize=8)
        ax.legend(fontsize=6, labelcolor="white", facecolor="#1a1a1a")

        fig.suptitle(os.path.basename(self._folder), color="white", fontsize=9)
        fig.tight_layout(pad=1.0)
        self._plot["canvas"].draw()

        fwhm_txt = f"{fit['fwhm_z']:.0f} nm" if np.isfinite(fit["fwhm_z"]) else "–"
        zr_txt = f"{fit['zR']:.0f} nm" if np.isfinite(fit["zR"]) else "–"
        self._status_label.config(
            text=(
                f"{a['n_ebenen']} Ebenen  |  {a['n_fit_ok']} Fits OK  |  "
                f"{a['n_interpoliert']} Zentren interpoliert  |  "
                f"FWHM_z (Gauß): {fwhm_txt}  |  z_R (Lorentz): {zr_txt}"
            )
        )

    # ── Export ────────────────────────────────────────────────────────────────

    def _save_results_csv(self):
        if not self._zeilen:
            return
        out_path = unique_output_path(
            self._get_output_dir(), timestamped_filename("krueger_zstack_ebenen", "csv"))
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ebene", "z_nm", "gefunden", "fit_ok", "cx_nm", "cy_nm",
                             "sigma_x_nm", "sigma_y_nm", "fwhm_x_nm", "fwhm_y_nm"])
            for z in self._zeilen:
                writer.writerow([z["ebene"], z["z_nm"], z["gefunden"], z["fit_ok"],
                                 z["cx_nm"], z["cy_nm"], z["sigma_x_nm"], z["sigma_y_nm"],
                                 z["fwhm_x_nm"], z["fwhm_y_nm"]])
        self._status_label.config(text=f"Exportiert: {os.path.basename(out_path)}")

    def _save_plot_png(self):
        if self._plot["fig"] is None:
            return
        out_path = unique_output_path(
            self._get_output_dir(), timestamped_filename("krueger_zstack", "png"))
        self._plot["fig"].savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
