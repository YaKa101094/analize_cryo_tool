"""
analysis_ics.py – Strenges ICS-Schichtdicken-Analysefenster
=============================================================
Enthält die Klasse ICSDialog. Bindet die eigenständige, wissenschaftlich
strenge Pipeline aus ics_thickness.py (Detrending, formaler Kovarianz-Fehler
UND empirische Kachelstreuung, Plausibilitäts-Warnungen) an die bereits im
ImageViewer geladenen Datensätze an – ohne den Umweg über Dateien auf der
Festplatte (dafür wird analyze_tiles() statt analyze() aus ics_thickness.py
verwendet, siehe dort).

Autor: Yannik Kasprzak, Institut für Physik, Universität zu Lübeck
"""

import warnings
from pathlib import Path

import tkinter as tk

from image_processing import get_channel_array, timestamped_filename, unique_output_path

try:
    import ics_thickness as _ics
    _ics_import_error = None
except Exception as _e:
    _ics = None
    _ics_import_error = str(_e)

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.pyplot as _plt
    _matplotlib_ok = True
except Exception as _e:
    print(f"[matplotlib import failed] {_e}")
    _matplotlib_ok = False


class ICSDialog:
    """Strenges ICS-Schichtdicken-Analysefenster (siehe ics_thickness.py).

    Parameters
    ----------
    parent            : tk.Tk | tk.Toplevel | tk.Frame  Elternwidget — entweder
                        ein eigenständiges Fenster oder ein Tab-Frame im
                        Haupt-Notebook; der gesamte Dialoginhalt wird direkt
                        hineingebaut.
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
        self._results = None
        self._plot = {"fig": None, "canvas": None}

        if _ics is None:
            self.win = parent
            close_cmd = on_close if on_close is not None else self.win.destroy
            tk.Label(self.win, text="ics_thickness.py konnte nicht importiert werden.",
                     font=("Arial", 11, "bold")).pack(pady=(22, 4))
            tk.Label(self.win, text=str(_ics_import_error), font=("Courier", 10),
                     wraplength=700, justify="left").pack(pady=4)
            tk.Label(self.win, text="(Vermutlich fehlt scipy: pip install scipy)",
                     font=("Arial", 9)).pack()
            tk.Button(self.win, text="Schließen", command=close_cmd,
                      width=12).pack(pady=14)
            return

        # ── Fenster/Tab ───────────────────────────────────────────────────────
        self.win = parent
        close_cmd = on_close if on_close is not None else self.win.destroy

        tk.Label(self.win, text="Schichtdicke – Strenge ICS-Analyse (ics_thickness.py)",
                 font=("Arial", 12, "bold")).pack(pady=(10, 4))

        self._build_param_frame()
        self._build_result_frame()

        btn_frame = tk.Frame(self.win)
        btn_frame.pack(side="bottom", pady=6)
        tk.Button(btn_frame, text="Berechnen", command=self._run_analysis,
                  width=14).pack(side="left", padx=6)
        tk.Button(btn_frame, text="Report speichern", command=self._save_report_txt,
                  width=16).pack(side="left", padx=6)
        tk.Button(btn_frame, text="Plot speichern", command=self._save_plot_png,
                  width=14).pack(side="left", padx=6)
        tk.Button(btn_frame, text="Schließen", command=close_cmd,
                  width=12).pack(side="left", padx=6)

        self._plot_frame = tk.Frame(self.win, bg="#1a1a2e")
        self._plot_frame.pack(fill="both", expand=True, padx=12, pady=(4, 0))

    # ── UI-Aufbau ─────────────────────────────────────────────────────────────

    def _build_param_frame(self):
        """Parameter-Eingabefelder mit Erläuterung."""
        frame = tk.LabelFrame(self.win, text="Parameter", padx=8, pady=4)
        frame.pack(fill="x", padx=12, pady=4)

        self._c0_var         = tk.StringVar(value="1.0")
        self._px_var         = tk.StringVar(value="50.0")
        self._w0_var         = tk.StringVar(value="250.0")
        self._fitradius_var  = tk.StringVar(value="15")
        self._detrend_var    = tk.BooleanVar(value=True)
        self._detrendsig_var = tk.StringVar(value="40.0")
        self._exclcenter_var = tk.BooleanVar(value=True)
        self._axial_var      = tk.StringVar(value="")

        input_frame = tk.Frame(frame)
        input_frame.pack(side="left", anchor="n")

        row_defs = [
            ("Konzentration c₀ (nM):",        self._c0_var),
            ("Pixelgröße (nm/px):",            self._px_var),
            ("Erwartetes w₀ (nm):",            self._w0_var),
            ("Fit-Radius (px):",               self._fitradius_var),
            ("Detrend-Sigma (px):",            self._detrendsig_var),
            ("Axiale PSF-Ausdehnung (nm, optional):", self._axial_var),
        ]
        for idx, (lbl_text, var) in enumerate(row_defs):
            tk.Label(input_frame, text=lbl_text, anchor="e").grid(
                row=idx, column=0, sticky="e", padx=6, pady=2)
            tk.Entry(input_frame, textvariable=var, width=10).grid(
                row=idx, column=1, sticky="w", padx=6, pady=2)

        chk_row = len(row_defs)
        tk.Checkbutton(input_frame, text="Detrending (Beleuchtungsinhomogenität)",
                       variable=self._detrend_var).grid(
            row=chk_row, column=0, columnspan=2, sticky="w", padx=4, pady=(6, 0))
        tk.Checkbutton(input_frame, text="Zentralpixel vom Fit ausschließen (Schrotrauschen)",
                       variable=self._exclcenter_var).grid(
            row=chk_row + 1, column=0, columnspan=2, sticky="w", padx=4)

        tk.Frame(frame, width=1, bg="#444444").pack(
            side="left", fill="y", padx=(12, 10), pady=2)

        desc_text = (
            "c₀          Molekülkonzentration der Lösung in nM.\n\n"
            "Pixelgröße  Physikalische Pixelkantenlänge in nm.\n\n"
            "Erw. w₀     Erwarteter PSF-e⁻²-Radius (aus NA und\n"
            "               Wellenlänge) — nur Fit-Startwert und\n"
            "               Plausibilitätsprüfung, kein harter Zwang.\n\n"
            "Fit-Radius  Halbe Fenstergröße (px) für den 2D-Gauß-Fit\n"
            "               an die gemittelte ACF.\n\n"
            "Detrend-    Breite (px) des Gauß-Hintergrundfilters für\n"
            "Sigma          die Beleuchtungskorrektur. Muss deutlich\n"
            "               größer als w₀/Pixelgröße sein.\n\n"
            "Axiale PSF  Konfokale Tiefenschärfe in nm (optional) —\n"
            "               nur für die Plausibilitätsprüfung 'h vs.\n"
            "               axiale Erfassung'. Leer = Prüfung übersprungen."
        )
        tk.Label(frame, text=desc_text, justify="left", anchor="nw",
                 font=("Arial", 8), fg="#aaaaaa").pack(
            side="left", anchor="n", pady=4)

    def _build_result_frame(self):
        """Ergebnis-Textbereich mit Mittelwert/Std.-Abw.-Anzeige."""
        result_outer = tk.Frame(self.win)
        result_outer.pack(fill="x", padx=12, pady=(4, 0))

        tk.Label(result_outer, text="Ergebnisbericht",
                 font=("Arial", 9, "bold")).pack(anchor="w")

        body = tk.Frame(result_outer)
        body.pack(fill="x")

        result_sb = tk.Scrollbar(body, orient="vertical")
        result_sb.pack(side="right", fill="y")

        summary_frame = tk.Frame(body, width=150)
        summary_frame.pack(side="right", fill="y", padx=(8, 4))
        summary_frame.pack_propagate(False)
        tk.Label(summary_frame, text="h (Ensemble-Fit)",
                 font=("Arial", 9, "bold")).pack(pady=(4, 0))
        self._mean_label = tk.Label(summary_frame, text="–",
                                    font=("Arial", 13, "bold"))
        self._mean_label.pack()
        tk.Label(summary_frame, text="Kachelstreuung",
                 font=("Arial", 9, "bold")).pack(pady=(12, 0))
        self._std_label = tk.Label(summary_frame, text="–",
                                   font=("Arial", 11, "bold"))
        self._std_label.pack()

        self._result_text = tk.Text(
            body, width=90, height=14, font=("Courier", 9),
            yscrollcommand=result_sb.set, state="disabled",
        )
        result_sb.config(command=self._result_text.yview)
        self._result_text.pack(side="left", fill="both", expand=True)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _set_result_text(self, content: str):
        self._result_text.config(state="normal")
        self._result_text.delete("1.0", "end")
        self._result_text.insert("1.0", content)
        self._result_text.config(state="disabled")

    def _run_analysis(self):
        """Baut die MeasurementConfig aus den GUI-Feldern, lädt die aktuell
        geladenen Datensätze als Kacheln und führt analyze_tiles() aus."""
        try:
            c0_nM       = float(self._c0_var.get().replace(",", "."))
            px_nm       = float(self._px_var.get().replace(",", "."))
            w0_nm       = float(self._w0_var.get().replace(",", "."))
            fit_radius  = int(self._fitradius_var.get())
            detrend     = self._detrend_var.get()
            exclude_ctr = self._exclcenter_var.get()
            detrend_sigma_px = (
                float(self._detrendsig_var.get().replace(",", "."))
                if detrend else None
            )
            axial_str = self._axial_var.get().strip()
            axial_psf_extent_m = float(axial_str.replace(",", ".")) * 1e-9 if axial_str else None
        except ValueError:
            self._set_result_text("Ungültige Eingabe in den Parametern.")
            self._mean_label.config(text="–")
            self._std_label.config(text="–")
            return

        if c0_nM <= 0 or px_nm <= 0 or w0_nm <= 0 or fit_radius < 1:
            self._set_result_text(
                "Konzentration, Pixelgröße und erwartetes w₀ müssen > 0 sein; "
                "Fit-Radius muss >= 1 sein.")
            self._mean_label.config(text="–")
            self._std_label.config(text="–")
            return

        if not self._datasets:
            self._set_result_text("Keine Datensätze geladen.")
            return

        tiles = [
            get_channel_array(ds, "sum", bg_correction=self._get_bg_correction()).astype(float)
            for ds in self._datasets
        ]

        try:
            config = _ics.MeasurementConfig(
                px_size_m=px_nm * 1e-9,
                conc_mol_per_L=c0_nM * 1e-9,
                expected_w0_m=w0_nm * 1e-9,
                fit_radius_px=fit_radius,
                output_dir=Path(self._get_output_dir()),
                exclude_center=exclude_ctr,
                detrend=detrend,
                detrend_sigma_px=detrend_sigma_px,
                axial_psf_extent_m=axial_psf_extent_m,
            )
            with warnings.catch_warnings():
                # Plausibilitäts-/Fit-Warnungen landen bereits sichtbar im
                # Report (results.warnings_raised) - keine doppelte
                # Konsolen-Ausgabe nötig.
                warnings.simplefilter("ignore", RuntimeWarning)
                results = _ics.analyze_tiles(tiles, config)
        except Exception as exc:
            self._results = None
            self._set_result_text(
                f"Fehler bei der ICS-Analyse:\n\n{type(exc).__name__}: {exc}"
            )
            self._mean_label.config(text="–")
            self._std_label.config(text="–")
            return

        self._results = results
        self._set_result_text(_ics.format_report(results))
        self._mean_label.config(text=f"{results.h_m * 1e9:.3f} nm")
        self._std_label.config(text=f"± {results.per_tile_h_std_m * 1e9:.3f} nm")

        self._draw_plot(results)

    def _draw_plot(self, results):
        """Erstellt/ersetzt den 4-Panel-Report-Plot aus ics_thickness.plot_results()."""
        if not _matplotlib_ok:
            return

        if self._plot["canvas"] is not None:
            self._plot["canvas"].get_tk_widget().destroy()
        if self._plot["fig"] is not None:
            _plt.close(self._plot["fig"])

        fig = _ics.plot_results(results)
        canvas = FigureCanvasTkAgg(fig, master=self._plot_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()

        self._plot["fig"] = fig
        self._plot["canvas"] = canvas

    # ── Export ────────────────────────────────────────────────────────────────

    def _save_report_txt(self):
        """Speichert den Textbericht (format_report) in den Ausgabeordner."""
        if self._results is None:
            return
        out_path = unique_output_path(self._get_output_dir(), "ics_report.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(_ics.format_report(self._results))

    def _save_plot_png(self):
        """Speichert den 4-Panel-Report als PNG in den Ausgabeordner."""
        if self._plot["fig"] is None:
            return
        out_path = unique_output_path(
            self._get_output_dir(), timestamped_filename("ics_report", "png"))
        self._plot["fig"].savefig(out_path, dpi=150, bbox_inches="tight")
