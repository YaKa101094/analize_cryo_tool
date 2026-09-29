"""
analysis_schichtdicke.py – Schichtdickenanalyse via Spot-Zählung
================================================================
Enthält die Klasse SchichtdickeDialog.

Schichtdickenberechnung
  Formel (aus 06_full_screening_browser.ipynb):
    Volumendichte: ρ_vol [Moleküle/µm³] = c [nM] × 10⁻⁹ × Nₐ / 10¹⁵
    Schichtdicke:  d [nm] = N_Spots / (ρ_vol × A [µm²]) × 1000

Autor: Yannik Kasprzak, Institut für Physik, Universität zu Lübeck
"""

import os
import tkinter as tk
from tkinter import ttk

import numpy as np

from image_processing import (count_spots, get_channel_array,
                               timestamped_filename, unique_output_path)

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    _matplotlib_ok = True
except Exception as _e:
    print(f"[matplotlib import failed] {_e}")
    _matplotlib_ok = False


class SchichtdickeDialog:
    """Schichtdicken-Analysefenster (Spot-Zählmethode).

    Parameters
    ----------
    parent         : tk.Tk | tk.Toplevel | tk.Frame  Elternwidget — entweder
                     ein eigenständiges Fenster oder ein Tab-Frame im
                     Haupt-Notebook; der gesamte Dialoginhalt wird direkt
                     hineingebaut.
    datasets       : list[dict]           Geladene Bilddatensätze
    current_index  : int                  Aktuell angezeigter Datensatz-Index
    get_threshold     : callable          Liefert den initialen threshold_value (float)
                                           des Hauptfensters als Startwert für den
                                           lokalen Threshold-Regler
    get_output_dir    : callable          Gibt den Ausgabepfad zurück (str)
    get_bg_correction : callable          Liefert bool zurück: ob die perzentilbasierte
                                          Hintergrundkorrektur des Hauptfensters aktiv ist
    on_close          : callable | None   Wird vom "Schließen"-Button aufgerufen,
                                          statt parent.destroy(). Bei Nutzung als
                                          Notebook-Tab entfernt der Aufrufer damit
                                          den Tab (siehe ImageViewer._new_tab).
    """

    def __init__(self, parent, datasets, current_index,
                 get_threshold, get_output_dir, get_bg_correction, on_close=None):
        self._datasets      = datasets
        self._current_index = current_index
        self._get_output_dir = get_output_dir
        self._get_bg_correction = get_bg_correction

        self._calc_data  = {"rows": [], "meta": {}}
        self._plot       = {"fig": None, "ax": None, "canvas": None}
        self._plot_data  = None

        # ── Fenster/Tab ───────────────────────────────────────────────────────
        self.win = parent
        close_cmd = on_close if on_close is not None else self.win.destroy

        self._threshold_var = tk.DoubleVar(value=float(get_threshold()))

        tk.Label(self.win, text="Schichtdicke-Analyse",
                 font=("Arial", 12, "bold")).pack(pady=(12, 4))

        thr_row = tk.Frame(self.win)
        thr_row.pack(pady=(0, 4))
        tk.Label(thr_row, text="Threshold:").pack(side="left", padx=(0, 4))
        tk.Scale(thr_row, from_=0.0, to=1.0, resolution=0.05,
                 orient=tk.HORIZONTAL, length=230,
                 variable=self._threshold_var,
                 command=self._on_threshold_change).pack(side="left")

        self._info_label = tk.Label(self.win, text="", justify="left")
        self._info_label.pack(pady=(0, 2))
        ttk.Separator(self.win, orient="horizontal").pack(fill="x", padx=12, pady=(6, 2))

        self._build_calc_section()

        # Plot-Bereich
        self._plot_frame = tk.Frame(self.win, bg="#1a1a1a")
        self._plot_frame.pack(padx=12, pady=(4, 0), fill="both", expand=True)

        btn_row = tk.Frame(self.win)
        btn_row.pack(pady=4)
        tk.Button(btn_row, text="Berechnen", command=self._calc_schichtdicke,
                  width=14).pack(side="left", padx=6)
        tk.Button(btn_row, text="Graph speichern", command=self._save_plot_png,
                  width=16).pack(side="left", padx=6)
        self._show_errors_var = tk.BooleanVar(value=True)
        tk.Checkbutton(btn_row, text="Fehlerbalken",
                       variable=self._show_errors_var,
                       command=self._redraw_plot).pack(side="left", padx=6)
        tk.Button(btn_row, text="Schließen", command=close_cmd,
                  width=12).pack(side="left", padx=6)

        self._refresh()

    # ── UI-Aufbau ─────────────────────────────────────────────────────────────

    def _build_calc_section(self):
        """Schichtdickenberechnung mit Eingabefeldern und Ergebnis-Textbereich."""
        tk.Label(self.win, text="Schichtdickenberechnung",
                 font=("Arial", 11, "bold")).pack(pady=(4, 2))

        input_frame = tk.Frame(self.win)
        input_frame.pack(padx=12, pady=2)

        tk.Label(input_frame, text="Konzentration (nM):").grid(
            row=0, column=0, sticky="e", padx=6, pady=2)
        self._conc_var = tk.StringVar(value="1.0")
        tk.Entry(input_frame, textvariable=self._conc_var, width=10).grid(
            row=0, column=1, sticky="w", padx=6, pady=2)

        tk.Label(input_frame, text="Fläche (µm²):").grid(
            row=1, column=0, sticky="e", padx=6, pady=2)
        self._area_var = tk.StringVar(value="400.0")
        tk.Entry(input_frame, textvariable=self._area_var, width=10).grid(
            row=1, column=1, sticky="w", padx=6, pady=2)

        calc_header = tk.Frame(self.win)
        calc_header.pack(fill="x", padx=12, pady=(4, 0))
        tk.Label(calc_header, text="Ergebnisse",
                 font=("Arial", 9, "bold")).pack(side="left")
        tk.Button(calc_header, text="CSV speichern",
                  command=self._save_calc_csv).pack(side="right")

        calc_frame = tk.Frame(self.win)
        calc_frame.pack(padx=12, pady=2, fill="both", expand=True)
        calc_sb = tk.Scrollbar(calc_frame, orient="vertical")
        calc_sb.pack(side="right", fill="y")

        summary_frame = tk.Frame(calc_frame, width=110)
        summary_frame.pack(side="right", fill="y", padx=(8, 4))
        summary_frame.pack_propagate(False)
        tk.Label(summary_frame, text="Mittelwert",
                 font=("Arial", 9, "bold")).pack(pady=(4, 0))
        self._mean_label = tk.Label(summary_frame, text="–",
                                    font=("Arial", 13, "bold"))
        self._mean_label.pack()
        tk.Label(summary_frame, text="Std.-Abw.",
                 font=("Arial", 9, "bold")).pack(pady=(12, 0))
        self._std_label = tk.Label(summary_frame, text="–",
                                   font=("Arial", 13, "bold"))
        self._std_label.pack()

        self._calc_text = tk.Text(
            calc_frame, width=32, height=8, font=("Courier", 9),
            yscrollcommand=calc_sb.set, state="disabled",
        )
        calc_sb.config(command=self._calc_text.yview)
        self._calc_text.pack(side="left", fill="both", expand=True)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_threshold_change(self, _value):
        """Aktualisiert Info-Anzeige und Schichtdickenberechnung live beim Verschieben des Reglers."""
        self._refresh()
        self._calc_schichtdicke()

    def _set_calc_text(self, content: str):
        self._calc_text.config(state="normal")
        self._calc_text.delete("1.0", "end")
        self._calc_text.insert("1.0", content)
        self._calc_text.config(state="disabled")

    def _refresh(self):
        """Aktualisiert die Info-Anzeige für den aktuellen Datensatz und Threshold."""
        if not self._datasets:
            self._info_label.config(text="Keine Bilder geladen.")
            return
        dataset = self._datasets[self._current_index]
        img_sum = get_channel_array(dataset, "sum",
                                    bg_correction=self._get_bg_correction())
        max_intensity = float(np.nanmax(img_sum)) if img_sum.size else 0.0
        thr_cur = self._threshold_var.get()
        current_spots = count_spots(img_sum, float(thr_cur))
        self._info_label.config(
            text=(
                f"Datei: {dataset['name']}\n"
                f"Aktueller Threshold: {thr_cur:.2f} | Spots: {current_spots}\n"
                f"Max. Intensität (sum): {max_intensity:.2f}"
            )
        )

    def _calc_schichtdicke(self):
        """Berechnet Schichtdicke für alle Bilder und aktualisiert Plot."""
        if not self._datasets:
            self._set_calc_text("Keine Bilder geladen.")
            self._mean_label.config(text="–")
            self._std_label.config(text="–")
            return
        N_A = 6.022e23  # Avogadro-Konstante [mol⁻¹]
        try:
            c_nM  = float(self._conc_var.get().replace(",", "."))
            A_um2 = float(self._area_var.get().replace(",", "."))
        except ValueError:
            self._set_calc_text("Ungültige Eingabe.")
            self._mean_label.config(text="–")
            self._std_label.config(text="–")
            return
        if c_nM <= 0 or A_um2 <= 0:
            self._set_calc_text("Konzentration und Fläche müssen > 0 sein.")
            self._mean_label.config(text="–")
            self._std_label.config(text="–")
            return

        # Volumendichte der Lösung [Moleküle/µm³]:
        #   c [nM] × 1e-9 → [mol/L]  ×  Nₐ → [Moleküle/L]  ÷ 1e15 → [Moleküle/µm³]
        rho = c_nM * 1e-9 * N_A / 1e15
        thr = self._threshold_var.get()

        lines = [
            f"Threshold: {thr:.2f}  |  c={c_nM} nM  |  A={A_um2} µm²",
            f"{'Nr.':>4}  {'Spots':>6}  {'Schichtdicke (nm)':>18}",
            "-" * 34,
        ]
        thicknesses = []
        spot_counts = []
        for i, ds in enumerate(self._datasets):
            img_sum = get_channel_array(ds, "sum",
                                        bg_correction=self._get_bg_correction())
            n_spots = count_spots(img_sum, thr)
            d_nm = n_spots / (rho * A_um2) * 1000.0
            thicknesses.append(d_nm)
            spot_counts.append(n_spots)
            lines.append(f"{i + 1:>4}  {n_spots:>6}  {d_nm:>18.2f}")

        values = np.array(thicknesses)
        mean_d = float(np.mean(values))
        std_d  = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        lines += [
            "-" * 34,
            f"Mittelwert: {mean_d:.2f} nm",
            f"Std.-Abw.:  {std_d:.2f} nm",
            f"n = {len(values)} Bilder",
        ]
        self._mean_label.config(text=f"{mean_d:.2f} nm")
        self._std_label.config(text=f"± {std_d:.2f} nm")

        self._calc_data["rows"] = [
            (i + 1, self._datasets[i]["name"], n, round(d, 2))
            for i, (n, d) in enumerate(zip(spot_counts, thicknesses))
        ]
        self._calc_data["meta"] = {
            "thr": thr, "c_nM": c_nM, "A_um2": A_um2,
            "mean_d": mean_d, "std_d": std_d,
        }
        self._set_calc_text("\n".join(lines))

        if not _matplotlib_ok:
            return

        # Poisson-Zählstatistik der Spots (σ_N = √N) fortgepflanzt auf d [nm]
        d_errors = [np.sqrt(n) / (rho * A_um2) * 1000.0 for n in spot_counts]
        self._plot_data = {
            "x": list(range(1, len(thicknesses) + 1)),
            "thicknesses": thicknesses,
            "d_errors": d_errors,
            "mean_d": mean_d,
            "std_d": std_d,
        }
        self._redraw_plot()

    def _redraw_plot(self):
        """Zeichnet den Schichtdicken-Plot anhand der zuletzt berechneten Daten neu."""
        if not _matplotlib_ok or self._plot_data is None:
            return
        data = self._plot_data
        x, thicknesses = data["x"], data["thicknesses"]
        mean_d, std_d = data["mean_d"], data["std_d"]

        # Verteilungsplot: einmal erstellen, danach nur neu zeichnen.
        if self._plot["fig"] is None:
            self._plot["fig"] = Figure(figsize=(5, 2.6), dpi=88,
                                       facecolor="#2a2a2a")
            self._plot["ax"]  = self._plot["fig"].add_subplot(111)
            self._plot["canvas"] = FigureCanvasTkAgg(
                self._plot["fig"], master=self._plot_frame)
            self._plot["canvas"].get_tk_widget().pack(fill="both", expand=True)

        ax = self._plot["ax"]
        ax.clear()
        ax.set_facecolor("#1a1a1a")

        yerr = data["d_errors"] if self._show_errors_var.get() else None
        ax.errorbar(x, thicknesses, yerr=yerr, fmt="o", color="#4249C9",
                    ecolor="#4249C9", elinewidth=1.2, capsize=4,
                    markersize=6, zorder=3)
        ax.axhline(mean_d, color="#ffff88", linewidth=1.4, linestyle="--",
                   label=f"Ø {mean_d:.1f} nm")
        if len(thicknesses) > 1:
            # ±1σ-Band um den Mittelwert
            ax.axhspan(mean_d - std_d, mean_d + std_d, alpha=0.12, color="#ffff88")

        ax.set_xlabel("Bild-Nr.", color="white", fontsize=11)
        ax.set_ylabel("d (nm)", color="white", fontsize=11)
        ax.set_xticks(x)
        ax.tick_params(colors="white", labelsize=10)
        for spine in ax.spines.values():
            spine.set_edgecolor("#555555")
        ax.legend(fontsize=10, labelcolor="white",
                  facecolor="#2a2a2a", edgecolor="#555555")
        self._plot["fig"].tight_layout(pad=1.0)
        self._plot["canvas"].draw()

    # ── Export ────────────────────────────────────────────────────────────────

    def _save_plot_png(self):
        """Speichert den Schichtdicken-Plot als PNG in den Ausgabeordner."""
        if self._plot["fig"] is None:
            return
        output_dir = self._get_output_dir()
        out_path = unique_output_path(
            output_dir, timestamped_filename("schichtdicke_plot", "png"))
        self._plot["fig"].savefig(out_path, dpi=150, bbox_inches="tight",
                                  facecolor=self._plot["fig"].get_facecolor())

    def _save_calc_csv(self):
        """Exportiert die Berechnungsergebnisse als CSV."""
        if not self._calc_data["rows"]:
            return
        output_dir = self._get_output_dir()
        out_path = unique_output_path(output_dir, "schichtdicke_ergebnisse.csv")
        m = self._calc_data["meta"]
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            f.write(
                f"# Threshold={m.get('thr', '')}, "
                f"c={m.get('c_nM', '')} nM, "
                f"A={m.get('A_um2', '')} µm²\n"
            )
            f.write("Nr.,Dateiname,Spots,Schichtdicke_nm\n")
            for row in self._calc_data["rows"]:
                f.write(",".join(str(v) for v in row) + "\n")
            f.write(f"Mittelwert,,, {m.get('mean_d', ''):.2f}\n")
            f.write(f"Std.-Abw.,,, {m.get('std_d', ''):.2f}\n")
