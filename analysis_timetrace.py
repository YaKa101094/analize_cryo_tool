"""
analysis_timetrace.py – FCS-Korrelationsanalyse-Dialog
======================================================
Enthält die Klasse TimeraceDialog für die FCS-Korrelationsanalyse (entspricht
analysis.ipynb). Baut ihre Oberfläche direkt in ein übergebenes Elternwidget
(eigenständiges Fenster ODER Tab-Frame im Haupt-Notebook, siehe ImageViewer).

Workflow
--------
1. HDF5-Datei laden  (photon_tools.load)
2. Photon-Timestamps nach Detektor trennen  (by_detector())
3. Logarithmisch gestufte Lag-Bins  (pycorrelate.make_loglags)
4. Normierte Kreuzkorrelation G(τ)  (pycorrelate.pcorrelate)
5. Modellfit:
     3D-FCS:       G(τ) = 1 + A₀/[1 + τ/τ_D] / √[1 + S²·τ/τ_D]
     2D-Diffusion: G(τ) = 1 + A₀/[1 + τ/τ_D]
     Eigene Funktion: beliebiger lmfit.ExpressionModel (Variable x = τ)
6. 2-Panel-Plot: G(τ) + Fit (oben) | Residuen (unten)

Autor: Yannik Kasprzak, Institut für Physik, Universität zu Lübeck
"""

import os
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np

from image_processing import timestamped_filename, unique_output_path

# ── Optionale Abhängigkeiten ──────────────────────────────────────────────────

try:
    import photon_tools as _pt
except Exception:
    _pt = None

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import matplotlib.gridspec as _mpl_gridspec
    _matplotlib_ok = True
except Exception as _e:
    print(f"[matplotlib import failed] {_e}")
    _matplotlib_ok = False

try:
    import pycorrelate as _pyc
    _pycorrelate_ok = True
    _pyc_err_msg = ""
except Exception as _e:
    print(f"[pycorrelate import failed] {_e}")
    _pyc = None
    _pycorrelate_ok = False
    _pyc_err_msg = str(_e)

try:
    import lmfit as _lmfit
    _lmfit_ok = True
    _lmfit_err_msg = ""
except Exception as _e:
    print(f"[lmfit import failed] {_e}")
    _lmfit = None
    _lmfit_ok = False
    _lmfit_err_msg = str(_e)


class TimeraceDialog:
    """FCS-Korrelationsanalyse-Fenster.

    Parameters
    ----------
    parent         : tk.Tk | tk.Toplevel | tk.Frame  Elternwidget — entweder
                     ein eigenständiges Fenster oder ein Tab-Frame im
                     Haupt-Notebook; der gesamte Dialoginhalt wird direkt
                     hineingebaut.
    get_output_dir : callable             Gibt den Ausgabepfad zurück (str)
    on_close       : callable | None      Wird vom "Schließen"-Button aufgerufen,
                                          statt parent.destroy() (siehe
                                          ImageViewer._new_tab).
    """

    def __init__(self, parent, get_output_dir, on_close=None):
        self._get_output_dir = get_output_dir
        self._plot = {"fig": None, "canvas": None}
        self._params_text_widget = None     # Text-Widget für "Eigene Funktion"

        # ── Fenster/Tab ───────────────────────────────────────────────────────
        self.win = parent
        close_cmd = on_close if on_close is not None else self.win.destroy

        tk.Label(self.win, text="FCS Korrelationsanalyse",
                 font=("Arial", 12, "bold")).pack(pady=(10, 4))

        self._build_file_frame()
        self._build_param_frame()
        self._build_model_frame()
        self._build_result_frame()

        # Plot-Bereich (expandierend)
        self.plot_frame = tk.Frame(self.win, bg="#1a1a1a")
        self.plot_frame.pack(fill="both", expand=True, padx=12, pady=(4, 0))

        # Buttons am unteren Rand
        btn_frame = tk.Frame(self.win)
        btn_frame.pack(side="bottom", pady=6)
        tk.Button(btn_frame, text="Berechnen", command=self.run_analysis,
                  width=14).pack(side="left", padx=6)
        tk.Button(btn_frame, text="Plot speichern", command=self._save_plot_png,
                  width=14).pack(side="left", padx=6)
        tk.Button(btn_frame, text="Schließen", command=close_cmd,
                  width=12).pack(side="left", padx=6)

    # ── UI-Aufbau ─────────────────────────────────────────────────────────────

    def _build_file_frame(self):
        """Dateiauswahl-Bereich."""
        frame = tk.LabelFrame(self.win, text="HDF5-Datei", padx=8, pady=4)
        frame.pack(fill="x", padx=12, pady=4)

        self.file_var = tk.StringVar()
        tk.Entry(frame, textvariable=self.file_var, width=70).pack(
            side="left", fill="x", expand=True, padx=(0, 6))
        tk.Button(frame, text="Durchsuchen…", command=self._browse_file).pack(side="right")

    def _build_param_frame(self):
        """Mess-Parameter-Bereich."""
        frame = tk.LabelFrame(self.win, text="Parameter", padx=8, pady=4)
        frame.pack(fill="x", padx=12, pady=4)

        self.unit_var    = tk.StringVar(value="5e-9")
        self.bpd_var     = tk.StringVar(value="20")
        self.lag_min_var = tk.StringVar(value="-5")
        self.lag_max_var = tk.StringVar(value="1")
        self.ch_var      = tk.StringVar(value="ch0 × ch0")

        param_defs = [
            ("Timing-Auflösung (s):", self.unit_var),
            ("Bins pro Dekade:",      self.bpd_var),
            ("Lag von (Dekaden):",    self.lag_min_var),
            ("Lag bis (Dekaden):",    self.lag_max_var),
        ]
        for idx, (lbl, var) in enumerate(param_defs):
            r, c = idx // 2, (idx % 2) * 2
            tk.Label(frame, text=lbl, anchor="e").grid(
                row=r, column=c, sticky="e", padx=6, pady=2)
            tk.Entry(frame, textvariable=var, width=10).grid(
                row=r, column=c + 1, sticky="w", padx=6, pady=2)

        tk.Label(frame, text="Korrelationsmodus:", anchor="e").grid(
            row=2, column=0, sticky="e", padx=6, pady=2)
        ttk.Combobox(
            frame, textvariable=self.ch_var, state="readonly", width=20,
            values=["ch0 × ch0", "ch1 × ch1", "ch0 × ch1 (Kreuzkorrelation)"],
        ).grid(row=2, column=1, sticky="w", padx=6, pady=2)

    def _build_model_frame(self):
        """Modell- und Fit-Parameter-Bereich."""
        frame = tk.LabelFrame(self.win, text="Modell & Fit-Parameter", padx=8, pady=4)
        frame.pack(fill="x", padx=12, pady=4)

        self.model_var = tk.StringVar(value="3D-FCS")
        model_row = tk.Frame(frame)
        model_row.pack(fill="x", pady=(0, 4))
        tk.Label(model_row, text="Modell:", anchor="e").pack(side="left", padx=(0, 6))
        ttk.Combobox(
            model_row, textvariable=self.model_var, state="readonly", width=24,
            values=["3D-FCS", "2D-Diffusion", "Eigene Funktion"],
        ).pack(side="left")

        # Fit-Parameter-Variablen
        self.A0_var      = tk.StringVar(value="1.0")
        self.A0_min_var  = tk.StringVar(value="0.001")
        self.A0_max_var  = tk.StringVar(value="15")
        self.td_var      = tk.StringVar(value="1e-3")
        self.td_min_var  = tk.StringVar(value="1e-7")
        self.td_max_var  = tk.StringVar(value="1e-3")
        self.wz_var      = tk.StringVar(value="0.2")
        self.wz_vary_var = tk.BooleanVar(value=False)
        self.expr_var    = tk.StringVar(value="1 + A0 / (1 + x/tau_diff)")

        self.fit_inner = tk.Frame(frame)
        self.fit_inner.pack(fill="x")

        self.model_var.trace_add("write", self._rebuild_fit_ui)
        self._rebuild_fit_ui()

    def _build_result_frame(self):
        """Ergebnis-Textbereich."""
        res_outer = tk.Frame(self.win)
        res_outer.pack(fill="x", padx=12, pady=(4, 0))
        res_sb = tk.Scrollbar(res_outer, orient="vertical")
        self.result_text = tk.Text(
            res_outer, width=80, height=5, font=("Courier", 9),
            yscrollcommand=res_sb.set, state="disabled",
        )
        res_sb.config(command=self.result_text.yview)
        res_sb.pack(side="right", fill="y")
        self.result_text.pack(side="left", fill="x", expand=True)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="HDF5-Datei öffnen",
            filetypes=[("HDF5 Dateien", "*.h5 *.hdf5"), ("Alle Dateien", "*.*")],
        )
        if path:
            self.file_var.set(path)

    def _rebuild_fit_ui(self, *_):
        """Baut den Fit-Parameter-Bereich abhängig vom gewählten Modell neu auf."""
        for w in self.fit_inner.winfo_children():
            w.destroy()
        m = self.model_var.get()
        if m in ("3D-FCS", "2D-Diffusion"):
            for ci, h in enumerate(["Parameter", "Startwert", "Min", "Max"]):
                tk.Label(self.fit_inner, text=h,
                         font=("Arial", 8, "bold")).grid(row=0, column=ci, padx=8, pady=2)
            for r, (lbl, v, vmin, vmax) in enumerate([
                ("A₀",         self.A0_var, self.A0_min_var, self.A0_max_var),
                ("τ_diff (s)", self.td_var, self.td_min_var, self.td_max_var),
            ], start=1):
                tk.Label(self.fit_inner, text=lbl).grid(row=r, column=0, sticky="w", padx=8)
                tk.Entry(self.fit_inner, textvariable=v,    width=9).grid(row=r, column=1, padx=4)
                tk.Entry(self.fit_inner, textvariable=vmin, width=9).grid(row=r, column=2, padx=4)
                tk.Entry(self.fit_inner, textvariable=vmax, width=9).grid(row=r, column=3, padx=4)
            if m == "3D-FCS":
                tk.Label(self.fit_inner, text="S = ω_z/ω_xy").grid(
                    row=3, column=0, sticky="w", padx=8)
                tk.Entry(self.fit_inner, textvariable=self.wz_var, width=9).grid(
                    row=3, column=1, padx=4)
                tk.Label(self.fit_inner, text="(fest)").grid(
                    row=3, column=2, columnspan=2, sticky="w")
                tk.Checkbutton(self.fit_inner, text="anpassen",
                               variable=self.wz_vary_var).grid(row=3, column=4, padx=4)
        else:
            # Eigene Funktion: Ausdrucks-Eingabe + Anfangsparameter
            tk.Label(self.fit_inner, text="Ausdruck (x = τ):",
                     anchor="w").grid(row=0, column=0, sticky="w", padx=6, pady=(2, 0))
            tk.Entry(self.fit_inner, textvariable=self.expr_var, width=55).grid(
                row=0, column=1, columnspan=3, sticky="we", padx=4, pady=(2, 0))
            tk.Label(self.fit_inner, text="Anfangsparameter\n(Name = Wert, je Zeile):",
                     anchor="w", justify="left").grid(
                row=1, column=0, sticky="nw", padx=6, pady=(4, 2))
            t = tk.Text(self.fit_inner, width=40, height=4, font=("Courier", 9))
            t.grid(row=1, column=1, columnspan=3, sticky="we", padx=4, pady=(4, 2))
            t.insert("1.0", "A0 = 1.0\ntau_diff = 1e-3")
            self._params_text_widget = t
            tk.Label(
                self.fit_inner,
                text="x ist die Zeitverzögerung τ.  Alle anderen Variablen = Fit-Parameter.",
                font=("Arial", 8), fg="gray",
            ).grid(row=2, column=0, columnspan=4, sticky="w", padx=6, pady=(0, 2))

    def _set_result(self, content: str):
        """Schreibt *content* in das (schreibgeschützte) Ergebnis-Textfeld."""
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", content)
        self.result_text.config(state="disabled")

    # ── Analyse ───────────────────────────────────────────────────────────────

    def run_analysis(self):
        """Führt FCS-Korrelation und Modellfit durch und aktualisiert den Plot."""
        import sys

        # Paket-Verfügbarkeit prüfen
        missing = [
            name for name, ok in [("pycorrelate", _pycorrelate_ok), ("lmfit", _lmfit_ok)]
            if not ok
        ]
        if missing:
            err_details = []
            if not _pycorrelate_ok and _pyc_err_msg:
                err_details.append(f"  pycorrelate: {_pyc_err_msg}")
            if not _lmfit_ok and _lmfit_err_msg:
                err_details.append(f"  lmfit: {_lmfit_err_msg}")
            self._set_result(
                "Fehlende Pakete: " + ", ".join(missing) + "\n"
                + "\n".join(err_details) + "\n\n"
                + f"Python: {sys.executable}\n"
                + "Pakete installieren mit:\n  pip install " + " ".join(missing)
            )
            return
        if _pt is None:
            self._set_result("Fehler: photon_tools ist nicht verfügbar.")
            return

        h5_path = self.file_var.get().strip()
        if not h5_path or not os.path.isfile(h5_path):
            self._set_result("Bitte zuerst eine gültige HDF5-Datei auswählen.")
            return

        # Parameter parsen
        try:
            unit  = float(self.unit_var.get().replace(",", "."))
            bpd   = int(self.bpd_var.get())
            l_min = float(self.lag_min_var.get().replace(",", "."))
            l_max = float(self.lag_max_var.get().replace(",", "."))
        except ValueError:
            self._set_result("Ungültige Parameter-Eingabe.")
            return

        # HDF5 laden
        try:
            ds = _pt.load(h5_path, timing_resolution=unit)
        except Exception as e:
            self._set_result(f"Fehler beim Laden der Datei:\n  {e}")
            return

        # Korrelation berechnen
        try:
            by_ch = ds.photons.by_detector()
            keys  = sorted(by_ch.keys())
            mode  = self.ch_var.get()
            if mode == "ch0 × ch0":
                ch_a = ch_b = by_ch[keys[0]]
            elif mode == "ch1 × ch1":
                k = keys[1] if len(keys) > 1 else keys[0]
                ch_a = ch_b = by_ch[k]
            else:
                ch_a = by_ch[keys[0]]
                ch_b = by_ch[keys[1]] if len(keys) > 1 else by_ch[keys[0]]

            # make_loglags erwartet ganzzahlige Exponenten.  astype(int64) ist
            # Pflicht, da pcorrelate numba-jit mit int-Typen rechnet.
            bins = (
                _pyc.make_loglags(int(l_min), int(l_max), bpd)[bpd // 2:] / unit
            ).astype(np.int64)
            Gn   = _pyc.pcorrelate(ch_a, ch_b, bins, normalize=True)
            # Zeitachse τ: Mittelpunkt jedes Bin-Intervalls in Sekunden
            tau  = 0.5 * (bins[1:] + bins[:-1]) * unit
        except Exception as e:
            self._set_result(f"Fehler bei der Korrelationsberechnung:\n  {e}")
            return

        # Modellfit
        fitres = None
        tau_diff_us = None
        m = self.model_var.get()
        try:
            if m == "3D-FCS":
                A0_init = float(self.A0_var.get().replace(",", "."))
                A0_min  = float(self.A0_min_var.get().replace(",", "."))
                A0_max  = float(self.A0_max_var.get().replace(",", "."))
                td_init = float(self.td_var.get().replace(",", "."))
                td_min  = float(self.td_min_var.get().replace(",", "."))
                td_max  = float(self.td_max_var.get().replace(",", "."))
                wz_val  = float(self.wz_var.get().replace(",", "."))

                def diffusion_3d_fcs(timelag, A0, tau_diff, waist_z_ratio=0.2):
                    return (
                        1 + A0 / (1 + timelag / tau_diff)
                        / np.sqrt(1 + waist_z_ratio ** 2 * timelag / tau_diff)
                    )
                model_fit = _lmfit.Model(diffusion_3d_fcs)
                params = model_fit.make_params(A0=A0_init, tau_diff=td_init,
                                               waist_z_ratio=wz_val)
                params["A0"].set(min=A0_min, max=A0_max)
                params["tau_diff"].set(min=td_min, max=td_max)
                params["waist_z_ratio"].set(value=wz_val, vary=self.wz_vary_var.get())
                fitres = model_fit.fit(Gn, timelag=tau, params=params,
                                       method="least_squares")
                tau_diff_us = fitres.values["tau_diff"] * 1e6
                self._set_result("\n".join([
                    f"Datei:   {os.path.basename(h5_path)}",
                    f"Modell:  3D-FCS  |  Modus: {mode}  |  Datenpunkte: {len(Gn)}",
                    "",
                    "Fit-Ergebnisse (3D-FCS):",
                    f"  G(0)-1  = A₀       = {fitres.values['A0']:.4f}",
                    f"  τ_D               = {tau_diff_us:.2f} µs",
                    f"  S = ω_z/ω_xy     = {fitres.values['waist_z_ratio']:.4f}",
                    f"  Reduziertes χ²   = {fitres.redchi:.4e}",
                ]))

            elif m == "2D-Diffusion":
                A0_init = float(self.A0_var.get().replace(",", "."))
                A0_min  = float(self.A0_min_var.get().replace(",", "."))
                A0_max  = float(self.A0_max_var.get().replace(",", "."))
                td_init = float(self.td_var.get().replace(",", "."))
                td_min  = float(self.td_min_var.get().replace(",", "."))
                td_max  = float(self.td_max_var.get().replace(",", "."))

                def diffusion_2d(timelag, A0, tau_diff):
                    return 1 + A0 / (1 + timelag / tau_diff)
                model_fit = _lmfit.Model(diffusion_2d)
                params = model_fit.make_params(A0=A0_init, tau_diff=td_init)
                params["A0"].set(min=A0_min, max=A0_max)
                params["tau_diff"].set(min=td_min, max=td_max)
                fitres = model_fit.fit(Gn, timelag=tau, params=params,
                                       method="least_squares")
                tau_diff_us = fitres.values["tau_diff"] * 1e6
                self._set_result("\n".join([
                    f"Datei:   {os.path.basename(h5_path)}",
                    f"Modell:  2D-Diffusion  |  Modus: {mode}  |  Datenpunkte: {len(Gn)}",
                    "",
                    "Fit-Ergebnisse (2D-Diffusion):",
                    f"  G(0)-1  = A₀       = {fitres.values['A0']:.4f}",
                    f"  τ_D               = {tau_diff_us:.2f} µs",
                    f"  Reduziertes χ²   = {fitres.redchi:.4e}",
                ]))

            else:  # Eigene Funktion
                expr = self.expr_var.get().strip()
                if not expr:
                    self._set_result("Bitte einen Ausdruck eingeben.")
                    return
                init_params = {}
                if self._params_text_widget is not None:
                    for line in self._params_text_widget.get("1.0", "end").splitlines():
                        line = line.strip()
                        if "=" in line and not line.startswith("#"):
                            name, val = line.split("=", 1)
                            try:
                                init_params[name.strip()] = float(val.strip())
                            except ValueError:
                                pass
                model_fit = _lmfit.ExpressionModel(expr, independent_vars=["x"])
                params = model_fit.make_params(**init_params)
                fitres = model_fit.fit(Gn, x=tau, params=params,
                                       method="least_squares")
                if "tau_diff" in fitres.values:
                    tau_diff_us = fitres.values["tau_diff"] * 1e6
                result_lines = [
                    f"Datei:   {os.path.basename(h5_path)}",
                    f"Modell:  {expr}",
                    f"Modus:   {mode}  |  Datenpunkte: {len(Gn)}",
                    "",
                    "Fit-Ergebnisse:",
                ]
                for name, p in fitres.params.items():
                    result_lines.append(f"  {name:20s} = {p.value:.6g}")
                result_lines.append(f"  Reduziertes χ²   = {fitres.redchi:.4e}")
                self._set_result("\n".join(result_lines))

        except Exception as e:
            self._set_result(
                f"Fit fehlgeschlagen:\n  {e}\n\n(Korrelation wird trotzdem gezeichnet)"
            )

        # Plot
        if not _matplotlib_ok:
            return

        if self._plot["fig"] is None:
            self._plot["fig"] = Figure(figsize=(10, 7), dpi=90, facecolor="#1a1a1a")
            self._plot["canvas"] = FigureCanvasTkAgg(
                self._plot["fig"], master=self.plot_frame)
            self._plot["canvas"].get_tk_widget().pack(fill="both", expand=True)

        fig = self._plot["fig"]
        fig.clear()

        gs  = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0,
                               left=0.09, right=0.97, top=0.93, bottom=0.09)
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1], sharex=ax0)

        BG, GR, TC = "#111111", "#2a2a4a", "#e0e0f0"
        for ax in (ax0, ax1):
            ax.set_facecolor(BG)
            ax.tick_params(colors=TC, labelsize=9)
            ax.xaxis.label.set_color(TC)
            ax.yaxis.label.set_color(TC)
            for sp in ax.spines.values():
                sp.set_edgecolor(GR)
            ax.grid(True, color=GR, lw=0.5, alpha=0.8)
            ax.grid(True, which="minor", color=GR, lw=0.3, alpha=0.5)

        ax0.semilogx(tau, Gn, color="#5bc8f5", lw=1.2, label="G(τ)")

        if fitres is not None:
            fit_label = (
                f"Fit ({m})  τ_D = {tau_diff_us:.1f} µs"
                if tau_diff_us is not None else f"Fit ({m})"
            )
            ax0.plot(tau, fitres.best_fit, color="#f5a623", lw=2.0, label=fit_label)
            res = fitres.residual
            ax1.plot(tau, res, color="#aaaaaa", lw=1.0)
            ym = max(float(np.abs(res).max()) * 1.2, 1e-9)
            ax1.set_ylim(-ym, ym)
            ax1.axhline(0, color="#f5a623", lw=0.8, ls="--")

        ax1.set_xlim(float(tau[0]) * 0.9, float(tau[-1]) * 1.1)
        ax0.set_ylabel("G(τ)", color=TC, fontsize=10)
        ax1.set_ylabel("Residuen", color=TC, fontsize=9)
        ax1.set_xlabel("Zeitverzögerung τ (s)", color=TC, fontsize=10)
        ax0.tick_params(labelbottom=False)
        ax0.legend(fontsize=9, facecolor=BG, labelcolor=TC, edgecolor=GR)
        ax0.set_title(os.path.basename(h5_path), color=TC, fontsize=10)

        self._plot["canvas"].draw()

    def _save_plot_png(self):
        """Speichert den FCS-Plot als PNG in den Ausgabeordner."""
        if self._plot["fig"] is None:
            return
        output_dir = self._get_output_dir()
        out_path = unique_output_path(
            output_dir, timestamped_filename("fcs_plot", "png"))
        self._plot["fig"].savefig(out_path, dpi=150, bbox_inches="tight",
                                  facecolor=self._plot["fig"].get_facecolor())
