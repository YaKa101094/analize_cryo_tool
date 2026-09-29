"""
analysis_krueger_simulation.py – Spot-Simulation für Analyse-Krüger
=========================================================================
Enthält die Klasse SpotSimulationDialog. Portiert die Testdaten-Generierung
aus spot_simulation.ipynb (Willi_Analyse_Tool/Software Willi): erzeugt
realistische, synthetische Einzelmolekül-Übersichtsscans mit bekannter
"Ground Truth" (wahre Molekülzahl, -positionen, -helligkeiten), um die
anderen Analyse-Krüger-Methoden (Spot-Zähler, ICS-Auswertung) damit zu
validieren.

Physik der Simulation (siehe Notebook für die vollständige Herleitung):
  1. Moleküle werden gleichverteilt über den Scanbereich platziert.
  2. PSF-Verschmierung über eine 2D-Gauß-Funktion (Breite psf_sigma_um).
  3. Helligkeits-Heterogenität zwischen Molekülen über eine Log-Normal-
     Verteilung (Variationskoeffizient brightness_cv), Mittelwert bleibt
     molecule_brightness.
  4. Konstanter Hintergrund + echtes Photonen-Schrotrauschen (Poisson).
  5. Zufällige Aufteilung der gezählten Photonen auf zwei Detektoren
     (binomialverteilt), sodass exakt Summe = Detektor1 + Detektor2 gilt.

Abweichend vom Notebook wird das .img-Ausgabeformat NICHT in Willis eigenem
Rohformat geschrieben, sondern im vom ImageViewer tatsächlich gelesenen
scanbin_s4-kompatiblen Format (88-Byte-Header, uint16-Bilddaten, siehe
image_processing.write_scanbin_img) — dadurch lassen sich die erzeugten
Testbilder direkt über "Bilder laden" im Hauptfenster öffnen und mit den
anderen Analyse-Krüger-Methoden auswerten.

Ursprüngliche Simulationsmethode entwickelt von Wilhelm Krüger.

Autor: Yannik Kasprzak, Institut für Physik, Universität zu Lübeck
"""

import csv
import os
import time
import tkinter as tk

import numpy as np

from image_processing import timestamped_filename, unique_output_path

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    _matplotlib_ok = True
except Exception as _e:
    print(f"[matplotlib import failed] {_e}")
    _matplotlib_ok = False


# ── Simulation (portiert aus dem Notebook) ──────────────────────────────────

def simulate_molecule_image(nx, ny, range_x_um, range_y_um, n_molecules,
                            molecule_brightness, psf_sigma_um, background_mean,
                            split_ratio=0.5, brightness_cv=0.0, rng=None):
    """Erzeugt ein simuliertes Bild-Tripel (Detektor1, Detektor2, Summe)."""
    if rng is None:
        rng = np.random.default_rng()

    pixel_size_x = range_x_um / nx
    pixel_size_y = range_y_um / ny
    x_centers = (np.arange(nx) + 0.5) * pixel_size_x
    y_centers = (np.arange(ny) + 0.5) * pixel_size_y
    X, Y = np.meshgrid(x_centers, y_centers)

    mol_x = rng.uniform(0, range_x_um, size=n_molecules)
    mol_y = rng.uniform(0, range_y_um, size=n_molecules)

    if brightness_cv > 0:
        sigma2 = np.log(1.0 + brightness_cv ** 2)
        mu = np.log(molecule_brightness) - sigma2 / 2.0
        mol_brightness = rng.lognormal(mean=mu, sigma=np.sqrt(sigma2), size=n_molecules)
    else:
        mol_brightness = np.full(n_molecules, molecule_brightness, dtype=float)

    expected = np.full((ny, nx), background_mean, dtype=float)
    two_sigma2 = 2.0 * psf_sigma_um ** 2
    for mx, my, b in zip(mol_x, mol_y, mol_brightness):
        expected += b * np.exp(-((X - mx) ** 2 + (Y - my) ** 2) / two_sigma2)

    total_photons = rng.poisson(expected)
    detector1 = rng.binomial(total_photons, split_ratio)
    detector2 = total_photons - detector1
    summe = total_photons

    info = dict(mol_x=mol_x, mol_y=mol_y, mol_brightness=mol_brightness)
    return detector1.astype(np.int32), detector2.astype(np.int32), summe.astype(np.int32), info


def save_spot_positions_csv(info, path, pixel_size_x_um, pixel_size_y_um):
    """Speichert die wahren (simulierten) Molekülpositionen als CSV ("Ground Truth")."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["molecule_id", "x_um", "y_um", "x_pixel", "y_pixel", "brightness_photons"])
        for idx, (mx, my, b) in enumerate(zip(info["mol_x"], info["mol_y"], info["mol_brightness"])):
            writer.writerow([
                idx, f"{mx:.4f}", f"{my:.4f}",
                f"{mx / pixel_size_x_um:.3f}", f"{my / pixel_size_y_um:.3f}", f"{b:.3f}",
            ])


def write_scanbin_img(path, detector1, detector2):
    """Schreibt ein .img im scanbin_s4-Fallback-Format des ImageViewers (88-Byte-
    Header mit Breite/Höhe bei Offset 24/28, danach uint16-Bilddaten,
    Frame-interleaved: Frame 0 = Detektor1, Frame 1 = Detektor2). Kompatibel
    mit ImageViewer.open_custom_img, damit erzeugte Testbilder direkt über
    "Bilder laden" nutzbar sind."""
    ny, nx = detector1.shape
    header = bytearray(88)
    header[24:28] = int(nx).to_bytes(4, byteorder="big", signed=False)
    header[28:32] = int(ny).to_bytes(4, byteorder="big", signed=False)

    det1_u16 = np.clip(detector1, 0, 65535).astype(">u2")
    det2_u16 = np.clip(detector2, 0, 65535).astype(">u2")
    with open(path, "wb") as f:
        f.write(bytes(header))
        f.write(det1_u16.tobytes())
        f.write(det2_u16.tobytes())


# ── Dialog / Tab ─────────────────────────────────────────────────────────────

class SpotSimulationDialog:
    """Erzeugt synthetische .img-Testdaten mit bekannter Molekülzahl/-position.

    Parameters
    ----------
    parent         : tk.Frame  Tab-Frame im Haupt-Notebook.
    get_output_dir : callable  Gibt den Ausgabepfad zurück (str); die erzeugten
                               Dateien landen in einem Unterordner "simulation".
    on_close       : callable | None  Wird vom "Schließen"-Button aufgerufen.
    """

    def __init__(self, parent, get_output_dir, on_close=None):
        self._get_output_dir = get_output_dir
        self._last_result = None
        self._plot = {"fig": None, "canvas": None}

        self.win = parent
        close_cmd = on_close if on_close is not None else self.win.destroy

        tk.Label(self.win, text="Spot-Simulation – synthetische Testdaten mit bekannter Ground Truth",
                 font=("Arial", 12, "bold")).pack(pady=(10, 4))

        self._build_param_frame()

        self._status_label = tk.Label(self.win, text="", justify="left", font=("Courier", 9))
        self._status_label.pack(pady=(0, 4))

        self._plot_frame = tk.Frame(self.win, bg="#1a1a1a")
        self._plot_frame.pack(fill="both", expand=True, padx=12, pady=(0, 4))

        btn_frame = tk.Frame(self.win)
        btn_frame.pack(pady=6)
        tk.Button(btn_frame, text="Bilder erzeugen", command=self._generate,
                  width=16).pack(side="left", padx=6)
        tk.Button(btn_frame, text="Ordner öffnen", command=self._open_output_folder,
                  width=14).pack(side="left", padx=6)
        tk.Button(btn_frame, text="Schließen", command=close_cmd,
                  width=12).pack(side="left", padx=6)

    # ── UI-Aufbau ─────────────────────────────────────────────────────────────

    def _build_param_frame(self):
        frame = tk.LabelFrame(self.win, text="Parameter", padx=8, pady=4)
        frame.pack(fill="x", padx=12, pady=4)

        self._n_images_var = tk.StringVar(value="30")
        self._n_mol_min_var = tk.StringVar(value="50")
        self._n_mol_max_var = tk.StringVar(value="50")
        self._nx_var = tk.StringVar(value="200")
        self._scan_um_var = tk.StringVar(value="20.0")
        self._background_var = tk.StringVar(value="3.0")
        self._brightness_var = tk.StringVar(value="40.0")
        self._brightness_cv_var = tk.StringVar(value="0.0")
        self._psf_sigma_var = tk.StringVar(value="0.15")
        self._split_ratio_var = tk.StringVar(value="0.5")
        self._dwell_ms_var = tk.StringVar(value="1.0")
        self._seed_var = tk.StringVar(value="")
        self._save_png_var = tk.BooleanVar(value=True)
        self._save_positions_var = tk.BooleanVar(value=True)

        input_frame = tk.Frame(frame)
        input_frame.pack(side="left", anchor="n")

        row_defs = [
            ("Anzahl Bilder:", self._n_images_var),
            ("Molekülzahl min:", self._n_mol_min_var),
            ("Molekülzahl max:", self._n_mol_max_var),
            ("Pixelzahl (nx=ny):", self._nx_var),
            ("Scanbereich (µm, quadratisch):", self._scan_um_var),
            ("Hintergrund (Photonen/Pixel):", self._background_var),
            ("Molekülhelligkeit (Photonen):", self._brightness_var),
            ("Helligkeits-CV (0 = alle gleich hell):", self._brightness_cv_var),
            ("PSF-Sigma (µm):", self._psf_sigma_var),
            ("Split-Ratio (Detektor1-Anteil):", self._split_ratio_var),
            ("Verweilzeit (ms, nur Metadaten):", self._dwell_ms_var),
            ("Zufalls-Seed (leer = zufällig):", self._seed_var),
        ]
        for idx, (lbl_text, var) in enumerate(row_defs):
            tk.Label(input_frame, text=lbl_text, anchor="e").grid(
                row=idx, column=0, sticky="e", padx=6, pady=2)
            tk.Entry(input_frame, textvariable=var, width=10).grid(
                row=idx, column=1, sticky="w", padx=6, pady=2)

        chk_row = len(row_defs)
        tk.Checkbutton(input_frame, text="Summenbild zusätzlich als PNG speichern",
                       variable=self._save_png_var).grid(
            row=chk_row, column=0, columnspan=2, sticky="w", padx=4, pady=(6, 0))
        tk.Checkbutton(input_frame, text="Wahre Molekülpositionen als CSV speichern",
                       variable=self._save_positions_var).grid(
            row=chk_row + 1, column=0, columnspan=2, sticky="w", padx=4)

        tk.Frame(frame, width=1, bg="#444444").pack(
            side="left", fill="y", padx=(12, 10), pady=2)

        desc_text = (
            "Pro Bild wird eine eigene, zufällige Molekülzahl aus\n"
            "[min, max] gezogen (gleich für alle Bilder bei min=max).\n\n"
            "Rauschmodell: echtes Photonen-Schrotrauschen (Poisson-\n"
            "Verteilung der erwarteten Helligkeit pro Pixel), danach\n"
            "binomialverteilte Aufteilung auf zwei Detektoren — exakt\n"
            "Summe = Detektor1 + Detektor2, wie am realen Strahlteiler.\n\n"
            "Helligkeits-CV  Log-Normal-Streuung der Molekülhelligkeit\n"
            "               um molecule_brightness (0 = alle exakt\n"
            "               gleich hell).\n\n"
            "Erzeugte .img-Dateien lassen sich direkt über \"Bilder\n"
            "laden\" im Hauptfenster öffnen und mit Spot-Zähler bzw.\n"
            "ICS-Auswertung gegen die bekannte Ground Truth testen."
        )
        tk.Label(frame, text=desc_text, justify="left", anchor="nw",
                 font=("Arial", 8), fg="#aaaaaa").pack(
            side="left", anchor="n", pady=4)

    # ── Erzeugung ─────────────────────────────────────────────────────────────

    def _generate(self):
        try:
            n_images = int(self._n_images_var.get())
            n_mol_min = int(self._n_mol_min_var.get())
            n_mol_max = int(self._n_mol_max_var.get())
            nx = ny = int(self._nx_var.get())
            scan_um = float(self._scan_um_var.get().replace(",", "."))
            background_mean = float(self._background_var.get().replace(",", "."))
            brightness = float(self._brightness_var.get().replace(",", "."))
            brightness_cv = float(self._brightness_cv_var.get().replace(",", "."))
            psf_sigma_um = float(self._psf_sigma_var.get().replace(",", "."))
            split_ratio = float(self._split_ratio_var.get().replace(",", "."))
            seed_str = self._seed_var.get().strip()
            seed = int(seed_str) if seed_str else None
        except ValueError:
            self._status_label.config(text="Ungültige Eingabe in den Parametern.")
            return

        if n_images < 1 or nx < 4 or scan_um <= 0 or psf_sigma_um <= 0:
            self._status_label.config(
                text="Anzahl Bilder, Pixelzahl, Scanbereich und PSF-Sigma müssen > 0 sein.")
            return
        if n_mol_min < 0 or n_mol_max < n_mol_min:
            self._status_label.config(text="Molekülzahl min/max ungültig.")
            return
        if not (0.0 < split_ratio < 1.0):
            self._status_label.config(text="Split-Ratio muss zwischen 0 und 1 liegen.")
            return

        output_dir = os.path.join(self._get_output_dir(), "simulation")
        os.makedirs(output_dir, exist_ok=True)
        png_dir = os.path.join(output_dir, "png")
        positions_dir = os.path.join(output_dir, "positions")
        if self._save_png_var.get():
            os.makedirs(png_dir, exist_ok=True)
        if self._save_positions_var.get():
            os.makedirs(positions_dir, exist_ok=True)

        rng = np.random.default_rng(seed)
        generated = []
        last_result = None

        for i in range(n_images):
            n_molecules = int(rng.integers(n_mol_min, n_mol_max + 1))
            det1, det2, summe, info = simulate_molecule_image(
                nx=nx, ny=ny, range_x_um=scan_um, range_y_um=scan_um,
                n_molecules=n_molecules, molecule_brightness=brightness,
                psf_sigma_um=psf_sigma_um, background_mean=background_mean,
                split_ratio=split_ratio, brightness_cv=brightness_cv, rng=rng,
            )

            timestamp_ms = int(time.time() * 1000)
            base = f"{timestamp_ms}_{i}_{n_molecules}mol"
            img_path = unique_output_path(output_dir, f"{base}.img")
            write_scanbin_img(img_path, det1, det2)
            generated.append(img_path)

            if self._save_png_var.get():
                import matplotlib.pyplot as _mplt
                vmax = max(int(summe.max()), 1)
                _mplt.imsave(os.path.join(png_dir, f"{base}_sum.png"),
                            summe, cmap="hot", vmin=0, vmax=vmax)

            if self._save_positions_var.get():
                save_spot_positions_csv(
                    info, os.path.join(positions_dir, f"{base}_positions.csv"),
                    pixel_size_x_um=scan_um / nx, pixel_size_y_um=scan_um / ny,
                )

            last_result = (det1, det2, summe, info, base)

        self._last_result = last_result
        self._status_label.config(
            text=(
                f"{len(generated)} Bild(er) erzeugt in {output_dir}"
                + (f"  |  PNGs: {png_dir}" if self._save_png_var.get() else "")
                + (f"  |  Positionen: {positions_dir}" if self._save_positions_var.get() else "")
            )
        )
        self._redraw(scan_um)

    def _open_output_folder(self):
        output_dir = os.path.join(self._get_output_dir(), "simulation")
        if os.path.isdir(output_dir):
            os.startfile(output_dir)

    # ── Darstellung ───────────────────────────────────────────────────────────

    def _redraw(self, scan_um):
        if not _matplotlib_ok or self._last_result is None:
            return
        det1, det2, summe, info, base = self._last_result

        if self._plot["fig"] is None:
            fig = Figure(figsize=(11, 3.2), dpi=90, facecolor="#1a1a1a")
            canvas = FigureCanvasTkAgg(fig, master=self._plot_frame)
            canvas.get_tk_widget().pack(fill="both", expand=True)
            self._plot["fig"] = fig
            self._plot["canvas"] = canvas
        fig = self._plot["fig"]
        fig.clear()

        axes = fig.subplots(1, 3)
        for ax, img, title in zip(axes, [det1, det2, summe], ["Detektor 1", "Detektor 2", "Summe"]):
            ax.set_facecolor("#1a1a1a")
            im = ax.imshow(img, cmap="hot", origin="upper", extent=[0, scan_um, scan_um, 0])
            ax.set_title(title, color="white", fontsize=9)
            ax.set_xlabel("x (µm)", color="white", fontsize=8)
            ax.set_ylabel("y (µm)", color="white", fontsize=8)
            ax.tick_params(colors="white", labelsize=7)

        max_b = info["mol_brightness"].max() if info["mol_brightness"].size else 1.0
        sizes = 12 + 35 * info["mol_brightness"] / max(max_b, 1.0)
        axes[2].scatter(info["mol_x"], info["mol_y"], s=sizes, facecolors="none",
                        edgecolors="cyan", linewidths=1.0)

        fig.suptitle(f"{base}  (letztes erzeugtes Bild, {len(info['mol_x'])} Moleküle)",
                    color="white", fontsize=9)
        fig.tight_layout(pad=1.0)
        self._plot["canvas"].draw()
