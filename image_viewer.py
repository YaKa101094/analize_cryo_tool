"""
image_viewer.py – Hauptfenster des Image Browser & Analysis Tools
=================================================================
Enthält die Klasse ImageViewer: Bildladen, Grid- und Einzelkanal-Ansicht,
Navigation, Export und Delegation an die drei Analyse-Dialoge.

Autor: Yannik Kasprzak, Institut für Physik, Universität zu Lübeck
"""

import math
import os
import re
import subprocess
import sys
import tkinter as tk
from datetime import datetime
from tkinter import Menu, StringVar, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

# ── Eigene Module ─────────────────────────────────────────────────────────────
import image_processing as _ip
from analysis_timetrace import TimeraceDialog
from analysis_schichtdicke import SchichtdickeDialog
from analysis_ics import ICSDialog
from analysis_spot_zaehler import SpotZaehlerDialog
from analysis_krueger_ics import KruegerICSDialog
from analysis_krueger_simulation import SpotSimulationDialog
from analysis_krueger_zstack import ZStackDialog

# ── Optionale Abhängigkeiten ──────────────────────────────────────────────────

try:
    import photon_tools as pt
except Exception:
    pt = None

try:
    import matplotlib.cm as _mpl_cm
    _matplotlib_ok = True
except Exception as _e:
    print(f"[matplotlib import failed] {_e}")
    _matplotlib_ok = False


def _natural_sort_key(s):
    """Sortierschlüssel für "natürliche" Reihenfolge (1, 2, ..., 10 statt 1, 10, 2)."""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", s)]


# Methoden im Tab "Analyse-Krüger" — alle in C:\Users\yanni\Desktop\Software Willi
# als eigenständige Jupyter-Notebooks entwickelt (Wilhelm Krüger), hier als
# auswählbare Untermenü-Punkte innerhalb eines gemeinsamen Tabs zusammengefasst
# (siehe ImageViewer._rebuild_krueger_tab).
KRUEGER_METHODS = [
    ("Spot-Zähler (Methode A)", "spot_zaehler"),
    ("ICS-Auswertung (Methode B)", "ics"),
    ("Spot-Simulation", "simulation"),
    ("Z-Stack (PSF)", "zstack"),
]

# Methoden im Tab "Analyse-Kasprzak" — die ursprünglichen, eigenständigen
# Analyse-Tabs (Timetrace/FCS, Schichtdicke Zählen/ICS), nach demselben
# Methodenauswahl-Prinzip wie Analyse-Krüger zusammengefasst (siehe
# ImageViewer._rebuild_kasprzak_tab).
KASPRZAK_METHODS = [
    ("Timetrace / FCS", "timetrace"),
    ("Schichtdicke (Zählen)", "schichtdicke"),
    ("Schichtdicke (ICS)", "ics"),
]


class ImageViewer:
    """Hauptanwendungsfenster.  Verwaltet Datensätze, Widgets und Unterfenster.

    Instance variables
    ------------------
    datasets      : list[dict]  Geladene Bilddatensätze.  Jeder Eintrag:
                                {"path", "name", "detector0", "detector1", "sum"}
    current_index : int         Index des aktuell angezeigten Datensatzes
    threshold_value : float     Relativer Schwellwert [0, 1] für Spot-Zählung
    _photo_refs   : list        GC-Schutz für ImageTk.PhotoImage-Objekte
    _img_loader_name : str      photon_tools-Loader für .img-Dateien
    """

    # ── Initialisierung & UI ──────────────────────────────────────────────────

    def __init__(self, root):
        """Initialisiert den ImageViewer im gegebenen Tk-Hauptfenster.

        Parameters
        ----------
        root : tk.Tk
            Das leere Hauptfenster (direkt von Schichtdicke.py übergeben).
        """
        self.root = root
        self.root.title("Image Browser & Analysis Tool")
        self.root.state("zoomed")                           # Vollbild beim Start
        self.root.protocol("WM_DELETE_WINDOW", self.exit_app)

        # Datenspeicher
        self.image_paths    = []
        self.datasets       = []
        self.filtered_images = []
        self.current_index  = 0
        self.threshold_value = 0.0
        self._img_dir       = None      # Verzeichnis der zuletzt geladenen Bilder

        # Tkinter-Steuervariablen
        self.display_mode_var = StringVar(value="grid_sum")
        self.rows_var         = tk.IntVar(value=3)
        self.cols_var         = tk.IntVar(value=4)
        self.file_var         = StringVar(value="")
        self.save_raster_var  = tk.BooleanVar(value=False)
        self.show_spots_var   = tk.BooleanVar(value=False)
        self.bg_correction_var = tk.BooleanVar(value=False)
        self.save_names_var   = tk.BooleanVar(value=True)
        self.grid_filenames_var = tk.BooleanVar(value=False)

        self._photo_refs       = []     # GC-Schutz für PhotoImage-Objekte
        self._img_loader_name  = "scanbin_s4"

        # Manuell in der Bildansicht (show_image_viewer) hinzugefügte Spots,
        # je Datensatz-Name: dataset["name"] -> [(row, col), ...] in
        # Original-Pixelkoordinaten. Nur aktiv/sichtbar wenn show_spots_var
        # aktiviert ist; wirkt sich nicht auf die automatische Zählung
        # anderswo (Grid-Beschriftung, Schichtdicke-Dialog) aus.
        self._manual_spots = {}

        # Bildauswahl: leere Menge = alle Bilder aktiv
        self.selected_indices: set = set()
        self._tile_wrappers:   dict = {}   # idx → wrapper-Frame (für Highlight-Update)
        self._sel_vars:        dict = {}   # idx → BooleanVar (Checkbox-Zustand)

        self.setup_ui()
        self.load_images()
        self._build_tool_tabs()
        self.notebook.select(self._bildanalyse_tab)

    def setup_ui(self):
        """Baut die gesamte Hauptoberfläche auf.

        Aufbau (von oben nach unten):
          1. Menüleiste  – Datei / Analyse / Hilfe
          2. Obere Toolbar – Navigation, Dateiauswahl, Anzeigemodus, Grid-Größe,
                            Threshold-Slider, Speichern-Button
          3. Scrollbarer Canvas-Bereich – zeigt Bildkacheln
          4. Untere Toolbar – Navigation + Detailinfo
        """
        # ── Menüleiste ────────────────────────────────────────────────────────
        menu_bar = Menu(self.root)
        self.root.config(menu=menu_bar)

        file_menu = Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Pfad öffnen", command=self.open_path)
        file_menu.add_command(label="Bilder laden", command=self.load_images)
        file_menu.add_separator()
        file_menu.add_command(label="Beenden", command=self.exit_app)
        menu_bar.add_cascade(label="Datei", menu=file_menu)

        hilfe_menu = Menu(menu_bar, tearoff=0)
        hilfe_menu.add_command(label="Info", command=self.show_about)
        hilfe_menu.add_command(label="About", command=self.show_author)
        menu_bar.add_cascade(label="Hilfe", menu=hilfe_menu)

        # ── Haupt-Notebook ────────────────────────────────────────────────────
        # "Bildanalyse" ist ein fester, nicht schließbarer erster Tab. Die
        # übrigen Analyse-Werkzeuge sind von Anfang an als weitere, schließbare
        # Tabs mit Methodenauswahl vorhanden (Analyse-Kasprzak, Analyse-Krüger
        # — siehe _build_tool_tabs) - kein separates Menü zum Öffnen mehr nötig.
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)

        self._bildanalyse_tab = tk.Frame(self.notebook)
        self.notebook.add(self._bildanalyse_tab, text="Bildanalyse")

        main_frame = tk.Frame(self._bildanalyse_tab)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # ── Obere Toolbar (nur Navigation & Ansicht) ──────────────────────────
        control_top = tk.Frame(main_frame)
        control_top.pack(side="top", fill="x", pady=5)

        self.prev_btn_top = tk.Button(control_top, text="◀ Prev",
                                      command=self.prev_image)
        self.prev_btn_top.pack(side="left", padx=5)
        self.next_btn_top = tk.Button(control_top, text="Next ▶",
                                      command=self.next_image)
        self.next_btn_top.pack(side="left", padx=5)

        tk.Label(control_top, text="File:").pack(side="left", padx=(12, 4))
        self.file_combo = ttk.Combobox(control_top, state="readonly",
                                        width=8, textvariable=self.file_var)
        self.file_combo.bind("<<ComboboxSelected>>", self.on_file_select)
        self.file_combo.pack(side="left", padx=5)

        tk.Label(control_top, text="Display:").pack(side="left", padx=(12, 4))
        self.display_combo = ttk.Combobox(
            control_top, state="readonly", width=20,
            textvariable=self.display_mode_var,
            values=["all", "det0_det1", "detector0", "detector1", "sum", "grid_sum"],
        )
        self.display_combo.bind("<<ComboboxSelected>>",
                                lambda _e: self.update_display())
        self.display_combo.pack(side="left", padx=5)

        tk.Label(control_top, text="Rows:").pack(side="left", padx=(12, 4))
        tk.Spinbox(control_top, from_=1, to=10, width=4,
                   textvariable=self.rows_var,
                   command=self.update_display).pack(side="left", padx=4)

        tk.Label(control_top, text="Cols:").pack(side="left", padx=(8, 4))
        tk.Spinbox(control_top, from_=1, to=10, width=4,
                   textvariable=self.cols_var,
                   command=self.update_display).pack(side="left", padx=4)

        tk.Label(control_top, text="Threshold:").pack(side="left", padx=(12, 4))
        self.threshold_scale = tk.Scale(
            control_top, from_=0.0, to=1.0, resolution=0.05,
            orient=tk.HORIZONTAL, length=230, command=self.update_threshold,
        )
        self.threshold_scale.set(0.0)
        self.threshold_scale.pack(side="left", padx=5)

        # ── Untere Toolbar ────────────────────────────────────────────────────
        # Vor dem expand=True-Bereich packen, damit er Platz reserviert.
        control_bottom = tk.Frame(main_frame)
        control_bottom.pack(side="bottom", fill="x", pady=5)

        self.details_label = tk.Label(control_bottom, text="",
                                       justify="left", anchor="w")
        self.details_label.pack(side="left", padx=12)

        # ── Inhaltsbereich: Bildcanvas links + rechtes Menü ───────────────────
        content_frame = tk.Frame(main_frame)
        content_frame.pack(fill="both", expand=True)

        # ── Rechtes Seitenmenü ────────────────────────────────────────────────
        right_panel = tk.Frame(content_frame, width=195, relief="groove", bd=1)
        right_panel.pack(side="right", fill="y", padx=(8, 0))
        right_panel.pack_propagate(False)   # fixe Breite erzwingen

        # Abschnitt: Datei (dieselben Aktionen wie im Datei-Menü)
        frm_datei = tk.LabelFrame(right_panel, text="Datei",
                                   padx=8, pady=6, font=("Arial", 9, "bold"))
        frm_datei.pack(fill="x", padx=6, pady=(10, 4))
        tk.Button(frm_datei, text="Bilder laden",
                  command=self.load_images).pack(fill="x", pady=(0, 3))
        tk.Button(frm_datei, text="Pfad öffnen",
                  command=self.open_path).pack(fill="x")

        # Abschnitt: Anzeige
        frm_anzeige = tk.LabelFrame(right_panel, text="Anzeige",
                                     padx=8, pady=6, font=("Arial", 9, "bold"))
        frm_anzeige.pack(fill="x", padx=6, pady=4)
        tk.Checkbutton(frm_anzeige, text="Spots einzeichnen",
                       variable=self.show_spots_var,
                       command=self.update_display).pack(anchor="w")
        tk.Checkbutton(frm_anzeige, text="Hintergrundkorrektur",
                       variable=self.bg_correction_var,
                       command=self.update_display).pack(anchor="w")

        # Abschnitt: Bildauswahl
        frm_sel = tk.LabelFrame(right_panel, text="Bildauswahl",
                                 padx=8, pady=6, font=("Arial", 9, "bold"))
        frm_sel.pack(fill="x", padx=6, pady=4)
        tk.Button(frm_sel, text="Alle auswählen",
                  command=self._select_all).pack(fill="x", pady=(0, 3))
        tk.Button(frm_sel, text="Auswahl aufheben",
                  command=self._clear_selection).pack(fill="x", pady=(0, 6))
        self._selection_label = tk.Label(frm_sel, text="alle aktiv",
                                          font=("Arial", 8), fg="#888888")
        self._selection_label.pack(anchor="w")

        # Abschnitt: Speichern
        frm_save = tk.LabelFrame(right_panel, text="Speichern",
                                  padx=8, pady=6, font=("Arial", 9, "bold"))
        frm_save.pack(fill="x", padx=6, pady=4)
        tk.Checkbutton(frm_save, text="Namen mitspeichern",
                       variable=self.save_names_var).pack(anchor="w")
        tk.Checkbutton(frm_save, text="Rasterbild",
                       variable=self.save_raster_var).pack(anchor="w")
        tk.Checkbutton(frm_save, text="   Dateinamen statt Nummern",
                       variable=self.grid_filenames_var).pack(anchor="w", pady=(0, 6))
        tk.Button(frm_save, text="Bilder speichern",
                  command=self.save_images_png).pack(fill="x", pady=(0, 6))
        tk.Button(frm_save, text="Output öffnen",
                  command=self._open_output_folder).pack(fill="x")

        # Statusanzeige am unteren Rand des Menüs
        self.status_label = tk.Label(right_panel, text="",
                                      anchor="w", justify="left",
                                      wraplength=148, font=("Arial", 8), fg="#555555")
        self.status_label.pack(padx=6, pady=(10, 4), anchor="w")

        tk.Button(right_panel, text="Schließen",
                  command=self.root.destroy).pack(
            side="bottom", fill="x", padx=6, pady=(0, 8))

        # ── Scrollbarer Bildbereich ───────────────────────────────────────────
        self.image_container = tk.Frame(content_frame)
        self.image_container.pack(side="left", fill="both", expand=True)

        self.image_canvas = tk.Canvas(self.image_container, highlightthickness=0)
        self.scrollbar_y  = tk.Scrollbar(self.image_container, orient="vertical",
                                          command=self.image_canvas.yview)
        self.scrollbar_x  = tk.Scrollbar(self.image_container, orient="horizontal",
                                          command=self.image_canvas.xview)
        self.image_canvas.configure(yscrollcommand=self.scrollbar_y.set,
                                     xscrollcommand=self.scrollbar_x.set)
        self.scrollbar_y.pack(side="right", fill="y")
        self.scrollbar_x.pack(side="bottom", fill="x")
        self.image_canvas.pack(side="left", fill="both", expand=True)

        # Innerer Frame im Canvas hält alle Bildkacheln.
        self.image_frame = tk.Frame(self.image_canvas)
        self._canvas_window = self.image_canvas.create_window(
            (0, 0), window=self.image_frame, anchor="nw")

        self.image_frame.bind("<Configure>", self._on_content_configure)
        self.image_canvas.bind("<Configure>", self._on_canvas_configure)
        self.image_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    # ── Datei-Laden ───────────────────────────────────────────────────────────

    def open_path(self):
        """Öffnet das aktuelle Bildverzeichnis im Windows Explorer."""
        if self._img_dir and os.path.isdir(self._img_dir):
            os.startfile(self._img_dir)
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            fallback = os.path.join(script_dir, "img")
            os.makedirs(fallback, exist_ok=True)
            os.startfile(fallback)

    def load_images(self, img_dir=None):
        """Durchsucht *img_dir* nach Bilddateien, baut Datensätze auf und zeigt sie an.

        Parameters
        ----------
        img_dir : str | None
            Zu ladender Ordner.  None → img/-Unterordner neben der Skriptdatei.
        """
        if img_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            img_dir = os.path.join(script_dir, "img")

        if not os.path.isdir(img_dir):
            self._set_empty_state(f"Ordner nicht gefunden: {img_dir}")
            return

        self._img_dir = img_dir
        allowed_extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff",
                               ".bmp", ".webp", ".img")
        image_paths = [
            os.path.join(img_dir, f)
            for f in sorted(os.listdir(img_dir), key=_natural_sort_key)
            if os.path.isfile(os.path.join(img_dir, f))
            and f.lower().endswith(allowed_extensions)
        ]

        if not image_paths:
            self._set_empty_state(f"Keine unterstützten Bilder in {img_dir}")
            return

        self.image_paths = image_paths
        self.datasets = [ds for ds in (self.open_image(p) for p in image_paths)
                         if ds is not None]

        if not self.datasets:
            self._set_empty_state("Bilder gefunden, aber nicht lesbar")
            return

        self.current_index = 0
        self.selected_indices.clear()
        self._sel_vars.clear()
        self._tile_wrappers.clear()
        self.file_combo["values"] = [f"{i + 1:03d}" for i in range(len(self.datasets))]
        self.file_var.set("001")
        self.status_label.config(
            text=f"{len(self.datasets)} Bild(er) geladen aus {os.path.basename(img_dir)}"
        )
        self.update_display()
        self._update_selection_label()
        self._maybe_refresh_dataset_tabs()

    def _maybe_refresh_dataset_tabs(self):
        """Aktualisiert die Methodenauswahl-Tabs, falls sie bereits existieren.

        Verhindert einen doppelten Aufbau beim allerersten Start: Dort ruft
        __init__ load_images() VOR _build_tool_tabs() auf, das den initialen
        Aufbau übernimmt. Bei jedem späteren Neuladen (z. B. "Bilder laden")
        existieren die Tabs schon und werden hier aktualisiert.
        """
        if hasattr(self, "_kasprzak_frame"):
            self._refresh_dataset_tabs()

    def _set_empty_state(self, message: str):
        """Leert alle Datenspeicher und zeigt *message* in der Statusleiste."""
        self.image_paths = []
        self.datasets = []
        self.file_combo["values"] = []
        self.file_var.set("")
        self.status_label.config(text=message)
        self.details_label.config(text="")
        for widget in self.image_frame.winfo_children():
            widget.destroy()
        self.image_canvas.yview_moveto(0.0)
        self.image_canvas.xview_moveto(0.0)
        self._maybe_refresh_dataset_tabs()

    def open_image(self, path):
        """Lädt eine Bilddatei und gibt einen Datensatz-Dict zurück (None bei Fehler)."""
        _, ext = os.path.splitext(path)
        ext = ext.lower()
        try:
            if ext == ".img":
                if pt is not None:
                    try:
                        ds = pt.load_image(path, loader=self._img_loader_name)
                        det0 = np.asarray(ds.channels["detector0"], dtype=np.float32)
                        det1 = np.asarray(ds.channels["detector1"], dtype=np.float32)
                        return {"path": path, "name": os.path.basename(path),
                                "detector0": det0, "detector1": det1,
                                "sum": det0 + det1}
                    except Exception:
                        pass
                return self.open_custom_img(path)

            # Standard-Rasterformate: als Graustufen laden
            arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
            zeros = np.zeros_like(arr, dtype=np.float32)
            return {"path": path, "name": os.path.basename(path),
                    "detector0": arr, "detector1": zeros, "sum": arr}
        except Exception:
            return None

    def open_custom_img(self, path):
        """Liest eine scanbin_s4-.img-Datei ohne photon_tools (Fallback-Parser).

        Binärformat (big-endian):
          Bytes   0–87 : Header
            Offset 24–27 : Bildbreite  (uint32, big-endian)
            Offset 28–31 : Bildhöhe   (uint32, big-endian)
          Bytes  88–   : Pixeldaten als uint16, Frame-interleaved:
                          Frame 0 (det0), Frame 1 (det1), Frame 2 (det0), …

        Alle Frames eines Detektors werden aufsummiert (Integriert).
        """
        with open(path, "rb") as fobj:
            data = fobj.read()
        if len(data) <= 88:
            raise ValueError(".img-Datei ist zu klein")

        header  = data[:88]
        payload = data[88:]
        width   = int.from_bytes(header[24:28], byteorder="big", signed=False)
        height  = int.from_bytes(header[28:32], byteorder="big", signed=False)

        if width <= 0 or height <= 0:
            raise ValueError("Ungültige Bilddimensionen im .img-Header")

        raw_u16 = np.frombuffer(payload, dtype=">u2")
        pixels_per_frame = width * height
        # Auf ganze Frames kürzen
        total_values = (len(raw_u16) // pixels_per_frame) * pixels_per_frame
        if total_values == 0:
            raise ValueError("Keine vollständigen Frames in .img-Datei gefunden")

        n_frames = total_values // pixels_per_frame
        frames   = raw_u16[:total_values].reshape(n_frames, height, width).astype(np.float32)

        # Gerade Frames → detector0, ungerade → detector1
        det0_frames = frames[0::2]
        det1_frames = frames[1::2]
        detector0 = (det0_frames.sum(axis=0) if det0_frames.size
                     else np.zeros((height, width), dtype=np.float32))
        detector1 = (det1_frames.sum(axis=0) if det1_frames.size
                     else np.zeros((height, width), dtype=np.float32))

        return {"path": path, "name": os.path.basename(path),
                "detector0": detector0, "detector1": detector1,
                "sum": detector0 + detector1}

    # ── Canvas-Helfer ─────────────────────────────────────────────────────────

    def _on_content_configure(self, _event=None):
        """Hält die Scroll-Region des Canvas mit dem inneren Frame synchron."""
        self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        """Streckt den inneren Frame immer auf die volle Canvas-Breite."""
        self.image_canvas.itemconfigure(self._canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        """Scrollt den Haupt-Canvas vertikal mit dem Mausrad."""
        if not self.datasets:
            return
        # Scroll nur im Hauptfenster, nicht in Toplevel-Dialogen
        widget = event.widget
        while widget is not None:
            if isinstance(widget, tk.Toplevel):
                return
            widget = getattr(widget, "master", None)
        self.image_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # ── Bildverarbeitung (Delegation an image_processing) ─────────────────────

    def _to_color_image(self, image_array, channel=""):
        """Delegiert an ip.to_color_image mit aktuellem threshold_value."""
        return _ip.to_color_image(image_array, channel, self.threshold_value)

    def _display_range(self, image_array):
        """Delegiert an ip.display_range mit aktuellem threshold_value."""
        return _ip.display_range(image_array, self.threshold_value)

    def _count_spots(self, img, threshold_fraction, dataset=None):
        n = _ip.count_spots(img, float(threshold_fraction))
        if dataset is not None:
            n += len(self._manual_spots.get(dataset["name"], []))
        return n

    def _count_spots_notebook(self, img, threshold_fraction):
        return _ip.count_spots(img, float(threshold_fraction))

    def _get_channel_image(self, dataset, channel):
        return _ip.get_channel_array(
            dataset, channel, bg_correction=self.bg_correction_var.get())

    def update_threshold(self, value):
        """Wird vom Threshold-Slider aufgerufen; aktualisiert die Anzeige."""
        self.threshold_value = float(value)
        self.update_display()

    def _center_zoom_array(self, image_array, zoom_factor=5.5):
        """Gibt den zentralen Ausschnitt von *image_array* beim gegebenen Zoom zurück."""
        if image_array.ndim != 2 or image_array.size == 0:
            return image_array
        if zoom_factor <= 1.0:
            return image_array
        height, width = image_array.shape
        crop_h = max(1, int(height / zoom_factor))
        crop_w = max(1, int(width / zoom_factor))
        y0 = (height - crop_h) // 2
        x0 = (width - crop_w) // 2
        return image_array[y0:y0 + crop_h, x0:x0 + crop_w]

    # ── Bildauswahl ───────────────────────────────────────────────────────────

    def _active_datasets(self):
        """Gibt die aktiven Datensätze zurück.

        Wenn eine Auswahl getroffen wurde, werden nur die ausgewählten Indizes
        zurückgegeben (sortiert).  Leere Auswahl = alle Datensätze.
        """
        if self.selected_indices:
            return [self.datasets[i] for i in sorted(self.selected_indices)]
        return self.datasets

    def _toggle_selection(self, idx: int):
        """Schaltet den Auswahlzustand von Bild *idx* um und aktualisiert die Optik."""
        if idx in self.selected_indices:
            self.selected_indices.discard(idx)
        else:
            self.selected_indices.add(idx)
        selected = idx in self.selected_indices
        # Wrapper-Hintergrund sofort aktualisieren (kein vollständiges Neurendern)
        if idx in self._tile_wrappers:
            bg = "#c8a800" if selected else self.image_frame.cget("bg")
            self._tile_wrappers[idx].config(
                bg=bg,
                padx=3 if selected else 0,
                pady=3 if selected else 0,
            )
        self._update_selection_label()
        self._maybe_refresh_dataset_tabs()

    def _select_all(self):
        """Wählt alle geladenen Bilder aus."""
        self.selected_indices = set(range(len(self.datasets)))
        for var in self._sel_vars.values():
            var.set(True)
        for wrapper in self._tile_wrappers.values():
            wrapper.config(bg="#c8a800", padx=3, pady=3)
        self._update_selection_label()
        self._maybe_refresh_dataset_tabs()

    def _clear_selection(self):
        """Hebt alle Auswahlen auf; alle Bilder werden wieder aktiv."""
        self.selected_indices.clear()
        for wrapper in self._tile_wrappers.values():
            wrapper.config(bg=self.image_frame.cget("bg"), padx=0, pady=0)
        for var in self._sel_vars.values():
            var.set(False)
        self._update_selection_label()
        self._maybe_refresh_dataset_tabs()

    def _update_selection_label(self):
        """Aktualisiert das Zähler-Label in der Toolbar."""
        n = len(self.selected_indices)
        total = len(self.datasets)
        if n == 0:
            self._selection_label.config(text="alle aktiv")
        else:
            self._selection_label.config(text=f"{n} / {total} ausgewählt")

    def _draw_spots_overlay(self, color_img: Image.Image,
                            image_array: np.ndarray, radius: int = 8,
                            dataset: dict = None) -> Image.Image:
        """Zeichnet gelbe Kreise an den Spot-Positionen auf *color_img* ein.

        Die Koordinaten stammen aus find_spots (8×8-Moore-Detektion), ergänzt
        um manuell in der Bildansicht hinzugefügte Punkte für *dataset*
        (siehe self._manual_spots) — beide werden gleich dargestellt.
        *color_img* muss nicht dieselbe Größe wie *image_array* haben —
        die Positionen werden auf die Display-Größe skaliert.

        Parameters
        ----------
        color_img   : PIL.Image  Ziel-Bild (bereits in Displaygröße skaliert).
        image_array : np.ndarray Originaldaten für die Spot-Detektion.
        radius      : int        Kreisradius in Display-Pixeln.
        dataset     : dict       Datensatz, um manuelle Punkte einzubeziehen.

        Returns
        -------
        PIL.Image  Kopie von *color_img* mit eingezeichneten Kreisen.
        """
        coords = list(map(tuple, _ip.find_spots(image_array, self.threshold_value)))
        if dataset is not None:
            coords += self._manual_spots.get(dataset["name"], [])
        if len(coords) == 0:
            return color_img
        orig_h, orig_w = image_array.shape
        disp_w, disp_h = color_img.size
        scale_x = disp_w / orig_w
        scale_y = disp_h / orig_h
        lw = max(1, radius // 4)
        # Transparente Kreise via RGBA-Composite (alpha=170 ≈ 67 % Deckkraft).
        base   = color_img.convert("RGBA")
        circles = Image.new("RGBA", color_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(circles)
        for row, col in coords:
            # +0.5: Zentrum des Quellpixels statt seiner oberen linken Ecke.
            cx = int((col + 0.5) * scale_x)
            cy = int((row + 0.5) * scale_y)
            draw.ellipse(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                outline=(255, 255, 0, 170), width=lw,
            )
        return Image.alpha_composite(base, circles).convert("RGB")

    # ── Darstellung & Rendering ───────────────────────────────────────────────

    def _add_intensity_scale(self, parent, min_value, max_value, height, channel=""):
        """Rendert eine vertikale Falschfarb-Intensitätsskala neben einer Bildkachel."""
        scale_wrap = tk.Frame(parent)
        scale_wrap.pack(side="left", padx=(6, 0), fill="y")

        tk.Label(scale_wrap, text=f"{max_value:.1f}", font=("Arial", 8)).pack(anchor="w")

        # 30 px Reserve für Max/Min-Beschriftung oben und unten.
        grad_h   = max(40, int(height) - 30)
        grad_arr = np.linspace(255, 0, grad_h, dtype=np.uint8).reshape(grad_h, 1)
        grad_img = Image.fromarray(grad_arr, mode="L").resize(
            (18, grad_h), Image.Resampling.NEAREST)

        if channel == "detector1" and _matplotlib_ok:
            grad_1d  = np.linspace(1.0, 0.0, grad_h)
            rgba     = _mpl_cm.cool(grad_1d)
            rgb_bar  = np.tile(
                (rgba[:, :3] * 255).astype(np.uint8)[:, np.newaxis, :], (1, 18, 1))
            grad_color = Image.fromarray(rgb_bar, mode="RGB")
        elif channel == "detector1":
            from PIL import ImageOps
            grad_color = ImageOps.colorize(
                grad_img, black="#000000", mid="#00aa00", white="#ccffcc")
        else:
            from PIL import ImageOps
            grad_color = ImageOps.colorize(
                grad_img, black="#000000", mid="#cc3300", white="#ffffcc")

        grad_photo = ImageTk.PhotoImage(grad_color)
        self._photo_refs.append(grad_photo)

        bar = tk.Label(scale_wrap, image=grad_photo, bd=1, relief="solid")
        bar.pack(anchor="w")
        bar.image = grad_photo  # extra GC-Schutz auf dem Label

        tk.Label(scale_wrap, text=f"{min_value:.1f}", font=("Arial", 8)).pack(anchor="w")

    def show_image_viewer(self, image_array, title_text, channel="", dataset=None):
        """Öffnet einen zoombaren Vollbild-Betrachter mit Next/Prev-Navigation.

        Zoom-Mathematik (Mausrad)
        -------------------------
        Sei cx, cy die Canvas-Koordinaten des Cursors.  Nach dem Zoom gilt:
            xview_moveto = (cx × zoom_ratio − event.x) / neue_Bildbreite
        Das hält den Bildpixel unter dem Cursor an seiner Fensterposition.

        Navigation
        ----------
        "Vorheriges"/"Nächstes" wechselt innerhalb von self.datasets bei
        gleichbleibendem Kanal (der Kanal, mit dem der Betrachter geöffnet
        wurde) und lädt das Bild im selben Fenster neu, statt es zu schließen.
        """
        win = tk.Toplevel(self.root)
        win.state("zoomed")

        title_label = tk.Label(win, font=("Arial", 11, "bold"))
        title_label.pack(pady=(8, 2))
        minmax_label = tk.Label(win, font=("Arial", 9))
        minmax_label.pack(pady=(0, 2))
        spots_info_label = tk.Label(win, font=("Arial", 9), fg="#5bc8f5")
        spots_info_label.pack(pady=(0, 2))

        content_frame = tk.Frame(win)
        content_frame.pack(fill="both", expand=True, padx=6, pady=2)

        canvas_frame = tk.Frame(content_frame)
        canvas_frame.pack(side="left", fill="both", expand=True)

        sb_y = tk.Scrollbar(canvas_frame, orient="vertical")
        sb_x = tk.Scrollbar(canvas_frame, orient="horizontal")
        canvas = tk.Canvas(canvas_frame, bg="#1a1a1a", highlightthickness=0,
                           yscrollcommand=sb_y.set, xscrollcommand=sb_x.set)
        sb_y.config(command=canvas.yview)
        sb_x.config(command=canvas.xview)
        sb_y.pack(side="right", fill="y")
        sb_x.pack(side="bottom", fill="x")
        canvas.pack(side="left", fill="both", expand=True)

        scale_holder = tk.Frame(content_frame)
        scale_holder.pack(side="left", fill="y")

        zoom_state = [1.0]
        photo_ref  = [None]
        state = {}   # Zustand des aktuell angezeigten Bildes; von apply_state() befüllt

        def update_spots_label():
            if not self.show_spots_var.get():
                spots_info_label.config(text="")
                return
            n_auto = len(state["spot_coords"]) if state["spot_coords"] is not None else 0
            n_manual = len(state["manual_coords"])
            spots_info_label.config(text=(
                f"Spots: {n_auto} automatisch + {n_manual} manuell = {n_auto + n_manual}   "
                f"(Klick auf leere Stelle: hinzufügen  |  Klick auf Punkt: entfernen)"
            ))

        def redraw():
            z = zoom_state[0]
            src_w, src_h = state["src_w"], state["src_h"]
            new_w = max(1, int(src_w * z))
            new_h = max(1, int(src_h * z))
            resized = state["color_img"].resize((new_w, new_h), Image.Resampling.NEAREST)
            spot_coords   = state["spot_coords"]
            manual_coords = state["manual_coords"]
            has_auto = spot_coords is not None and len(spot_coords) > 0
            if has_auto or manual_coords:
                orig_h, orig_w = state["image_array"].shape
                sx = new_w / orig_w
                sy = new_h / orig_h
                # Radius = halbe Breite der 8×8-Moore-Nachbarschaft aus find_spots,
                # auf die aktuelle Zoomstufe skaliert.
                r  = max(2, int(4 * min(sx, sy)))
                lw = max(1, r // 4)
                base    = resized.convert("RGBA")
                circles = Image.new("RGBA", resized.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(circles)
                if has_auto:
                    for row, col in spot_coords:
                        # +0.5: Zentrum des Quellpixels statt seiner oberen linken Ecke.
                        cx = int((col + 0.5) * sx)
                        cy = int((row + 0.5) * sy)
                        draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                                     outline=(255, 255, 0, 170), width=lw)
                # Manuelle Punkte: cyan mit Kreuz, um sie von automatischen
                # (gelbe Kreise) optisch klar zu unterscheiden.
                for row, col in manual_coords:
                    cx = int((col + 0.5) * sx)
                    cy = int((row + 0.5) * sy)
                    draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                                 outline=(0, 220, 255, 220), width=lw)
                    draw.line([cx - r, cy, cx + r, cy], fill=(0, 220, 255, 220), width=lw)
                    draw.line([cx, cy - r, cx, cy + r], fill=(0, 220, 255, 220), width=lw)
                resized = Image.alpha_composite(base, circles).convert("RGB")
            photo = ImageTk.PhotoImage(resized)
            photo_ref[0] = photo        # GC-Schutz
            canvas.delete("all")
            canvas.create_image(0, 0, anchor="nw", image=photo)
            canvas.configure(scrollregion=(0, 0, new_w, new_h))
            update_spots_label()

        def apply_state(idx, ds, arr, title):
            """Setzt den Betrachter auf ein (neues) Bild und rendert es neu."""
            color = self._to_color_image(arr, channel)
            d_min, d_max = self._display_range(arr)
            key = ds["name"] if ds is not None else title

            state["idx"] = idx
            state["dataset"] = ds
            state["image_array"] = arr
            state["title_text"] = title
            state["color_img"] = color
            state["src_w"], state["src_h"] = color.size
            state["disp_min"], state["disp_max"] = d_min, d_max
            # Manuell hinzugefügte Punkte für diesen Datensatz (persistiert über
            # mehrfaches Öffnen/Navigieren hinweg, solange die App läuft).
            state["manual_coords"] = self._manual_spots.setdefault(key, [])
            state["spot_coords"] = (_ip.find_spots(arr, self.threshold_value)
                                    if self.show_spots_var.get() else None)

            win.title(title)
            title_label.config(text=title)
            minmax_label.config(text=f"min: {d_min:.1f}  |  max: {d_max:.1f}")

            for w in scale_holder.winfo_children():
                w.destroy()
            self._add_intensity_scale(scale_holder, d_min, d_max,
                                      height=400, channel=channel)

            cw, ch_px = canvas.winfo_width(), canvas.winfo_height()
            if cw > 1 and ch_px > 1 and state["src_w"] > 0 and state["src_h"] > 0:
                zoom_state[0] = min(cw / state["src_w"], ch_px / state["src_h"])
            redraw()

        def load_index(idx):
            ds = self.datasets[idx]
            arr = self._get_channel_image(ds, channel or "sum")
            title = f"{ds['name']} | {channel}" if channel else ds["name"]
            apply_state(idx, ds, arr, title)

        idx0 = None
        if dataset is not None:
            for i, d in enumerate(self.datasets):
                if d is dataset:
                    idx0 = i
                    break
        can_navigate = idx0 is not None and len(self.datasets) > 1

        def navigate(delta):
            if state.get("idx") is None:
                return
            load_index((state["idx"] + delta) % len(self.datasets))

        def on_canvas_click(event):
            """Fügt bei aktivierter Spots-Checkbox einen manuellen Punkt hinzu,
            oder entfernt einen bestehenden manuellen Punkt bei Klick darauf."""
            if not self.show_spots_var.get():
                return
            z = zoom_state[0]
            if z <= 0:
                return
            cx = canvas.canvasx(event.x)
            cy = canvas.canvasy(event.y)
            col = cx / z - 0.5
            row = cy / z - 0.5
            orig_h, orig_w = state["image_array"].shape
            if not (0 <= row < orig_h and 0 <= col < orig_w):
                return

            # Toleranz in Original-Pixeln, an aktuelle Zoomstufe angepasst,
            # damit Treffer-Radius auf dem Bildschirm etwa konstant bleibt.
            manual_coords = state["manual_coords"]
            click_radius_img = max(2.0, 6.0 / z)
            for pt in manual_coords:
                if np.hypot(pt[0] - row, pt[1] - col) < click_radius_img:
                    manual_coords.remove(pt)
                    redraw()
                    self._refresh_main_view()
                    return
            manual_coords.append((row, col))
            redraw()
            self._refresh_main_view()

        canvas.bind("<Button-1>", on_canvas_click)

        def on_mousewheel(event):
            # Bildpixel unter dem Cursor vor dem Zoom merken
            cx = canvas.canvasx(event.x)
            cy = canvas.canvasy(event.y)
            old_zoom = zoom_state[0]
            factor = 1.15 if event.delta > 0 else 1 / 1.15
            zoom_state[0] = max(0.05, min(30.0, old_zoom * factor))
            new_zoom = zoom_state[0]
            redraw()
            new_w = max(1, int(state["src_w"] * new_zoom))
            new_h = max(1, int(state["src_h"] * new_zoom))
            # Viewport verschieben, damit der Pixel an gleicher Stelle bleibt
            canvas.xview_moveto((cx * new_zoom / old_zoom - event.x) / new_w)
            canvas.yview_moveto((cy * new_zoom / old_zoom - event.y) / new_h)

        canvas.bind("<MouseWheel>", on_mousewheel)

        def save_single():
            output_dir = self._get_output_dir()
            include_name = self.save_names_var.get()
            lbl_h    = 14
            header_h = 20 if include_name else 0
            img_y    = max(header_h, lbl_h)
            color_img = state["color_img"]
            img_w, img_h = color_img.size
            cbar = _ip.make_colorbar_pil(state["disp_min"], state["disp_max"], img_h, channel)
            try:
                fnt = ImageFont.truetype("arial.ttf", 11)
            except Exception:
                fnt = ImageFont.load_default()
            out = Image.new("RGB", (img_w + cbar.width, img_h + img_y), (26, 26, 26))
            if include_name:
                ImageDraw.Draw(out).text((4, 4), state["title_text"],
                                          fill=(255, 255, 200), font=fnt)
            out.paste(color_img, (0, img_y))
            out.paste(cbar, (img_w, img_y - lbl_h))
            safe_name = "".join(
                c if c.isalnum() or c in "._- " else "_" for c in state["title_text"]
            ).strip()
            out.save(_ip.unique_output_path(
                output_dir, _ip.timestamped_filename(safe_name, "png")))
            orig_title = win.title()
            win.title(f"{state['title_text']} — gespeichert!")
            win.after(2000, lambda: win.title(orig_title))

        btn_frame = tk.Frame(win)
        btn_frame.pack(pady=(4, 10))
        if can_navigate:
            tk.Button(btn_frame, text="◀ Vorheriges", command=lambda: navigate(-1),
                      width=12).pack(side="left", padx=6)
        tk.Button(btn_frame, text="Bild speichern", command=save_single,
                  width=14).pack(side="left", padx=6)
        tk.Button(
            btn_frame, text="Metadaten", width=12,
            command=lambda: self._show_metadata(
                state["dataset"], state["image_array"], channel, state["title_text"]),
        ).pack(side="left", padx=6)
        tk.Button(btn_frame, text="Schließen", command=win.destroy,
                  width=12).pack(side="left", padx=6)
        if can_navigate:
            tk.Button(btn_frame, text="Nächstes ▶", command=lambda: navigate(1),
                      width=12).pack(side="left", padx=6)

        def init_view():
            # Bild beim ersten Render in den Canvas einpassen
            if idx0 is not None:
                load_index(idx0)
            else:
                apply_state(None, dataset, image_array, title_text)

        win.after(50, init_view)

    def _show_metadata(self, dataset, image_array, channel, title_text):
        """Zeigt Datei- und Bildkennwerte des aktuell im Zoom-Viewer offenen Bildes."""
        lines = [f"Titel:      {title_text}"]
        if dataset is not None:
            lines.append(f"Datei:      {dataset.get('name', '–')}")
            path = dataset.get("path")
            if path:
                lines.append(f"Pfad:       {path}")
                try:
                    st = os.stat(path)
                    lines.append(f"Dateigröße: {st.st_size / 1024:.1f} KB")
                    lines.append(
                        "Geändert:   "
                        f"{datetime.fromtimestamp(st.st_mtime):%Y-%m-%d %H:%M:%S}")
                except OSError:
                    pass

        h, w = image_array.shape
        lines += [
            "",
            f"Kanal:      {channel or 'sum'}",
            f"Bildgröße:  {w} x {h} px",
            f"Datentyp:   {image_array.dtype}",
            "",
            f"Min:        {float(np.nanmin(image_array)):.2f}",
            f"Max:        {float(np.nanmax(image_array)):.2f}",
            f"Mittelwert: {float(np.nanmean(image_array)):.2f}",
            f"Summe:      {float(np.nansum(image_array)):.2f}",
            "",
            f"Threshold:  {self.threshold_value:.2f}",
            f"Spots:      {self._count_spots(image_array, self.threshold_value, dataset=dataset)}",
        ]

        win = tk.Toplevel(self.root)
        win.title("Metadaten")
        win.resizable(False, False)

        text = tk.Text(win, width=46, height=len(lines) + 1,
                       font=("Courier", 10))
        text.pack(padx=14, pady=(14, 6))
        text.insert("1.0", "\n".join(lines))
        text.config(state="disabled")

        tk.Button(win, text="Schließen", command=win.destroy,
                  width=12).pack(pady=(0, 14))

    def _refresh_main_view(self):
        """Baut die aktuell sichtbare Grid-/Mehrkanalansicht neu auf, ohne
        Scroll-Position oder Dateiauswahl zurückzusetzen. Wird u. a. nach
        manuellen Punktänderungen in der Bildansicht aufgerufen, damit die
        Hauptansicht (z. B. Grid) sofort den aktuellen Stand zeigt."""
        if not self.datasets:
            return
        mode = self.display_mode_var.get()
        if mode == "grid_sum":
            self.show_grid_view()
        else:
            self.show_single_or_multi_view(mode)

    def update_display(self):
        """Leitet an die korrekte Ansicht basierend auf dem Anzeigemodus weiter."""
        if not self.datasets:
            return
        self._refresh_main_view()
        self.file_var.set(f"{self.current_index + 1:03d}")
        self.image_canvas.yview_moveto(0.0)
        self.image_canvas.xview_moveto(0.0)

    def show_grid_view(self):
        """Rendert alle Datensätze als Kachel-Grid (220×220 px) mit Sum-Kanal."""
        for widget in self.image_frame.winfo_children():
            widget.destroy()
        self._photo_refs = []
        self._tile_wrappers = {}
        cols = max(1, int(self.cols_var.get()))
        requested_rows = max(1, int(self.rows_var.get()))
        rows = max(requested_rows, math.ceil(len(self.datasets) / cols))

        for i, dataset in enumerate(self.datasets):
            row = i // cols
            col = i % cols
            img_array = self._get_channel_image(dataset, "sum")
            color_img = self._to_color_image(img_array)
            color_img = color_img.resize((220, 220), Image.Resampling.LANCZOS)
            if self.show_spots_var.get():
                color_img = self._draw_spots_overlay(color_img, img_array, radius=8,
                                                     dataset=dataset)
            photo = ImageTk.PhotoImage(color_img)
            self._photo_refs.append(photo)

            # Äußerer Wrapper: goldfarbener Rand wenn ausgewählt
            selected = i in self.selected_indices
            wrapper = tk.Frame(
                self.image_frame,
                bg="#c8a800" if selected else self.image_frame.cget("bg"),
                padx=3 if selected else 0,
                pady=3 if selected else 0,
            )
            wrapper.grid(row=row, column=col, padx=5, pady=5)
            self._tile_wrappers[i] = wrapper

            tile = tk.Frame(wrapper, relief="groove", bd=1)
            tile.pack(fill="both", expand=True)

            # Auswahlzeile: Checkbox links, Bildnummer rechts
            if i not in self._sel_vars:
                self._sel_vars[i] = tk.BooleanVar(value=selected)
            else:
                self._sel_vars[i].set(selected)

            sel_row = tk.Frame(tile)
            sel_row.pack(fill="x", padx=4, pady=(3, 0))
            tk.Checkbutton(
                sel_row, variable=self._sel_vars[i], font=("Arial", 8),
                command=lambda idx=i: self._toggle_selection(idx),
            ).pack(side="left")
            tk.Label(sel_row, text=f"{i + 1:03d}",
                     font=("Arial", 8, "bold")).pack(side="left")

            img_row = tk.Frame(tile)
            img_row.pack(padx=4, pady=(0, 4))

            label = tk.Label(img_row, image=photo)
            label.pack(side="left")
            label.bind(
                "<Button-1>",
                lambda _e, arr=img_array, name=dataset["name"], ds=dataset:
                    self.show_image_viewer(arr, name, channel="sum", dataset=ds),
            )

            disp_min, disp_max = self._display_range(img_array)
            self._add_intensity_scale(img_row, disp_min, disp_max, height=220)

            max_intensity = float(np.max(img_array)) if img_array.size else 0.0
            spots   = self._count_spots(img_array, self.threshold_value, dataset=dataset)
            caption = tk.Label(
                tile, text=f"max: {max_intensity:.2f} | spots: {spots}")
            caption.pack()
            caption.bind(
                "<Button-1>",
                lambda _e, arr=img_array, name=dataset["name"], ds=dataset:
                    self.show_image_viewer(arr, name, channel="sum", dataset=ds),
            )

        # Leer-Zellen auffüllen, damit das Grid gleichmäßig bleibt
        for i in range(len(self.datasets), rows * cols):
            filler = tk.Frame(self.image_frame, width=220, height=250)
            filler.grid(row=i // cols, column=i % cols, padx=5, pady=5)

        self.details_label.config(
            text=(f"Grid {rows} x {cols} (sum) for {len(self.datasets)} images "
                  f"| threshold={self.threshold_value:.2f}")
        )

    def show_single_or_multi_view(self, mode):
        """Rendert einen oder mehrere Kanäle des aktuellen Bildes (420×420 px)."""
        for widget in self.image_frame.winfo_children():
            widget.destroy()
        self._photo_refs = []
        dataset = self.datasets[self.current_index]

        if mode == "all":
            channels = ["detector0", "detector1", "sum"]
        elif mode == "det0_det1":
            channels = ["detector0", "detector1"]
        else:
            channels = [mode]

        panel = tk.Frame(self.image_frame)
        panel.pack(fill="both", expand=True)

        details_parts = []
        for channel in channels:
            image_array = self._get_channel_image(dataset, channel)
            color_img   = self._to_color_image(image_array, channel)
            color_img   = color_img.resize((420, 420), Image.Resampling.LANCZOS)
            if self.show_spots_var.get():
                color_img = self._draw_spots_overlay(color_img, image_array, radius=12,
                                                     dataset=dataset)
            photo = ImageTk.PhotoImage(color_img)
            self._photo_refs.append(photo)

            box = tk.Frame(panel, relief="groove", bd=1)
            box.pack(side="left", padx=8, pady=8)

            tk.Label(box, text=channel, font=("Arial", 11, "bold")).pack(pady=(4, 2))

            img_row = tk.Frame(box)
            img_row.pack(padx=4, pady=4)

            label = tk.Label(img_row, image=photo)
            label.pack(side="left")
            label.bind(
                "<Button-1>",
                lambda _e, arr=image_array,
                       t=f"{dataset['name']} | {channel}",
                       ch=channel, ds=dataset:
                    self.show_image_viewer(arr, t, channel=ch, dataset=ds),
            )

            disp_min, disp_max = self._display_range(image_array)
            self._add_intensity_scale(img_row, disp_min, disp_max,
                                       height=220, channel=channel)

            max_intensity = float(np.max(image_array)) if image_array.size else 0.0
            spots = self._count_spots(image_array, self.threshold_value, dataset=dataset)
            tk.Label(
                box,
                text=(f"max={max_intensity:.2f} | spots={spots} | "
                      f"thr={self.threshold_value:.2f} "
                      f"(>= {self.threshold_value * max_intensity:.2f})"),
            ).pack(pady=(2, 6))

            details_parts.append(
                f"{channel}: shape={image_array.shape}, max={max_intensity:.2f}")

        self.details_label.config(
            text=" | ".join(details_parts) + f" | file={dataset['name']}")

    # ── Navigation ────────────────────────────────────────────────────────────

    def prev_image(self):
        if not self.datasets:
            return
        self.current_index = max(0, self.current_index - 1)
        self.update_display()

    def next_image(self):
        if not self.datasets:
            return
        self.current_index = min(len(self.datasets) - 1, self.current_index + 1)
        self.update_display()

    def on_file_select(self, _event=None):
        """Springt zum Bild, das in der Datei-Combobox ausgewählt wurde."""
        value = self.file_var.get()
        if value and value.isdigit():
            idx = int(value) - 1
            if 0 <= idx < len(self.datasets):
                self.current_index = idx
                self.update_display()

    # ── Export ────────────────────────────────────────────────────────────────

    def _get_output_dir(self):
        """Gibt das output/-Verzeichnis neben dem Skript zurück (erzeugt es bei Bedarf)."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _open_output_folder(self):
        """Öffnet den output/-Ordner im Datei-Explorer (erzeugt ihn bei Bedarf)."""
        output_dir = self._get_output_dir()
        if sys.platform == "win32":
            os.startfile(output_dir)
        elif sys.platform == "darwin":
            subprocess.run(["open", output_dir])
        else:
            subprocess.run(["xdg-open", output_dir])

    @staticmethod
    def _fit_label_font(text, avail_w, measure_draw, max_size=13, min_size=7):
        """Größte Schriftgröße (zwischen min_size und max_size), bei der
        *text* in avail_w Pixel Breite passt; kürzt mit "…", falls *text*
        selbst bei min_size noch zu breit für avail_w ist."""
        def width(s, f):
            try:
                return measure_draw.textlength(s, font=f)
            except AttributeError:
                return f.getsize(s)[0]

        font = ImageFont.load_default()
        for size in range(max_size, min_size - 1, -1):
            try:
                f = ImageFont.truetype("arial.ttf", size)
            except Exception:
                break
            font = f
            if width(text, f) <= avail_w:
                return font, text

        truncated = text
        while len(truncated) > 1 and width(truncated + "…", font) > avail_w:
            truncated = truncated[:-1]
        if truncated != text:
            truncated += "…"
        return font, truncated

    def save_images_png(self):
        """Exportiert Einzel-PNGs oder ein Grid-Übersichtsbild (je nach Checkbox)."""
        if not self.datasets:
            self.status_label.config(text="Keine Bilder zum Speichern.")
            return
        output_dir = self._get_output_dir()
        if self.save_raster_var.get():
            self._save_grid_png(output_dir)
            self.status_label.config(text="Rasterbild gespeichert in output/")
        else:
            include_name = self.save_names_var.get()
            lbl_h    = 14   # Höhe des Cbar-Max-Labels (muss mit make_colorbar_pil übereinstimmen)
            header_h = 20 if include_name else 0
            img_y    = max(header_h, lbl_h)  # Bildstart: mindestens lbl_h für Cbar-Label
            OUT_SIZE = 640   # Zielgröße der exportierten Einzel-PNGs (Datei = OUT_SIZE x OUT_SIZE)
            cbar_w   = 40    # Breite der Farbleiste (muss mit make_colorbar_pil übereinstimmen)
            img_w, img_h = OUT_SIZE - cbar_w, OUT_SIZE - img_y
            active = self._active_datasets()
            _measure_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
            for dataset in active:
                img_array = self._get_channel_image(dataset, "sum")
                color_img = self._to_color_image(img_array).resize(
                    (img_w, img_h), Image.Resampling.LANCZOS)
                disp_min, disp_max = self._display_range(img_array)
                cbar = _ip.make_colorbar_pil(disp_min, disp_max, img_h)

                out = Image.new("RGB", (OUT_SIZE, OUT_SIZE), (26, 26, 26))
                if include_name:
                    fnt_title, label = self._fit_label_font(
                        dataset["name"], img_w - 8, _measure_draw)
                    ImageDraw.Draw(out).text((4, 4), label,
                                             fill=(255, 255, 200), font=fnt_title)
                out.paste(color_img, (0, img_y))
                out.paste(cbar, (img_w, img_y - lbl_h))
                name = os.path.splitext(dataset["name"])[0]
                out.save(_ip.unique_output_path(
                    output_dir, _ip.timestamped_filename(name, "png")))
            self.status_label.config(
                text=f"{len(active)} Einzelbild(er) (640x640) gespeichert in output/")

    def _save_grid_png(self, output_dir):
        """Erstellt ein einzelnes Grid-Übersichts-PNG aller aktiven Bilder.

        Die Anordnung wird automatisch möglichst quadratisch gewählt (statt die
        manuellen Cols/Rows-Spinboxen zu nutzen), damit die Bildfläche exakt zur
        Anzahl der Bilder passt und kein leerer Leerraum entsteht.
        """
        active = self._active_datasets()
        n = len(active)
        cols = max(1, math.ceil(math.sqrt(n)))
        rows = math.ceil(n / cols)

        include_name = self.save_names_var.get()
        use_filenames = self.grid_filenames_var.get()
        thumb_w, thumb_h = 220, 220
        label_h = 22 if include_name else 0
        cbar_w  = 40    # Breite der Farbleiste inkl. Platz für Labels oben/unten
        lbl_h   = 14    # muss mit make_colorbar_pil übereinstimmen
        pad     = 14

        grid_w = cols * (thumb_w + cbar_w + pad) + pad
        grid_h = rows * (thumb_h + label_h + pad) + pad

        grid_img = Image.new("RGB", (grid_w, grid_h), color=(26, 26, 26))
        draw = ImageDraw.Draw(grid_img)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            font = ImageFont.load_default()

        for i, dataset in enumerate(active):
            row = i // cols
            col = i % cols
            x = pad + col * (thumb_w + cbar_w + pad)
            y = pad + row * (thumb_h + label_h + pad)

            if include_name:
                if use_filenames:
                    label_font, label_text = self._fit_label_font(
                        dataset["name"], thumb_w - 4, draw, max_size=14, min_size=7)
                else:
                    label_font, label_text = font, f"{i + 1:03d}"
                draw.text((x, y), label_text, fill=(255, 255, 200), font=label_font)
            img_array = self._get_channel_image(dataset, "sum")
            thumb = self._to_color_image(img_array).resize(
                (thumb_w, thumb_h), Image.Resampling.LANCZOS)
            grid_img.paste(thumb, (x, y + label_h))

            disp_min, disp_max = self._display_range(img_array)
            cbar = _ip.make_colorbar_pil(disp_min, disp_max, thumb_h)
            grid_img.paste(cbar, (x + thumb_w, y + label_h - lbl_h))

        grid_img.save(_ip.unique_output_path(
            output_dir, _ip.timestamped_filename("grid_overview", "png")))

    # ── Analyse-Dialoge ───────────────────────────────────────────────────────

    def _new_tab(self, title):
        """Erstellt einen neuen, schließbaren Tab im Haupt-Notebook.

        Parameters
        ----------
        title : str  Beschriftung des neuen Tabs.

        Returns
        -------
        (tk.Frame, callable)
            Das neue Tab-Frame (als Elternwidget für den Dialoginhalt) sowie
            eine close_callback-Funktion, die dem Dialog als on_close
            übergeben werden sollte — sie entfernt den Tab wieder aus dem
            Notebook, wenn der Dialog seinen "Schließen"-Button auslöst.
        """
        frame = tk.Frame(self.notebook)
        self.notebook.add(frame, text=title)
        self.notebook.select(frame)

        def close_tab():
            try:
                self.notebook.forget(frame)
            except tk.TclError:
                pass
            frame.destroy()

        return frame, close_tab

    def _build_tool_tabs(self):
        """Erstellt die festen Analyse-Tabs neben "Bildanalyse" — alle direkt
        beim Start vorhanden, kein manuelles Öffnen über ein Menü mehr nötig.

        Analyse-Kasprzak und Analyse-Krüger werden über _refresh_dataset_tabs
        erzeugt, das auch nach jedem Neuladen von Bildern erneut aufgerufen
        wird (siehe dort).
        """
        self._refresh_dataset_tabs()

    def _refresh_dataset_tabs(self):
        """Baut die Methodenauswahl-Tabs (Analyse-Kasprzak, Analyse-Krüger)
        neu auf, damit sie den aktuellen Datenstand zeigen.

        Wird einmal beim Start (aus _build_tool_tabs) und danach jedes Mal
        aufgerufen, wenn load_images() neue Bilder geladen hat.
        """
        current = self.notebook.select()

        self._rebuild_kasprzak_tab()
        self._rebuild_krueger_tab()

        # Falls vor dem Neuaufbau ein unbeteiligter Tab (z. B. "Bildanalyse")
        # aktiv war, den Fokus dort belassen statt auf den zuletzt neu
        # gebauten Tab zu springen.
        if current and current in self.notebook.tabs():
            self.notebook.select(current)

    def _rebuild_kasprzak_tab(self):
        """Baut den Tab "Analyse-Kasprzak" neu auf (löscht einen ggf.
        bestehenden vorherigen Tab zuerst).

        Fasst die früher eigenständigen Tabs Timetrace/FCS, Schichtdicke
        (Zählen) und Schichtdicke (ICS) über eine Methodenauswahl zusammen —
        Aufbau analog zu _rebuild_krueger_tab. Die Auswahl bleibt über ein
        erneutes Laden von Bildern hinweg erhalten; der komplette Tab baut
        sich dabei aber neu auf (Timetrace-Zwischenstand geht dabei wie
        gehabt verloren, da Timetrace ohnehin unabhängig von den geladenen
        Bilddatensätzen arbeitet)."""
        old_frame = getattr(self, "_kasprzak_frame", None)
        previous_method = getattr(self, "_kasprzak_method_var", None)
        selected_label = previous_method.get() if previous_method is not None else KASPRZAK_METHODS[0][0]
        if old_frame is not None:
            try:
                self.notebook.forget(old_frame)
            except tk.TclError:
                pass
            old_frame.destroy()

        frame, close_tab = self._new_tab("Analyse-Kasprzak")
        self._kasprzak_frame = frame
        self._kasprzak_close_tab = close_tab

        selector_frame = tk.Frame(frame)
        selector_frame.pack(fill="x", padx=12, pady=(10, 0))
        tk.Label(selector_frame, text="Methode:", font=("Arial", 10, "bold")).pack(
            side="left", padx=(0, 6))
        self._kasprzak_method_var = tk.StringVar(value=selected_label)
        method_combo = ttk.Combobox(
            selector_frame, textvariable=self._kasprzak_method_var, state="readonly",
            width=32, values=[label for label, _ in KASPRZAK_METHODS])
        method_combo.pack(side="left")
        method_combo.bind("<<ComboboxSelected>>", lambda _e: self._build_kasprzak_method_content())

        self._kasprzak_content_frame = tk.Frame(frame)
        self._kasprzak_content_frame.pack(fill="both", expand=True)

        self._build_kasprzak_method_content()

    def _build_kasprzak_method_content(self):
        """Baut den Inhalt des Analyse-Kasprzak-Tabs für die aktuell in der
        Methodenauswahl gewählte Methode neu auf."""
        for widget in self._kasprzak_content_frame.winfo_children():
            widget.destroy()

        method_key = dict(KASPRZAK_METHODS).get(
            self._kasprzak_method_var.get(), KASPRZAK_METHODS[0][1])
        close_cmd = self._kasprzak_close_tab
        content = self._kasprzak_content_frame

        if method_key == "timetrace":
            TimeraceDialog(content, self._get_output_dir, on_close=close_cmd)
        elif method_key == "schichtdicke":
            active = self._active_datasets()
            active_idx = min(self.current_index, len(active) - 1) if active else 0
            SchichtdickeDialog(
                content, active, active_idx,
                lambda: self.threshold_value, self._get_output_dir,
                lambda: self.bg_correction_var.get(),
                on_close=close_cmd,
            )
        elif method_key == "ics":
            ICSDialog(content, self._active_datasets(), self._get_output_dir,
                      lambda: self.bg_correction_var.get(), on_close=close_cmd)

    def _rebuild_krueger_tab(self):
        """Baut den Tab "Analyse-Krüger" neu auf (löscht einen ggf. bestehenden
        vorherigen Tab zuerst).

        Der Tab enthält oben eine Methodenauswahl (siehe KRUEGER_METHODS) und
        darunter den Dialog der jeweils gewählten Methode. Die Auswahl bleibt
        über ein erneutes Laden von Bildern hinweg erhalten (wie bei den
        anderen Schichtdicke-Tabs baut sich aber auch hier der komplette Tab
        bei jedem Neuladen neu auf, damit datensatzabhängige Methoden den
        aktuellen Stand zeigen — Simulation/Z-Stack-Zwischenstände gehen dabei
        mit verloren, das betrifft aber ohnehin nur die anderen beiden
        Methoden nicht)."""
        old_frame = getattr(self, "_krueger_frame", None)
        previous_method = getattr(self, "_krueger_method_var", None)
        selected_label = previous_method.get() if previous_method is not None else KRUEGER_METHODS[0][0]
        if old_frame is not None:
            try:
                self.notebook.forget(old_frame)
            except tk.TclError:
                pass
            old_frame.destroy()

        frame, close_tab = self._new_tab("Analyse-Krüger")
        self._krueger_frame = frame
        self._krueger_close_tab = close_tab

        selector_frame = tk.Frame(frame)
        selector_frame.pack(fill="x", padx=12, pady=(10, 0))
        tk.Label(selector_frame, text="Methode:", font=("Arial", 10, "bold")).pack(
            side="left", padx=(0, 6))
        self._krueger_method_var = tk.StringVar(value=selected_label)
        method_combo = ttk.Combobox(
            selector_frame, textvariable=self._krueger_method_var, state="readonly",
            width=32, values=[label for label, _ in KRUEGER_METHODS])
        method_combo.pack(side="left")
        method_combo.bind("<<ComboboxSelected>>", lambda _e: self._build_krueger_method_content())

        self._krueger_content_frame = tk.Frame(frame)
        self._krueger_content_frame.pack(fill="both", expand=True)

        self._build_krueger_method_content()

    def _build_krueger_method_content(self):
        """Baut den Inhalt des Analyse-Krüger-Tabs für die aktuell in der
        Methodenauswahl gewählte Methode neu auf."""
        for widget in self._krueger_content_frame.winfo_children():
            widget.destroy()

        method_key = dict(KRUEGER_METHODS).get(
            self._krueger_method_var.get(), KRUEGER_METHODS[0][1])
        close_cmd = self._krueger_close_tab
        content = self._krueger_content_frame

        if method_key == "spot_zaehler":
            SpotZaehlerDialog(content, self._active_datasets(), self._get_output_dir,
                              lambda: self.bg_correction_var.get(), on_close=close_cmd)
        elif method_key == "ics":
            KruegerICSDialog(content, self._active_datasets(), self._get_output_dir,
                             lambda: self.bg_correction_var.get(), on_close=close_cmd)
        elif method_key == "simulation":
            SpotSimulationDialog(content, self._get_output_dir, on_close=close_cmd)
        elif method_key == "zstack":
            ZStackDialog(content, self._get_output_dir, on_close=close_cmd)

    # ── App-Lifecycle ─────────────────────────────────────────────────────────

    def exit_app(self):
        """Beendet die Anwendung sauber.

        root.quit() allein kann den Interpreter in manchen Konfigurationen
        am Laufen lassen; destroy() + SystemExit stellt sicher, dass der
        Prozess wirklich terminiert.
        """
        try:
            self.root.quit()
            self.root.destroy()
        finally:
            raise SystemExit(0)

    # ── Info-Dialoge ──────────────────────────────────────────────────────────

    def show_about(self):
        """Öffnet das Info-Fenster mit Funktionsübersicht."""
        win = tk.Toplevel(self.root)
        win.title("Über Image Browser")
        win.grab_set()
        tk.Label(win, text="Image Browser & Analysis Tool",
                 font=("Arial", 14, "bold")).pack(pady=(18, 4), padx=28)
        tk.Label(win, text="Version 1.7.0",
                 font=("Arial", 10)).pack(pady=(0, 10))
        tk.Label(
            win,
            text=(
                "Lädt Bilddateien und Photon-Zeitreihendaten für Mikroskopie-Analysen.\n\n"
                "Unterstützte Bildformate: .img (scanbin_s4), PNG, JPG, TIF, BMP, WEBP\n"
                "Unterstützte Zeitreihendaten: HDF5 (.h5, .hdf5)\n\n"
                "Oberfläche:\n"
                "  • Alle Werkzeuge (Bildanalyse, Analyse-Kasprzak, Analyse-Krüger) "
                "sind\n"
                "    von Anfang an als Tabs im Hauptfenster vorhanden — kein "
                "Auswahlbildschirm\n"
                "    beim Start, kein separates Öffnen über ein Menü mehr nötig\n"
                "  • Analyse-Kasprzak und Analyse-Krüger bündeln ihre jeweiligen "
                "Methoden\n"
                "    über eine Methodenauswahl im Tab statt eigener Einzel-Tabs\n"
                "  • Tabs einzeln schließbar, aktualisieren sich automatisch nach "
                "jedem\n"
                "    Neuladen von Bildern\n\n"
                "Bildbetrachter:\n"
                "  • Grid-Ansicht mit konfigurierbaren Rows/Cols\n"
                "  • Einzelkanal-Ansicht (detector0, detector1, sum)\n"
                "  • Zoom-Viewer (Mausrad) mit Intensitätsskala und Dateiname\n"
                "  • Einzelbild speichern direkt aus dem Zoom-Viewer\n"
                "  • Threshold-Filter relativ zu Imax\n"
                "  • Hintergrundkorrektur (Perzentil-basiert, an/aus, "
                "gilt für Anzeige und Analyse)\n"
                "  • Spot-Zählung (lokale Maxima)\n"
                "  • Bilder speichern: Einzel-PNGs oder Grid-Übersicht\n"
                "  • Bilder laden / Pfad öffnen: über das Datei-Menü ODER über den "
                "„Datei“-\n"
                "    Abschnitt im rechten Seitenmenü\n\n"
                "Analyse:\n"
                "  • Analyse-Kasprzak (Methodenauswahl im Tab):\n"
                "      – Timetrace / FCS – FCS-Korrelationsanalyse:\n"
                "          · Laden von HDF5-Photon-Daten via photon_tools\n"
                "          · Logarithmisch gelagerte Korrelation mit pycorrelate\n"
                "          · Modi: ch0×ch0, ch1×ch1, Kreuzkorrelation ch0×ch1\n"
                "          · 3D-FCS-Fit (A₀, τ_D, S = ω_z/ω_xy) mit lmfit\n"
                "          · 2-Panel-Plot: G(τ) + Fit oben, Residuen unten\n"
                "      – Schichtdicke (Zählen):\n"
                "          · Lokaler Threshold-Regler mit Live-Vorschau\n"
                "          · Dickenberechnung aus Spot-Anzahl, Konzentration, "
                "Fläche\n"
                "          · Verteilungsplot mit Mittelwert, ±1σ-Band, "
                "Fehlerbalken (an/aus)\n"
                "          · CSV-Export der Ergebnisse\n"
                "      – Schichtdicke (ICS, streng):\n"
                "          · Image Correlation Spectroscopy (Petersen et al. "
                "1993)\n"
                "          · 2-D-Bild-ACF via FFT, mit Detrending gegen "
                "Beleuchtungsinhomogenität\n"
                "          · 2-D-Gauß-Fit (g0, w0, g_∞) mit "
                "Schrotrausch-Ausschluss\n"
                "          · Formale Fehlerfortpflanzung UND empirische "
                "Kachel-zu-Kachel-Streuung\n"
                "          · Laufzeit-Plausibilitätsprüfungen (PSF-Abweichung, "
                "Unterabtastung, u. a.)\n"
                "          · 4-Panel-Plot, Report- und Plot-Export\n"
                "          · Implementiert in ics_thickness.py (eigenständig "
                "testbar, auch als CLI nutzbar)\n\n"
                "  • Analyse-Krüger (Methodenauswahl im Tab, Willi_Analyse_Tool):\n"
                "      – Spot-Zähler (Methode A): robuste Hintergrundschätzung +\n"
                "        matched-filter-Glättung, 2D-Gauß-Fit pro Spot, automat.\n"
                "        Doppelspot-Aufteilung, interaktive visuelle Prüfung\n"
                "      – ICS-Auswertung (Methode B): Spotzahl aus der räumlichen\n"
                "        Autokorrelation (2D-Gauß-Fit frei/fixiert), wählbare\n"
                "        Untergrundschätzer, optionale cv-Korrektur\n"
                "      – Spot-Simulation: synthetische .img-Testdaten mit\n"
                "        bekannter Molekülzahl/-position (Ground Truth)\n"
                "      – Z-Stack (PSF): Bead-Lokalisation + 2D-Gauß-Fit je Ebene,\n"
                "        drift-korrigierter ROI-Stack, x-z/y-z-Schnitt, axiale\n"
                "        PSF-Breite (FWHM_z) aus dem Zentralpixel-Profil\n"
                "      – Je Methode CSV-/PNG-Export in den Ausgabeordner\n\n"
                "Benötigte Pakete: numpy, pillow, matplotlib, scipy, photon_tools,\n"
                "                  pycorrelate, lmfit"
            ),
            justify="left",
        ).pack(pady=(0, 16), padx=28)
        tk.Button(win, text="Schließen", command=win.destroy,
                  width=12).pack(pady=(0, 16))

    def show_author(self):
        """Öffnet das About-Fenster mit Autorenangaben."""
        win = tk.Toplevel(self.root)
        win.title("About")
        win.resizable(False, False)
        win.grab_set()

        w, h = 340, 300
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        win.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

        tk.Label(win, text="Image Browser & Analysis Tool",
                 font=("Arial", 13, "bold")).pack(pady=(24, 2))
        tk.Label(win, text="Version 1.7.0",
                 font=("Arial", 10), fg="#555555").pack(pady=(0, 18))

        tk.Frame(win, height=1, bg="#cccccc").pack(fill="x", padx=30)

        tk.Label(win, text="Autor", font=("Arial", 9, "bold"),
                 fg="#444444").pack(pady=(14, 2))
        tk.Label(win, text="Yannik Kasprzak",
                 font=("Arial", 11)).pack()
        tk.Label(win, text="Institut für Physik",
                 font=("Arial", 10), fg="#555555").pack()
        tk.Label(win, text="Universität zu Lübeck",
                 font=("Arial", 10), fg="#555555").pack()

        tk.Frame(win, height=1, bg="#cccccc").pack(fill="x", padx=30, pady=(16, 0))

        tk.Label(win, text="Analyse-Krüger – Analyse-Methoden", font=("Arial", 9, "bold"),
                 fg="#444444").pack(pady=(14, 2))
        tk.Label(win, text="Wilhelm Krüger",
                 font=("Arial", 11)).pack()

        tk.Button(win, text="Schließen", command=win.destroy,
                  width=12).pack(pady=(20, 16))
