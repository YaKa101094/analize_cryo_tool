"""
image_processing.py – Bildverarbeitungs-Hilfsfunktionen
=======================================================
Reine Berechnungs- und Rendering-Funktionen ohne tkinter-Abhängigkeit.
Alle Funktionen sind zustandslos und können direkt importiert werden.

Autor: Yannik Kasprzak, Institut für Physik, Universität zu Lübeck
"""

import os
import struct
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

# ── Optionale Importe ─────────────────────────────────────────────────────────

try:
    import matplotlib.cm as _mpl_cm
    _matplotlib_ok = True
except Exception as _e:
    print(f"[matplotlib import failed] {_e}")
    _mpl_cm = None
    _matplotlib_ok = False

try:
    # label()          – Zusammenhängende Komponenten (Spot-Detektion)
    # center_of_mass() – Schwerpunktsberechnung für Spot-Zentroide
    from scipy.ndimage import label as _sc_label, center_of_mass as _sc_center_of_mass
    _scipy_ok = True
except Exception as _e:
    print(f"[scipy import failed] {_e}")
    _scipy_ok = False


# ── Ausgabe-Hilfsfunktion ─────────────────────────────────────────────────────

def unique_output_path(directory: str, filename: str) -> str:
    """Gibt einen Ausgabepfad zurück, der bestehende Dateien nicht überschreibt.

    Existiert *filename* bereits in *directory*, wird ein Zähler-Suffix
    "_1", "_2", … vor die Dateiendung eingefügt, bis ein freier Name
    gefunden ist. So bleiben bei wiederholtem Speichern (z. B. nach
    Parameteränderung) alle bisherigen Exporte erhalten.

    Parameters
    ----------
    directory : str  Zielverzeichnis
    filename  : str  Gewünschter Dateiname (inkl. Endung)

    Returns
    -------
    str  Vollständiger Pfad, der zum Zeitpunkt des Aufrufs noch nicht existiert.
    """
    base, ext = os.path.splitext(filename)
    path = os.path.join(directory, filename)
    counter = 1
    while os.path.exists(path):
        path = os.path.join(directory, f"{base}_{counter}{ext}")
        counter += 1
    return path


def timestamped_filename(base_name: str, ext: str) -> str:
    """Baut einen Dateinamen mit Datums-/Uhrzeit-Suffix.

    Format: "<base_name>_Tag_Monat_Jahr_Uhrzeit.<ext>", z. B.
    "schichtdicke_plot_07_09_2026_14-30-22.png".

    Parameters
    ----------
    base_name : str  Basisname ohne Endung
    ext       : str  Dateiendung, mit oder ohne führenden Punkt

    Returns
    -------
    str  Dateiname inkl. Zeitstempel und Endung.
    """
    ts = datetime.now().strftime("%d_%m_%Y_%H-%M-%S")
    ext = ext.lstrip(".")
    return f"{base_name}_{ts}.{ext}"


# ── Kanal-Hilfsfunktion ───────────────────────────────────────────────────────

def get_channel_array(dataset: dict, channel: str,
                      bg_correction: bool = False,
                      bg_percentile: float = 5.0) -> np.ndarray:
    """Gibt das (optional hintergrundkorrigierte) numpy-Array für den Kanal zurück.

    Parameters
    ----------
    dataset       : dict  Datensatz mit Schlüsseln "detector0", "detector1", "sum"
    channel       : str   "detector0", "detector1" oder beliebiger anderer Wert → "sum"
    bg_correction : bool  Perzentilbasierte Hintergrundkorrektur an/aus
    bg_percentile : float Perzentil für die Hintergrundschätzung (Default 5.0)
    """
    if channel == "detector0":
        arr = dataset["detector0"]
    elif channel == "detector1":
        arr = dataset["detector1"]
    else:
        arr = dataset["sum"]
    return apply_background_correction(arr, bg_correction, bg_percentile)


def apply_background_correction(img: np.ndarray, enabled: bool,
                                percentile: float = 5.0) -> np.ndarray:
    """Subtrahiert ein perzentilbasiertes Hintergrundniveau von *img*.

    Algorithmus
    -----------
    bg        = np.percentile(img, percentile)
    corrected = np.clip(img - bg, 0, None)

    Wird zustandslos auf das jeweils angeforderte Kanal-Array angewendet
    (detector0 / detector1 / sum bekommen jeweils ihr eigenes Perzentil, da
    sie unterschiedliche Dunkel-Offsets haben können). Mutiert *img* nicht.

    Parameters
    ----------
    img        : np.ndarray  2-D float-Array (ein einzelner Kanal)
    enabled    : bool        Wenn False, wird img unverändert zurückgegeben
    percentile : float       Perzentil [0, 100] zur Hintergrundschätzung

    Returns
    -------
    np.ndarray  Hintergrundkorrigiertes Array, oder *img* selbst wenn enabled=False.
    """
    if not enabled or img.size == 0:
        return img
    bg = np.percentile(img, percentile)
    return np.clip(img - bg, 0, None)


# ── Intensitätsbereich ────────────────────────────────────────────────────────

def display_range(image_array: np.ndarray, threshold_value: float):
    """Gibt (min, max) des sichtbaren Intensitätsbereichs nach Threshold-Maskierung zurück.

    Parameters
    ----------
    image_array     : 2-D float-Array
    threshold_value : relativer Schwellwert [0, 1]; 0 = kein Filter

    Returns
    -------
    (disp_min, disp_max) : tuple[float, float]
    """
    if image_array.size == 0:
        return 0.0, 0.0
    max_val = float(np.max(image_array))
    if max_val <= 0.0:
        return 0.0, 0.0
    threshold = threshold_value * max_val
    masked = (
        np.where(image_array >= threshold, image_array, 0.0)
        if threshold_value > 0.0
        else image_array
    )
    disp_max = float(np.max(masked)) if masked.size else 0.0
    disp_min = float(np.min(masked)) if masked.size else 0.0
    return disp_min, disp_max


# ── Falschfarb-Rendering ──────────────────────────────────────────────────────

def to_color_image(image_array: np.ndarray, channel: str = "",
                   threshold_value: float = 0.0) -> Image.Image:
    """Wandelt ein float32-Array in ein Falschfarb-PIL-Image um.

    Colormaps
    ---------
      detector1  →  matplotlib "cool"  (cyan → magenta)
      alle anderen →  ImageOps.colorize  schwarz → rot → gelb

    Normierung
    ----------
    Pixel unterhalb von threshold_value × I_max werden auf 0 gesetzt,
    bevor das Array auf [0, 255] normiert wird.

    Parameters
    ----------
    image_array     : np.ndarray  2-D float-Intensitätsbild
    channel         : str         "detector1" wählt die cool-Colormap
    threshold_value : float       relativer Threshold [0, 1]

    Returns
    -------
    PIL.Image  RGB-Bild gleicher Größe wie image_array.
    """
    max_val = float(np.max(image_array)) if image_array.size else 0.0
    if max_val <= 0.0:
        norm_u8 = np.zeros_like(image_array, dtype=np.uint8)
    else:
        threshold = threshold_value * max_val
        masked = (
            np.where(image_array >= threshold, image_array, 0.0)
            if threshold_value > 0.0
            else image_array
        )
        display_max = float(np.max(masked)) if masked.size else 0.0
        if display_max <= 0.0:
            norm_u8 = np.zeros_like(image_array, dtype=np.uint8)
        else:
            norm_u8 = np.clip((masked / display_max) * 255.0, 0, 255).astype(np.uint8)

    gray_img = Image.fromarray(norm_u8, mode="L")
    if channel == "detector1" and _matplotlib_ok:
        rgba = _mpl_cm.cool(norm_u8.astype(np.float32) / 255.0)
        return Image.fromarray((rgba[:, :, :3] * 255).astype(np.uint8), mode="RGB")
    if channel == "detector1":
        return ImageOps.colorize(gray_img, black="#000000", mid="#00aa00", white="#ccffcc")
    return ImageOps.colorize(gray_img, black="#000000", mid="#cc3300", white="#ffffcc")


def make_colorbar_pil(min_val: float, max_val: float, height: int,
                      channel: str = "") -> Image.Image:
    """Erstellt eine beschriftete, vertikale Falschfarb-Intensitätsskala als PIL-Image.

    Layout (von oben nach unten):
      max-Label  (lbl_h px) – oben über dem Balken
      Farbgradient (height px) – oben hell, unten dunkel
      min-Label  (lbl_h px) – unten unter dem Balken
    Gesamtgröße: bar_w × (height + 2 × lbl_h).

    Parameters
    ----------
    min_val, max_val : float  Intensitätswerte für die Beschriftung
    height           : int    Höhe des Gradientstreifens in Pixeln
    channel          : str    "detector1" → cool-Colormap, sonst rot-gelb

    Returns
    -------
    PIL.Image  RGB-Bild der Farbskala.
    """
    lbl_h, bar_w = 14, 40   # bar_w breit genug für Labels wie "9999.9" bei 10 pt
    try:
        fnt = ImageFont.truetype("arial.ttf", 10)
    except Exception:
        fnt = ImageFont.load_default()
    img = Image.new("RGB", (bar_w, height + 2 * lbl_h), (26, 26, 26))
    draw = ImageDraw.Draw(img)
    grad = np.linspace(255, 0, height, dtype=np.uint8).reshape(height, 1)
    grad_pil = Image.fromarray(grad, mode="L").resize((bar_w, height), Image.Resampling.NEAREST)
    if channel == "detector1" and _matplotlib_ok:
        grad_1d = np.linspace(1.0, 0.0, height)
        rgba = _mpl_cm.cool(grad_1d)
        rgb_bar = np.tile((rgba[:, :3] * 255).astype(np.uint8)[:, np.newaxis, :], (1, bar_w, 1))
        col = Image.fromarray(rgb_bar, mode="RGB")
    elif channel == "detector1":
        col = ImageOps.colorize(grad_pil, black="#000000", mid="#00aa00", white="#ccffcc")
    else:
        col = ImageOps.colorize(grad_pil, black="#000000", mid="#cc3300", white="#ffffcc")
    img.paste(col, (0, lbl_h))
    # Labels über und unter dem Balken – gleiche x-Position wie der Balken
    draw.text((2, 1), f"{max_val:.1f}", fill=(220, 220, 200), font=fnt)
    draw.text((2, lbl_h + height + 1), f"{min_val:.1f}", fill=(220, 220, 200), font=fnt)
    return img


# ── Spot-Zählung ──────────────────────────────────────────────────────────────

def find_spots(img: np.ndarray, threshold_fraction: float) -> np.ndarray:
    """Gibt die (row, col)-Koordinaten aller detektierten Spots zurück.

    Algorithmus
    -----------
    1. Für jeden Pixel wird das Maximum seiner 8×8-Moore-Nachbarschaft bestimmt
       (win_max; asymmetrisches Padding: 3 Pixel vor, 4 nach dem Zielpixel).
    2. Kandidat ist ein Pixel, wenn es selbst dieses Maximum ist
       (img == win_max) und mindestens threshold_fraction × max(img) beträgt.
    3. Direkt benachbarte Kandidaten mit identischem Wert (Plateaus – z. B.
       zwei gleich helle Nachbarpixel desselben Spots) werden über
       scipy.ndimage.label zu einem einzigen Spot an ihrem Schwerpunkt
       zusammengefasst, statt verworfen zu werden.

    Parameters
    ----------
    img               : 2-D float-Array
    threshold_fraction: relativer Schwellwert [0, 1]; 0 = kein Filter

    Returns
    -------
    np.ndarray  Shape (N, 2) mit (row, col) je Spot; leer wenn keine gefunden.
    """
    if img.ndim != 2 or img.size == 0:
        return np.empty((0, 2), dtype=np.intp)
    max_val = float(np.nanmax(img))
    if max_val <= 0.0:
        return np.empty((0, 2), dtype=np.intp)
    thresh_val = threshold_fraction * max_val

    # Asymmetrisches Padding für 8×8-Fenster (Moore-Nachbarschaft):
    # 3 Pixel vor, 4 Pixel nach dem Zielpixel → Ausgabegröße = (H, W, 8, 8).
    # mode="edge" würde den Randwert in sein eigenes Fenster duplizieren und
    # damit echte Randmaxima verfälschen -> stattdessen mit -inf auffüllen,
    # damit Padding-Werte nie mit echten Nachbarn konkurrieren.
    padded = np.pad(img.astype(float), ((3, 4), (3, 4)),
                    mode="constant", constant_values=-np.inf)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (8, 8))
    win_max = windows.max(axis=(-2, -1))
    candidates = (img == win_max) & (img >= thresh_val)
    if not np.any(candidates):
        return np.empty((0, 2), dtype=np.intp)

    if not _scipy_ok:
        # Ohne scipy keine Plateau-Zusammenfassung möglich: strikte
        # Eindeutigkeit wie zuvor (benachbarte, gleich helle Pixel eines
        # Plateaus werden dabei verworfen statt zusammengefasst).
        win_second = np.partition(
            windows.reshape(*windows.shape[:2], -1), -2, axis=-1
        )[..., -2]
        local_max = candidates & (img > win_second)
        return np.column_stack(np.where(local_max))

    # Zusammenhängende Kandidaten mit identischem Maximalwert (Plateaus) zu
    # einem einzigen Spot an ihrem Schwerpunkt zusammenfassen.
    labeled, n_labels = _sc_label(candidates)
    if n_labels == 0:
        return np.empty((0, 2), dtype=np.intp)
    centroids = _sc_center_of_mass(candidates, labeled, range(1, n_labels + 1))
    return np.round(np.array(centroids)).astype(np.intp)


def count_spots(img: np.ndarray, threshold_fraction: float) -> int:
    """Zählt lokale Maxima als Spots — delegiert an find_spots."""
    return len(find_spots(img, threshold_fraction))


# ── .img-Dateien einzeln laden (außerhalb der Hauptdatensatzliste) ────────────

@dataclass
class ImageMeta:
    """Physical scan metadata (µm, seconds)."""
    n_pixels_x: int
    n_pixels_y: int
    range_x_um: float
    range_y_um: float
    offset_x_um: float = 0.0
    offset_y_um: float = 0.0
    pixel_dwell_time_s: float | None = None

    @property
    def pixel_size_x_nm(self) -> float:
        return self.range_x_um / self.n_pixels_x * 1000.0

    @property
    def pixel_size_y_nm(self) -> float:
        return self.range_y_um / self.n_pixels_y * 1000.0


def _scanbin_meta_from_header(header: bytes) -> ImageMeta | None:
    """Liest die Scan-Metadaten aus den ersten 56 Bytes einer Messdatei
    (Layout siehe parse_scanbin_img). Gibt None zurück, wenn der Header nicht
    dem Messformat entspricht (z. B. ältere simulierte Dateien ohne Metadaten)."""
    if len(header) < 56 or header[24:32] != header[48:56]:
        return None
    dwell_s, offset_x_m, offset_y_m = struct.unpack_from(">ddd", header, 0)
    n_x, n_y = struct.unpack_from(">II", header, 24)
    range_x_m, range_y_m = struct.unpack_from(">dd", header, 32)
    if n_x <= 0 or n_y <= 0 or not (range_x_m > 0 and range_y_m > 0):
        return None
    return ImageMeta(
        n_pixels_x=n_x, n_pixels_y=n_y,
        range_x_um=range_x_m * 1e6, range_y_um=range_y_m * 1e6,
        offset_x_um=offset_x_m * 1e6, offset_y_um=offset_y_m * 1e6,
        pixel_dwell_time_s=dwell_s if dwell_s > 0 else None,
    )


def read_scanbin_meta(path: str) -> ImageMeta | None:
    """Scan-Metadaten einer .img-Datei (None, wenn nicht lesbar oder nicht vorhanden)."""
    try:
        with open(path, "rb") as fobj:
            return _scanbin_meta_from_header(fobj.read(56))
    except OSError:
        return None


def parse_scanbin_img(path: str) -> dict:
    """Liest eine scanbin_s4-.img-Datei ohne photon_tools.

    Messdateien (big-endian):
      Bytes  0–55 : Header
        Offset  0 : Verweilzeit pro Pixel   (float64)
        Offset 24 : Breite, Höhe            (2× uint32)
        Offset 32 : Scanbereich X, Y [m]    (2× float64)
        Offset 48 : Breite, Höhe (wiederholt)
      ab Byte 56  : Kanäle als uint32-Bilder (Höhe × Breite), vor jedem
                    weiteren Kanal ein 8-Byte-Marker (Breite, Höhe).
                    Kanal 0 = Summe (transponiert gespeichert),
                    Kanal 1 = detector0, Kanal 2 = detector1, Rest leer.

    Älteres Format (von analysis_krueger_simulation.write_scanbin_img erzeugt):
      88-Byte-Header mit Breite/Höhe bei Offset 24/28, danach uint16-Frames,
      abwechselnd detector0 / detector1; alle Frames je Detektor werden summiert.

    Der zurückgegebene Dict enthält zusätzlich "meta" (ImageMeta oder None).

    Raises ValueError, wenn die Datei keinem der beiden Formate entspricht.
    """
    with open(path, "rb") as fobj:
        data = fobj.read()
    if len(data) <= 56:
        raise ValueError(".img-Datei ist zu klein")

    width = int.from_bytes(data[24:28], byteorder="big", signed=False)
    height = int.from_bytes(data[28:32], byteorder="big", signed=False)
    if width <= 0 or height <= 0:
        raise ValueError("Ungültige Bilddimensionen im .img-Header")
    n_pixels = width * height
    name = os.path.basename(path)

    # Messformat: 56 + k·(4·n) + (k−1)·8 Bytes, Marker zwischen den Kanälen
    block = 4 * n_pixels + 8
    n_channels, rest = divmod(len(data) - 56 + 8, block)
    marker = data[24:32]
    if (rest == 0 and n_channels >= 1
            and all(data[56 + k * block - 8:56 + k * block] == marker
                    for k in range(1, n_channels))):
        channels = [np.frombuffer(data, dtype=">u4", count=n_pixels, offset=56 + k * block)
                    .reshape(height, width).astype(np.float32)
                    for k in range(n_channels)]
        if n_channels >= 3:
            detector0, detector1 = channels[1], channels[2]
        else:
            detector0 = channels[0]
            detector1 = channels[1] if n_channels == 2 else np.zeros_like(detector0)
        return {"path": path, "name": name,
                "detector0": detector0, "detector1": detector1,
                "sum": detector0 + detector1,
                "meta": _scanbin_meta_from_header(data[:56])}

    # Älteres uint16-Format
    raw_u16 = np.frombuffer(data[88:], dtype=">u2")
    total_values = (len(raw_u16) // n_pixels) * n_pixels
    if total_values == 0:
        raise ValueError("Keine vollständigen Frames in .img-Datei gefunden")
    frames = raw_u16[:total_values].reshape(-1, height, width).astype(np.float32)
    det0_frames = frames[0::2]
    det1_frames = frames[1::2]
    detector0 = det0_frames.sum(axis=0) if det0_frames.size else np.zeros((height, width), dtype=np.float32)
    detector1 = det1_frames.sum(axis=0) if det1_frames.size else np.zeros((height, width), dtype=np.float32)
    return {"path": path, "name": name,
            "detector0": detector0, "detector1": detector1,
            "sum": detector0 + detector1, "meta": None}


def build_mosaic(datasets: list, channel: str = "sum", grid_um: float = 200.0,
                 include_full_scans: bool = True,
                 bg_correction: bool = False) -> dict | None:
    """Setzt Bilder anhand ihrer Scan-Metadaten zu einem grid_um × grid_um-Mosaik zusammen.

    Koordinaten: Der Scan-Offset eines Bildes ist dessen linke obere Ecke im
    Grid (0 … grid_um, Mitte = grid_um / 2); x läuft nach rechts, y nach unten –
    wie innerhalb eines Einzelbildes. Ein Vollscan (Offset 0/0, Scanbereich
    grid_um) füllt das Grid also genau aus.

    Auflösung: feinste Pixelgröße aller Bilder; gröbere Bilder (z. B. der
    Vollscan) werden per Nearest-Neighbour hochskaliert. Bilder mit größerem
    Scanbereich werden zuerst gezeichnet, damit kleine Kacheln oben liegen.

    Parameters
    ----------
    datasets           : list  Datensätze (mit "meta"; Bilder ohne werden übersprungen)
    channel            : str   "sum", "detector0" oder "detector1"
    grid_um            : float Kantenlänge des Grids in µm
    include_full_scans : bool  Bilder mit Scanbereich ≥ grid_um als Hintergrund einbeziehen
    bg_correction      : bool  Hintergrundkorrektur je Einzelbild (siehe get_channel_array)

    Returns
    -------
    None, wenn kein Bild platziert werden konnte, sonst dict mit
      "image"      : float32-Array (n × n), leere Bereiche = 0
      "covered"    : bool-Array (n × n), True wo ein Bild liegt
      "px_um"      : Pixelgröße des Mosaiks in µm
      "placements" : Liste (name, row0, col0, height, width) in Mosaik-Pixeln
      "no_meta"    : Namen ohne Scan-Metadaten
      "outside"    : Namen, die ganz außerhalb des Grids liegen
      "duplicates" : Gruppen von Namen mit identischer Position und Größe
    """
    no_meta = [ds["name"] for ds in datasets if ds.get("meta") is None]
    tiles = [ds for ds in datasets if ds.get("meta") is not None]
    if not include_full_scans:
        # Toleranz: Scanbereiche stehen als float64 in Metern in der Datei
        # (z. B. 199.99999999999997 µm statt 200).
        limit = grid_um * (1 - 1e-6)
        tiles = [ds for ds in tiles
                 if ds["meta"].range_x_um < limit or ds["meta"].range_y_um < limit]
    if not tiles:
        return None

    px_um = min(min(ds["meta"].range_x_um / ds["meta"].n_pixels_x,
                    ds["meta"].range_y_um / ds["meta"].n_pixels_y) for ds in tiles)
    n = int(round(grid_um / px_um))
    image = np.zeros((n, n), dtype=np.float32)
    covered = np.zeros((n, n), dtype=bool)
    placements, outside = [], []

    by_position = {}
    for ds in tiles:
        m = ds["meta"]
        key = tuple(round(v, 3) for v in (m.offset_x_um, m.offset_y_um, m.range_x_um, m.range_y_um))
        by_position.setdefault(key, []).append(ds["name"])
    duplicates = [names for names in by_position.values() if len(names) > 1]

    for ds in sorted(tiles, key=lambda d: -(d["meta"].range_x_um * d["meta"].range_y_um)):
        m = ds["meta"]
        w = max(1, int(round(m.range_x_um / px_um)))
        h = max(1, int(round(m.range_y_um / px_um)))
        arr = get_channel_array(ds, channel, bg_correction=bg_correction).astype(np.float32)
        if arr.shape != (h, w):
            arr = np.asarray(Image.fromarray(arr, mode="F").resize(
                (w, h), Image.Resampling.NEAREST), dtype=np.float32)
        r0 = int(round(m.offset_y_um / px_um))
        c0 = int(round(m.offset_x_um / px_um))
        # Auf das Grid zuschneiden
        rs, cs = max(r0, 0), max(c0, 0)
        re, ce = min(r0 + h, n), min(c0 + w, n)
        if rs >= re or cs >= ce:
            outside.append(ds["name"])
            continue
        image[rs:re, cs:ce] = arr[rs - r0:re - r0, cs - c0:ce - c0]
        covered[rs:re, cs:ce] = True
        placements.append((ds["name"], r0, c0, h, w))

    if not placements:
        return None
    return {"image": image, "covered": covered, "px_um": px_um,
            "placements": placements, "no_meta": no_meta,
            "outside": outside, "duplicates": duplicates}


def load_img_dataset(path: str, loader_name: str = "scanbin_s4") -> dict | None:
    """Lädt eine einzelne .img-Datei unabhängig von den im Hauptfenster bereits
    geladenen Datensätzen (z. B. für Werkzeuge, die einen ganzen Ordner selbst
    durchsuchen, wie die Z-Stack-Analyse).

    Versucht zuerst photon_tools (loader="scanbin_s4"), fällt sonst auf
    parse_scanbin_img zurück.

    Returns
    -------
    dict mit "path", "name", "detector0", "detector1", "sum" (alle 2D-float32-
    Arrays), oder None wenn die Datei nicht lesbar ist.
    """
    try:
        import photon_tools as _pt
    except Exception:
        _pt = None

    if _pt is not None:
        try:
            ds = _pt.load_image(path, loader=loader_name)
            det0 = np.asarray(ds.channels["detector0"], dtype=np.float32)
            det1 = np.asarray(ds.channels["detector1"], dtype=np.float32)
            return {"path": path, "name": os.path.basename(path),
                    "detector0": det0, "detector1": det1, "sum": det0 + det1,
                    "meta": read_scanbin_meta(path)}
        except Exception:
            pass

    try:
        return parse_scanbin_img(path)
    except Exception:
        return None
