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


# ── .img-Header (scanbin_s4) ──────────────────────────────────────────────────

IMG_HEADER_SIZE = 88

# Byte-Offsets im 88-Byte-Header, alle big-endian. Abgeleitet und gegen die
# Referenzwerte von photon_tools verifiziert (tests/data/img/
# img_FCS_80uW_Alexa488_150pM_012.img -> dwelltime 1 ms, offset 130 µm,
# range 10 µm, 150x150 px). Längen/Zeiten stehen in SI-Einheiten, also
# Meter und Sekunden.
_IMG_HEADER_LAYOUT_F64 = {
    "dwelltime": 0,
    "offset_x": 8,
    "offset_y": 16,
    "range_x": 32,
    "range_y": 40,
}
_IMG_HEADER_LAYOUT_U32 = {
    "pixel_x": 24,
    "pixel_y": 28,
}


@dataclass
class ImageMeta:
    """Physikalische Aufnahmeparameter eines scanbin_s4-Scans.

    Feldnamen und Einheiten entsprechen der ImageMeta von photon_tools, damit
    beide Quellen (natives Laden und Fallback-Parser) austauschbar sind.
    """

    n_pixels_x: int
    n_pixels_y: int
    range_x_um: float
    range_y_um: float
    offset_x_um: float
    offset_y_um: float
    pixel_dwell_time_s: float

    @property
    def pixel_size_x_um(self) -> float:
        """Kantenlänge eines Pixels in x-Richtung [µm]."""
        return self.range_x_um / self.n_pixels_x

    @property
    def pixel_size_y_um(self) -> float:
        """Kantenlänge eines Pixels in y-Richtung [µm]."""
        return self.range_y_um / self.n_pixels_y

    @property
    def frame_time_s(self) -> float:
        """Reine Pixel-Verweilzeit eines kompletten Frames [s] (ohne Overhead)."""
        return self.pixel_dwell_time_s * self.n_pixels_x * self.n_pixels_y


def parse_img_header(raw: bytes) -> dict:
    """Zerlegt die ersten 88 Bytes einer .img-Datei in ein Feld-Dict.

    Parameters
    ----------
    raw : bytes  Mindestens IMG_HEADER_SIZE Bytes vom Dateianfang.

    Returns
    -------
    dict  Schlüssel "dwelltime", "offset_x", "offset_y", "range_x", "range_y"
          (float, SI: Sekunden bzw. Meter) sowie "pixel_x", "pixel_y" (int).
    """
    if len(raw) < IMG_HEADER_SIZE:
        raise ValueError(".img-Header ist zu kurz")

    header = {name: struct.unpack_from(">d", raw, off)[0]
              for name, off in _IMG_HEADER_LAYOUT_F64.items()}
    header.update({name: int.from_bytes(raw[off:off + 4], "big", signed=False)
                   for name, off in _IMG_HEADER_LAYOUT_U32.items()})
    if header["pixel_x"] <= 0 or header["pixel_y"] <= 0:
        raise ValueError("Ungültige Bilddimensionen im .img-Header")
    return header


def img_meta_from_header(header: dict) -> ImageMeta:
    """Rechnet ein Header-Dict in ImageMeta um (SI -> Pixel, µm, Sekunden)."""
    return ImageMeta(
        n_pixels_x=int(header["pixel_x"]),
        n_pixels_y=int(header["pixel_y"]),
        range_x_um=float(header["range_x"]) * 1e6,
        range_y_um=float(header["range_y"]) * 1e6,
        offset_x_um=float(header["offset_x"]) * 1e6,
        offset_y_um=float(header["offset_y"]) * 1e6,
        pixel_dwell_time_s=float(header["dwelltime"]),
    )


def img_frame_count(path: str, meta: ImageMeta) -> int | None:
    """Zahl der Frames in einer .img-Datei, beide Detektoren zusammengezählt.

    Ergibt sich aus der Dateigröße: nach dem Header folgen uint16-Pixel, also
    2 Byte pro Pixel. Unvollständige Frames am Ende werden abgeschnitten.
    """
    try:
        payload = os.path.getsize(path) - IMG_HEADER_SIZE
    except OSError:
        return None
    return payload // 2 // (meta.n_pixels_x * meta.n_pixels_y)


def read_img_meta(path: str) -> ImageMeta | None:
    """Liest nur den Header einer .img-Datei und gibt die Metadaten zurück.

    Returns None, wenn die Datei keine lesbare scanbin_s4-Datei ist — der
    Aufrufer zeigt die Metadaten dann einfach nicht an.
    """
    try:
        with open(path, "rb") as fobj:
            return img_meta_from_header(parse_img_header(fobj.read(IMG_HEADER_SIZE)))
    except (OSError, ValueError, KeyError, struct.error):
        return None


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

def load_img_dataset(path: str, loader_name: str = "scanbin_s4") -> dict | None:
    """Lädt eine einzelne .img-Datei unabhängig von den im Hauptfenster bereits
    geladenen Datensätzen (z. B. für Werkzeuge, die einen ganzen Ordner selbst
    durchsuchen, wie die Z-Stack-Analyse).

    Versucht zuerst photon_tools (loader="scanbin_s4"), fällt sonst auf den
    byteweisen scanbin_s4-Fallback-Parser zurück (siehe
    ImageViewer.open_custom_img für die identische Formatbeschreibung).

    Returns
    -------
    dict mit "path", "name", "detector0", "detector1", "sum" (alle 2D-float32-
    Arrays) sowie "meta" (ImageMeta oder None), oder None wenn die Datei nicht
    lesbar ist.
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
            meta = read_img_meta(path)
            return {"path": path, "name": os.path.basename(path),
                    "detector0": det0, "detector1": det1, "sum": det0 + det1,
                    "meta": meta,
                    "n_frames": (img_frame_count(path, meta)
                                 if meta is not None else None)}
        except Exception:
            pass

    try:
        with open(path, "rb") as fobj:
            data = fobj.read()
        if len(data) <= IMG_HEADER_SIZE:
            return None

        header_fields = parse_img_header(data[:IMG_HEADER_SIZE])
        payload = data[IMG_HEADER_SIZE:]
        width = header_fields["pixel_x"]
        height = header_fields["pixel_y"]

        raw_u16 = np.frombuffer(payload, dtype=">u2")
        pixels_per_frame = width * height
        total_values = (len(raw_u16) // pixels_per_frame) * pixels_per_frame
        if total_values == 0:
            return None

        n_frames = total_values // pixels_per_frame
        frames = raw_u16[:total_values].reshape(n_frames, height, width).astype(np.float32)
        det0_frames = frames[0::2]
        det1_frames = frames[1::2]
        detector0 = det0_frames.sum(axis=0) if det0_frames.size else np.zeros((height, width), dtype=np.float32)
        detector1 = det1_frames.sum(axis=0) if det1_frames.size else np.zeros((height, width), dtype=np.float32)
        return {"path": path, "name": os.path.basename(path),
                "detector0": detector0, "detector1": detector1,
                "sum": detector0 + detector1,
                "meta": img_meta_from_header(header_fields),
                "n_frames": n_frames}
    except Exception:
        return None
