"""
ics_thickness.py — Schichtdickenbestimmung via Image Correlation Spectroscopy (ICS)
====================================================================================

Bestimmt die Schichtdicke eines duennen Farbstofffilms aus konfokalen
Einzelmolekuel-Rasterbildern mittels klassischer Image Correlation
Spectroscopy (Petersen et al. 1993, Biophys. J. 65, 1135).

Wissenschaftliches Modell (siehe auch Docstrings der einzelnen Funktionen)
---------------------------------------------------------------------------
Normierte raeumliche Autokorrelationsfunktion des Bildes i(x,y), mit der
Intensitaetsfluktuation di = i - <i>:

    g(xi, eta) = < di(x,y) * di(x+xi, y+eta) >_{x,y} / <i>^2

berechnet ueber das Wiener-Khinchin-Theorem (FFT):

    g = ifft2( |fft2(di)|^2 ) / (N_px * <i>^2)      [anschliessend fftshift]

Fitmodell (2D-Gauss plus Offset):

    g(xi, eta) = g0 * exp( -(xi^2 + eta^2) / w0^2 ) + g_inf

Auswertung:

    <N>   = 1 / g0                      # mittlere Teilchenzahl pro Strahlflaeche
    sigma = <N> / (pi * w0^2)           # Flaechendichte [1/m^2]
    rho_V = c * N_A                     # Volumendichte [1/m^3], c in mol/m^3
    h     = sigma / rho_V               # Schichtdicke [m]

Physikalische Randbedingungen (siehe PSF-/Modell-Warnungen zur Laufzeit)
---------------------------------------------------------------------------
* w0 ist direkt der e^-2-Radius der PSF: Die Autokorrelation von
  exp(-2 r^2 / w0^2) mit sich selbst ist proportional zu exp(-xi^2 / w0^2) -
  dieselbe Breite w0 taucht daher unveraendert im ACF-Fit wieder auf.
* sigma = rho_V * h gilt nur, solange h klein gegen die axiale
  PSF-Ausdehnung ist (die Kachel erfasst die Schicht vollstaendig axial).
* g0 = 1/<N> gilt streng nur bei einheitlicher Emitterhelligkeit. Bei
  Helligkeitsstreuung gilt allgemein g0 = (1/<N>) * <eps^2>/<eps>^2, wodurch
  <N> und damit h systematisch UNTERSCHAETZT werden (dieses Modul geht von
  einheitlicher Helligkeit aus und korrigiert diesen Effekt NICHT).

Anwendungsbeispiel
-------------------
    from pathlib import Path
    from ics_thickness import MeasurementConfig, analyze, format_report, plot_results

    config = MeasurementConfig(
        tile_dir=Path("./raster_scan"),
        tile_glob="*.tif",
        px_size_m=20e-9,                # ANNAHME: aus Scanner-Kalibrierung
        conc_mol_per_L=1e-9,             # angesetzte Loesungskonzentration
        expected_w0_m=250e-9,            # aus NA und Anregungswellenlaenge
        fit_radius_px=20,
        detrend=True,
        detrend_sigma_px=60.0,           # >> w0/px_size_m
        axial_psf_extent_m=700e-9,       # konfokale Tiefenschaerfe
        output_dir=Path("./ics_out"),
    )
    results = analyze(config)
    print(format_report(results))
    plot_results(results, output_path=config.output_dir / "ics_report.png")

Oder als CLI:

    python ics_thickness.py --tile-dir ./raster_scan --tile-glob "*.tif" \\
        --px-size-m 20e-9 --conc-mol-per-l 1e-9 --expected-w0-m 250e-9 \\
        --fit-radius-px 20 --detrend-sigma-px 60 --output-dir ./ics_out

Autor: Yannik Kasprzak, Institut fuer Physik, Universitaet zu Luebeck
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.constants import N_A
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit

try:
    import tifffile
except ImportError:
    tifffile = None  # nur benoetigt, wenn TIFF-Dateien geladen werden

try:
    import h5py
except ImportError:
    h5py = None  # nur benoetigt, wenn HDF5-Dateien geladen werden


# ── Fehlerklasse ────────────────────────────────────────────────────────────


class ACFFitError(RuntimeError):
    """Wird ausgeloest, wenn der 2D-Gauss-Fit an eine ACF fehlschlaegt oder ein
    unphysikalisches Ergebnis liefert.

    Diese Fehler werden NIE stillschweigend abgefangen und als Erfolg
    ausgegeben (siehe Modul-Abschnitt "Verbote" der Spezifikation) - beim
    Ensemble-Fit propagiert der Fehler direkt aus analyze(); bei
    Einzel-Kachel-Fits wird er als Python-`warnings.warn` sichtbar gemacht
    und die betroffene Kachel aus der Kachelstatistik ausgeschlossen.
    """


# ── Konfiguration & Ergebnis-Datentypen ──────────────────────────────────────


@dataclass(frozen=True)
class MeasurementConfig:
    """Konfiguration einer ICS-Schichtdickenmessung.

    Alle Groessen sind SI-Einheiten (Meter), mit Ausnahme von
    ``conc_mol_per_L`` (mol/L, wird intern in SI umgerechnet). Es gibt
    bewusst KEINE Defaults fuer Groessen, die die Physik der Messung
    festlegen (``px_size_m``, ``conc_mol_per_L``, ``expected_w0_m``,
    ``fit_radius_px``, ``tile_dir``, ``tile_glob``, ``output_dir``) - diese
    muessen pro Messung explizit gesetzt werden.

    Die Rasterstruktur der Kacheln (z. B. 10x10-Anordnung) wird NICHT als
    Parameter gefuehrt, da ICS jede Kachel unabhaengig auswertet und keine
    raeumliche Nachbarschaftsinformation zwischen Kacheln benoetigt.

    ``tile_dir``/``tile_glob`` werden nur fuer den datei-basierten Pfad
    (``load_tiles``/``analyze``) benoetigt und sind daher optional: wird eine
    aufrufende Anwendung (z. B. eine GUI, die Bilder ueber ein eigenes Format
    laedt) die Kacheln bereits als Arrays im Speicher haben, nutzt sie
    ``analyze_tiles(tiles, config)`` direkt und laesst beide Felder auf
    ``None``.

    Attributes
    ----------
    px_size_m : float
        Physikalische Kantenlaenge eines Pixels in Metern.
    conc_mol_per_L : float
        Molare Fluorophorkonzentration der angesetzten Loesung [mol/L].
    expected_w0_m : float
        Erwarteter PSF-e^-2-Radius in Metern (aus NA und Anregungswellenlaenge
        abgeschaetzt). Dient NUR als Fit-Startwert und fuer die
        Plausibilitaetspruefung in Abschnitt 7 - kein harter Zwang fuer den Fit.
    fit_radius_px : int
        Halbe Kantenlaenge (in Pixeln) des quadratischen Fit-Fensters um
        Lag (0, 0) der ACF.
    exclude_center : bool, optional
        Ob der Lag (0, 0) (Schrotrausch-Spitze) vom Fit ausgeschlossen wird.
        Default True (siehe Abschnitt 4, Punkt 1 der Spezifikation).
    detrend : bool, optional
        Ob vor der Korrelation ein grossskaliger Gauss-Hintergrund
        (Beleuchtungsinhomogenitaet) abgezogen wird. Default True.
    detrend_sigma_px : float | None, optional
        Sigma des Gauss-Hintergrundfilters in Pixeln. MUSS gesetzt sein, wenn
        ``detrend=True``. Muss >> w0 (in Pixeln) gewaehlt werden - sonst wird
        das molekulare Signal selbst mitgefiltert und g0 verfaelscht. Kein
        Default vorhanden, da dieser Wert von der tatsaechlichen
        Beleuchtungs-/Detektionsinhomogenitaet der Messung abhaengt.
    axial_psf_extent_m : float | None, optional
        Axiale PSF-Ausdehnung (z. B. konfokale Tiefenschaerfe) in Metern, nur
        fuer die Plausibilitaetspruefung "h vs. axiale Erfassung" (Abschnitt
        7, Punkt 3). ``None`` => diese Pruefung wird uebersprungen (mit
        Warnung), da kein erfundener Defaultwert verwendet werden darf.
    hdf5_dataset_key : str | None, optional
        Name des Datasets innerhalb einer HDF5-Kacheldatei. Erforderlich,
        wenn ``tile_glob`` auf ``.h5``/``.hdf5``-Dateien passt.
    output_dir : Path
        Zielverzeichnis fuer Report- und Plot-Ausgaben.
    tile_dir : Path | None, optional
        Verzeichnis mit den Rasterkacheln. Nur fuer ``load_tiles``/``analyze``
        (datei-basierter Pfad) erforderlich.
    tile_glob : str | None, optional
        Glob-Muster innerhalb von ``tile_dir`` (z. B. ``"*.tif"``). Passt eine
        Datei zu einem 3D-Array (Stack), wird jede Ebene entlang der ersten
        Achse als eigene Kachel behandelt (TIFF-Stack-Unterstuetzung). Nur
        fuer den datei-basierten Pfad erforderlich.
    """

    px_size_m: float
    conc_mol_per_L: float
    expected_w0_m: float
    fit_radius_px: int
    output_dir: Path
    tile_dir: Path | None = None
    tile_glob: str | None = None
    exclude_center: bool = True
    detrend: bool = True
    detrend_sigma_px: float | None = None
    axial_psf_extent_m: float | None = None
    hdf5_dataset_key: str | None = None

    def __post_init__(self) -> None:
        if self.px_size_m <= 0:
            raise ValueError(f"px_size_m muss > 0 sein, erhalten {self.px_size_m}.")
        if self.conc_mol_per_L <= 0:
            raise ValueError(f"conc_mol_per_L muss > 0 sein, erhalten {self.conc_mol_per_L}.")
        if self.expected_w0_m <= 0:
            raise ValueError(f"expected_w0_m muss > 0 sein, erhalten {self.expected_w0_m}.")
        if self.fit_radius_px < 1:
            raise ValueError(f"fit_radius_px muss >= 1 sein, erhalten {self.fit_radius_px}.")
        if self.detrend and self.detrend_sigma_px is None:
            raise ValueError(
                "detrend=True erfordert detrend_sigma_px (siehe Docstring von "
                "MeasurementConfig) - kein Default vorhanden, da dieser Wert von "
                "der Beleuchtungsinhomogenitaet der jeweiligen Messung abhaengt."
            )
        if self.detrend_sigma_px is not None and self.detrend_sigma_px <= 0:
            raise ValueError(
                f"detrend_sigma_px muss > 0 sein, erhalten {self.detrend_sigma_px}.")


@dataclass(frozen=True)
class GaussianACFParams:
    """Parameter des 2D-Gauss-Fits an die ACF.

    Attributes
    ----------
    g0 : float
        Amplitude der ACF bei Lag 0, aus dem umgebenden Verlauf extrapoliert
        (dimensionslos).
    w0_m : float
        e^-2-Radius der PSF in Metern (Breite des Gauss-Fits, ueber
        ``px_size_m`` in SI-Einheiten umgerechnet).
    g_inf : float
        Konstanter Offset des Fits (Residual-Korrelation/Basislinie,
        dimensionslos).
    """

    g0: float
    w0_m: float
    g_inf: float


@dataclass(frozen=True)
class AnalysisResults:
    """Gesamtergebnis der ICS-Pipeline fuer einen Satz Rasterkacheln.

    Attributes
    ----------
    config : MeasurementConfig
        Konfiguration, mit der dieses Ergebnis erzeugt wurde.
    n_tiles : int
        Anzahl geladener Kacheln.
    example_tile : np.ndarray
        Rohdaten der ersten Kachel (nur fuer Plot (a)).
    acf_mean : np.ndarray
        Ueber alle Kacheln gemittelte, zentrierte 2D-ACF (volle Kachelgroesse).
    ensemble_params : GaussianACFParams
        Fit-Ergebnis (Punktschaetzer) an ``acf_mean``.
    ensemble_errors : GaussianACFParams
        Formale 1-sigma-Standardfehler aus der Kovarianzmatrix des
        Ensemble-Fits.
    sigma_per_m2 : float
        Flaechendichte [1/m^2] aus dem Ensemble-Fit.
    sigma_rel_error : float
        Relative formale Unsicherheit von sigma aus Fehlerfortpflanzung von
        g0 und w0 (dimensionslos, z. B. 0.05 = 5 %).
    h_m : float
        Schichtdicke [m] aus dem Ensemble-Fit (sigma / rho_V).
    h_rel_error : float
        Relative formale Unsicherheit von h (== ``sigma_rel_error``, da
        rho_V als exakt angenommen wird - siehe ``format_report``).
    per_tile_w0_m : np.ndarray
        w0 [m] aus dem Einzel-Fit jeder erfolgreich gefitteten Kachel.
    per_tile_h_m : np.ndarray
        h [m] aus dem Einzel-Fit jeder erfolgreich gefitteten Kachel.
    per_tile_h_mean_m : float
        Mittelwert von ``per_tile_h_m``.
    per_tile_h_std_m : float
        Empirische Standardabweichung (ddof=1) von ``per_tile_h_m`` - die
        Kachel-zu-Kachel-Streuung als zweite, unabhaengige Unsicherheitsangabe.
    warnings_raised : list[str]
        Alle waehrend der Analyse ausgeloesten Plausibilitaets-/Fit-Warnungen,
        als Klartext fuer den Report.
    """

    config: MeasurementConfig
    n_tiles: int
    example_tile: np.ndarray
    acf_mean: np.ndarray
    ensemble_params: GaussianACFParams
    ensemble_errors: GaussianACFParams
    sigma_per_m2: float
    sigma_rel_error: float
    h_m: float
    h_rel_error: float
    per_tile_w0_m: np.ndarray
    per_tile_h_m: np.ndarray
    per_tile_h_mean_m: float
    per_tile_h_std_m: float
    warnings_raised: list[str]


# ── Kachel-Laden ──────────────────────────────────────────────────────────────


def _load_single_image(path: Path, hdf5_dataset_key: str | None) -> np.ndarray:
    """Laedt eine einzelne Bilddatei anhand ihrer Endung.

    Unterstuetzt ``.tif``/``.tiff`` (auch Stacks), ``.npy`` sowie
    ``.h5``/``.hdf5`` (erfordert ``hdf5_dataset_key``).

    Parameters
    ----------
    path : Path
        Pfad zur Bilddatei.
    hdf5_dataset_key : str | None
        Dataset-Name innerhalb einer HDF5-Datei; erforderlich fuer
        ``.h5``/``.hdf5``.

    Returns
    -------
    np.ndarray
        Geladenes Array (2D oder 3D-Stack), unveraendert im urspruenglichen
        dtype.
    """
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        if tifffile is None:
            raise ImportError(
                "tifffile ist nicht installiert (pip install tifffile), aber fuer "
                f"das Laden von '{path}' erforderlich."
            )
        return np.asarray(tifffile.imread(str(path)))
    if suffix == ".npy":
        return np.asarray(np.load(path))
    if suffix in (".h5", ".hdf5"):
        if h5py is None:
            raise ImportError(
                "h5py ist nicht installiert (pip install h5py), aber fuer das "
                f"Laden von '{path}' erforderlich."
            )
        if hdf5_dataset_key is None:
            raise ValueError(
                f"'{path}': HDF5-Datei erfordert MeasurementConfig.hdf5_dataset_key "
                "(Name des Datasets innerhalb der Datei) - kein Default vorhanden."
            )
        with h5py.File(path, "r") as f:
            if hdf5_dataset_key not in f:
                raise KeyError(
                    f"'{path}': Dataset '{hdf5_dataset_key}' nicht gefunden. "
                    f"Vorhandene Keys: {list(f.keys())}"
                )
            return np.asarray(f[hdf5_dataset_key][()])
    raise ValueError(
        f"'{path}': nicht unterstuetztes Dateiformat '{suffix}'. Unterstuetzt: "
        ".tif/.tiff, .npy, .h5/.hdf5"
    )


def load_tiles(config: MeasurementConfig) -> list[np.ndarray]:
    """Laedt alle Rasterkacheln gemaess ``config`` robust von der Festplatte.

    Prueft, dass alle Kacheln dieselbe Pixel-Shape haben (Voraussetzung fuer
    die Kachelmittelung der ACFs in ``analyze``), und gibt bei Verstoessen
    aussagekraeftige Fehlermeldungen statt stillschweigend zu clampen oder
    zu ueberspringen.

    Parameters
    ----------
    config : MeasurementConfig
        Enthaelt ``tile_dir``, ``tile_glob`` und ``hdf5_dataset_key``.
        ``tile_dir``/``tile_glob`` muessen gesetzt sein (siehe
        ``analyze_tiles`` fuer den Pfad mit bereits im Speicher vorliegenden
        Kacheln).

    Returns
    -------
    list[np.ndarray]
        Liste von 2D-float64-Arrays, eines pro Kachel.

    Raises
    ------
    ValueError
        Wenn ``tile_dir``/``tile_glob`` nicht gesetzt sind, Kacheln
        unterschiedliche Shapes haben, oder ein geladenes Array weder 2D
        noch 3D ist.
    FileNotFoundError
        Wenn ``tile_dir`` nicht existiert oder kein Datei zum Glob-Muster passt.
    """
    if config.tile_dir is None or config.tile_glob is None:
        raise ValueError(
            "load_tiles() erfordert MeasurementConfig.tile_dir und .tile_glob. "
            "Liegen die Kacheln bereits als Arrays im Speicher vor (z. B. aus "
            "einer GUI mit eigenem Ladeformat), stattdessen analyze_tiles(tiles, "
            "config) direkt aufrufen."
        )
    if not config.tile_dir.is_dir():
        raise FileNotFoundError(
            f"tile_dir '{config.tile_dir}' existiert nicht oder ist kein Verzeichnis.")

    paths = sorted(config.tile_dir.glob(config.tile_glob))
    if not paths:
        raise FileNotFoundError(
            f"Keine Dateien gefunden fuer Muster '{config.tile_glob}' in "
            f"'{config.tile_dir}'."
        )

    tiles: list[np.ndarray] = []
    shapes: set[tuple[int, ...]] = set()
    for path in paths:
        arr = _load_single_image(path, config.hdf5_dataset_key)
        if arr.ndim == 2:
            candidates = [arr]
        elif arr.ndim == 3:
            # TIFF-/HDF5-Stack: erste Achse wird als Kachel-Index interpretiert.
            candidates = [arr[i] for i in range(arr.shape[0])]
        else:
            raise ValueError(
                f"'{path}': erwarte 2D-Bild oder 3D-Stack (Kachel-Achse zuerst), "
                f"erhalten shape {arr.shape}."
            )
        for c in candidates:
            c64 = np.asarray(c, dtype=np.float64)
            tiles.append(c64)
            shapes.add(c64.shape)

    if len(shapes) > 1:
        raise ValueError(
            f"Kacheln haben unterschiedliche Formen: {sorted(shapes)}. Alle "
            "Kacheln muessen dieselbe Pixelanzahl haben, da spatial_acf() und "
            "die Kachelmittelung dieselbe Geometrie voraussetzen."
        )
    return tiles


# ── Raeumliche Autokorrelation ────────────────────────────────────────────────


def spatial_acf(
    img: np.ndarray,
    detrend: bool = True,
    detrend_sigma_px: float | None = None,
) -> np.ndarray:
    """Normierte, zentrierte 2D-Autokorrelation einer Rasterkachel (ICS).

    Implementiert exakt

        g(xi, eta) = < di(x,y) * di(x+xi, y+eta) >_{x,y} / <i>^2

    ueber das Wiener-Khinchin-Theorem:

        g = ifft2( |fft2(di)|^2 ) / (N_px * <i>^2)     [danach fftshift]

    mit ``di = i - <i>``. Es handelt sich um eine zirkulaere (periodische)
    Korrelation - fuer Lags weit unterhalb der Kachelgroesse (wie sie fuer
    den spaeteren Fit relevant sind) ist der Unterschied zur linearen ACF
    vernachlaessigbar.

    Detrending (optional, ``detrend=True``)
    -----------------------------------------
    Zieht einen grossskaligen Gauss-Hintergrund (sigma = ``detrend_sigma_px``,
    MUSS >> w0/px_size_m gewaehlt werden) vom Bild ab, um
    Beleuchtungs-/Detektionsinhomogenitaet ueber die Kachel zu entfernen,
    OHNE den Mittelwert zu veraendern (der Mittelwert wird nach der
    Subtraktion exakt wiederhergestellt). Dies ist bewusst KEIN
    Tiefpassfilter auf das eigentliche Korrelationssignal: Da
    ``detrend_sigma_px`` weit oberhalb der PSF-Breite liegen muss, wird
    ausschliesslich Struktur auf Ortsfrequenzen weit unterhalb von 1/w0
    entfernt - das molekulare Signal (und damit g0) bleibt unangetastet. Ein
    Tiefpassfilter, der auch die molekulare Ortsfrequenz beeinflusst, ist
    NICHT zulaessig, da er die Schrotrausch-/Emitterstatistik veraendern und
    g0 verfaelschen wuerde.

    Parameters
    ----------
    img : np.ndarray
        2D-Rasterkachel (Rohintensitaet, z. B. Photonenzaehlrate pro Pixel).
    detrend : bool, optional
        Ob ein Gauss-Hintergrund vor der Korrelation abgezogen wird.
        Default True.
    detrend_sigma_px : float | None, optional
        Sigma des Gauss-Hintergrundfilters in Pixeln. Erforderlich, wenn
        ``detrend=True``.

    Returns
    -------
    np.ndarray
        Zentrierte 2D-ACF g(xi, eta), gleiche Shape wie ``img``; Lag (0, 0)
        liegt in der Mitte des Arrays (``shape[i] // 2``).

    Raises
    ------
    ValueError
        Wenn ``img`` nicht 2D ist, ``detrend=True`` ohne
        ``detrend_sigma_px`` aufgerufen wird, oder die mittlere Intensitaet
        <= 0 ist (Normierung nicht definiert).
    """
    if img.ndim != 2:
        raise ValueError(f"spatial_acf erwartet ein 2D-Bild, erhalten shape {img.shape}.")
    img = np.asarray(img, dtype=np.float64)

    if detrend:
        if detrend_sigma_px is None:
            raise ValueError(
                "detrend=True erfordert detrend_sigma_px (Breite des "
                "Gauss-Hintergrunds in Pixeln, muss >> w0 gewaehlt werden)."
            )
        if detrend_sigma_px <= 0:
            raise ValueError(f"detrend_sigma_px muss > 0 sein, erhalten {detrend_sigma_px}.")
        background = gaussian_filter(img, sigma=detrend_sigma_px, mode="reflect")
        diff = img - background
        # Mittelwert EXAKT wiederherstellen (unabhaengig von Randeffekten des
        # Filters) - "Gauss-Hintergrund subtrahieren, Mittelwert erhalten".
        img_proc = diff - diff.mean() + img.mean()
    else:
        img_proc = img

    mean_i = float(img_proc.mean())
    if mean_i <= 0:
        raise ValueError(
            f"Mittlere Intensitaet <i> = {mean_i:.4g} <= 0 - ACF-Normierung "
            "(Division durch <i>^2) ist nicht definiert."
        )

    di = img_proc - mean_i
    n_px = di.size
    spectrum = np.fft.fft2(di)
    g = np.fft.ifft2(np.abs(spectrum) ** 2).real / (n_px * mean_i ** 2)
    return np.fft.fftshift(g)


# ── 2D-Gauss-Fit ──────────────────────────────────────────────────────────────


def _gaussian_acf_model(xy: np.ndarray, g0: float, w0_px: float, g_inf: float) -> np.ndarray:
    """Fitmodell g(xi, eta) = g0 * exp(-(xi^2 + eta^2) / w0_px^2) + g_inf."""
    xi, eta = xy
    return g0 * np.exp(-(xi**2 + eta**2) / w0_px**2) + g_inf


def fit_acf(
    acf: np.ndarray,
    px_size_m: float,
    fit_radius_px: int,
    exclude_center: bool = True,
    initial_w0_px: float | None = None,
) -> tuple[GaussianACFParams, GaussianACFParams]:
    """2D-Gauss-Fit an eine zentrierte ACF, auf ein quadratisches Fenster um Lag 0.

    Schrotrausch-Korrektur (verpflichtend, siehe Modul-Docstring)
    -----------------------------------------------------------------
    Der Zentralpixel g(0, 0) enthaelt neben der PSF-Korrelation eine
    Delta-Spitze aus unkorreliertem Schrotrauschen (Poisson-Statistik der
    Detektion). Mit ``exclude_center=True`` (Default) wird dieser Punkt aus
    den Fit-Daten entfernt; g0 wird stattdessen aus dem umgebenden
    Gauss-Verlauf extrapoliert.

    Parameters
    ----------
    acf : np.ndarray
        Quadratische, zentrierte 2D-ACF (z. B. aus ``spatial_acf``).
    px_size_m : float
        Pixelgroesse in Metern, zur Umrechnung von w0 in SI-Einheiten.
    fit_radius_px : int
        Halbe Kantenlaenge des quadratischen Fit-Fensters um Lag (0, 0), in
        Pixeln.
    exclude_center : bool, optional
        Ob der Lag (0, 0) vom Fit ausgeschlossen wird. Default True.
    initial_w0_px : float | None, optional
        Numerischer Startwert fuer w0 (in Pixeln) fuer den Optimierer - KEINE
        physikalische Annahme, sondern reine Konvergenzhilfe. Ohne Angabe
        wird ``fit_radius_px / 2`` verwendet (ebenfalls rein numerisch,
        strukturell an das Fit-Fenster gekoppelt).

    Returns
    -------
    (GaussianACFParams, GaussianACFParams)
        Punktschaetzer und zugehoerige 1-sigma-Standardfehler (aus der
        Kovarianzmatrix von ``scipy.optimize.curve_fit``), beide in
        SI-Einheiten (w0 in Metern).

    Raises
    ------
    ValueError
        Bei ungueltiger Geometrie (nicht-quadratische ACF, unpassender
        ``fit_radius_px``, zu wenige Datenpunkte).
    ACFFitError
        Wenn ``curve_fit`` nicht konvergiert, die Kovarianzmatrix nicht
        finite Standardfehler liefert, oder das Ergebnis unphysikalisch ist
        (g0 <= 0). Wird NICHT stillschweigend abgefangen.
    """
    if acf.ndim != 2 or acf.shape[0] != acf.shape[1]:
        raise ValueError(f"fit_acf erwartet eine quadratische 2D-ACF, erhalten shape {acf.shape}.")
    ny, nx = acf.shape
    cy, cx = ny // 2, nx // 2
    if fit_radius_px < 1 or fit_radius_px > min(cy, cx):
        raise ValueError(
            f"fit_radius_px={fit_radius_px} liegt ausserhalb des gueltigen "
            f"Bereichs [1, {min(cy, cx)}] fuer eine ACF der Groesse {acf.shape}."
        )

    crop = acf[cy - fit_radius_px:cy + fit_radius_px + 1,
               cx - fit_radius_px:cx + fit_radius_px + 1]
    yy, xx = np.mgrid[-fit_radius_px:fit_radius_px + 1, -fit_radius_px:fit_radius_px + 1]

    mask = np.ones_like(crop, dtype=bool)
    if exclude_center:
        mask &= ~((xx == 0) & (yy == 0))

    xdata = np.vstack([xx[mask].ravel().astype(np.float64), yy[mask].ravel().astype(np.float64)])
    ydata = crop[mask].ravel().astype(np.float64)
    if ydata.size < 4:
        raise ValueError(
            f"Nur {ydata.size} Datenpunkte im Fit-Fenster (fit_radius_px="
            f"{fit_radius_px}, exclude_center={exclude_center}) - zu wenige "
            "fuer den 3-Parameter-Fit (g0, w0, g_inf)."
        )

    g0_guess = max(float(np.max(ydata)), 1e-8)
    w0_guess = initial_w0_px if initial_w0_px is not None else max(1.0, fit_radius_px / 2.0)
    p0 = [g0_guess, w0_guess, 0.0]

    try:
        popt, pcov = curve_fit(
            _gaussian_acf_model, xdata, ydata, p0=p0,
            bounds=([0.0, 0.3, -np.inf], [np.inf, fit_radius_px * 4.0, np.inf]),
            maxfev=20000,
        )
    except RuntimeError as exc:
        raise ACFFitError(f"2D-Gauss-Fit an die ACF konvergierte nicht: {exc}") from exc

    if not np.all(np.isfinite(pcov)):
        raise ACFFitError(
            "Kovarianzmatrix des ACF-Fits enthaelt nicht-finite Werte - "
            "Standardfehler nicht bestimmbar. Fit wird als fehlgeschlagen "
            "gewertet (keine stille Erfolgsmeldung)."
        )
    perr = np.sqrt(np.diag(pcov))
    if not np.all(np.isfinite(perr)):
        raise ACFFitError("Standardfehler des ACF-Fits enthalten nicht-finite Werte.")

    g0, w0_px, g_inf = (float(v) for v in popt)
    g0_err, w0_px_err, g_inf_err = (float(v) for v in perr)

    if g0 <= 0:
        raise ACFFitError(
            f"Gefittetes g0={g0:.4g} <= 0 - unphysikalisches Ergebnis "
            "(<N> = 1/g0 waere negativ oder unendlich)."
        )

    params = GaussianACFParams(g0=g0, w0_m=w0_px * px_size_m, g_inf=g_inf)
    errors = GaussianACFParams(g0=g0_err, w0_m=w0_px_err * px_size_m, g_inf=g_inf_err)
    return params, errors


def _radial_profile(acf: np.ndarray, max_r_px: int) -> tuple[np.ndarray, np.ndarray]:
    """Azimuthal gemitteltes radiales Profil g(r) einer zentrierten 2D-ACF.

    Nur fuer die Visualisierung in ``plot_results`` (Panel c) - nicht Teil
    der eigentlichen Fit-Pipeline, die auf dem vollen 2D-Fenster fittet.

    Parameters
    ----------
    acf : np.ndarray
        Zentrierte 2D-ACF.
    max_r_px : int
        Maximaler Radius in Pixeln.

    Returns
    -------
    (np.ndarray, np.ndarray)
        Radiale Lag-Achse [px] und gemitteltes g(r).
    """
    ny, nx = acf.shape
    cy, cx = ny // 2, nx // 2
    yy, xx = np.mgrid[0:ny, 0:nx]
    r_int = np.round(np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)).astype(int)
    sums = np.bincount(r_int.ravel(), weights=acf.ravel(), minlength=max_r_px + 1)[:max_r_px + 1]
    counts = np.bincount(r_int.ravel(), minlength=max_r_px + 1)[:max_r_px + 1]
    counts[counts == 0] = 1
    return np.arange(max_r_px + 1), sums / counts


# ── Dichte- und Dickenberechnung mit Fehlerfortpflanzung ─────────────────────


def _density_and_thickness(
    params: GaussianACFParams,
    errors: GaussianACFParams,
    conc_mol_per_L: float,
) -> tuple[float, float, float, float]:
    """Berechnet sigma, h und deren relative Unsicherheiten aus einem ACF-Fit.

    Fehlerfortpflanzung (vorgegebene Formel):

        d_sigma / sigma = sqrt( (d_g0/g0)^2 + (2 * d_w0/w0)^2 )

    ``rho_V`` (aus ``conc_mol_per_L``) wird als exakt behandelt (keine
    Unsicherheit der Konzentration spezifiziert) - daher ist die relative
    Unsicherheit von h identisch zu der von sigma.

    Parameters
    ----------
    params : GaussianACFParams
        Fit-Punktschaetzer (g0, w0_m, g_inf).
    errors : GaussianACFParams
        Zugehoerige 1-sigma-Standardfehler.
    conc_mol_per_L : float
        Fluorophorkonzentration [mol/L].

    Returns
    -------
    (sigma_per_m2, sigma_rel_error, h_m, h_rel_error)

    Raises
    ------
    ACFFitError
        Wenn ``params.g0 <= 0`` (bereits von ``fit_acf`` ausgeschlossen,
        hier als zusaetzliche Absicherung bei direkter Verwendung).
    """
    if params.g0 <= 0:
        raise ACFFitError(f"g0={params.g0:.4g} <= 0: <N> nicht definiert.")

    n_mean = 1.0 / params.g0
    sigma = n_mean / (np.pi * params.w0_m ** 2)

    rel_g0 = errors.g0 / params.g0
    rel_w0 = errors.w0_m / params.w0_m
    sigma_rel_error = float(np.sqrt(rel_g0 ** 2 + (2.0 * rel_w0) ** 2))

    rho_V = conc_mol_per_L * 1000.0 * N_A  # mol/L -> mol/m^3 (*1e3) -> 1/m^3 (*N_A)
    h = sigma / rho_V
    h_rel_error = sigma_rel_error

    return sigma, sigma_rel_error, h, h_rel_error


# ── Plausibilitaetspruefungen (Abschnitt 7) ───────────────────────────────────


def _run_plausibility_checks(
    config: MeasurementConfig,
    params: GaussianACFParams,
    h_m: float,
) -> list[str]:
    """Laufzeit-Plausibilitaetspruefungen; warnt, bricht aber nicht ab.

    Schwellenwerte fuer (1) und (2) sind explizit vom Nutzer vorgegeben.
    Schwellenwerte fuer (3) und (4) wurden vom Nutzer nur qualitativ
    beschrieben ("in die Groessenordnung kommt", "nicht deutlich kleiner")
    und sind unten als GEWAEHLTER SCHWELLENWERT gekennzeichnet - das sind
    Implementierungs-Konventionen, keine vom Nutzer gemessenen/vorgegebenen
    Werte.

    Parameters
    ----------
    config : MeasurementConfig
        Aktuelle Konfiguration.
    params : GaussianACFParams
        Ensemble-Fit-Ergebnis.
    h_m : float
        Berechnete Schichtdicke [m].

    Returns
    -------
    list[str]
        Alle ausgeloesten Warn-Meldungen (Klartext, fuer den Report).
    """
    msgs: list[str] = []

    def _emit(msg: str) -> None:
        msgs.append(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)

    # (1) w0 weicht > 30 % vom erwarteten PSF-Wert ab (Schwelle vom Nutzer vorgegeben).
    w0_dev = abs(params.w0_m - config.expected_w0_m) / config.expected_w0_m
    if w0_dev > 0.30:
        _emit(
            f"Gefittetes w0 = {params.w0_m * 1e9:.1f} nm weicht um {w0_dev * 100:.0f}% "
            f"vom erwarteten PSF-Wert ({config.expected_w0_m * 1e9:.1f} nm) ab (> 30%). "
            "-> Moegliche Modellverletzung: Defokus, Probendrift oder Aggregate "
            "statt einzelner Emitter."
        )

    # (2) <N> < 0.1 (Schwelle vom Nutzer vorgegeben).
    n_mean = 1.0 / params.g0
    if n_mean < 0.1:
        _emit(
            f"<N> = {n_mean:.3f} < 0.1: sehr duenn besetztes Regime. Die "
            "ICS-Statistik ist hier schwach (wenige korrelierte Ereignisse pro "
            "Beobachtungsflaeche); direktes Punktzaehlen ist in diesem Regime "
            "vermutlich das belastbarere Verfahren."
        )

    # (3) h in Groessenordnung der axialen PSF-Ausdehnung.
    # GEWAEHLTER SCHWELLENWERT (nicht vom Nutzer vorgegeben): 20 % der axialen
    # PSF-Ausdehnung als konservative Schwelle fuer "Groessenordnung erreicht".
    if config.axial_psf_extent_m is None:
        _emit(
            "axial_psf_extent_m ist nicht gesetzt - die Pruefung 'h vs. axiale "
            "PSF-Ausdehnung' wird uebersprungen. Ohne diesen Wert kann nicht "
            "beurteilt werden, ob die Annahme vollstaendiger axialer Erfassung "
            "(sigma = rho_V * h) noch gueltig ist."
        )
    elif h_m > 0.2 * config.axial_psf_extent_m:
        _emit(
            f"h = {h_m * 1e9:.1f} nm erreicht die Groessenordnung der axialen "
            f"PSF-Ausdehnung ({config.axial_psf_extent_m * 1e9:.1f} nm; Verhaeltnis "
            f"{h_m / config.axial_psf_extent_m * 100:.0f}%). -> Die Annahme "
            "vollstaendiger axialer Erfassung (sigma = rho_V * h) ist "
            "moeglicherweise verletzt."
        )

    # (4) Pixelgroesse nicht deutlich kleiner als w0.
    # GEWAEHLTER SCHWELLENWERT (nicht vom Nutzer vorgegeben): px_size < w0 / 3,
    # angelehnt an die Faustregel von mindestens ~2-3 Abtastpunkten pro PSF-Radius.
    if config.px_size_m >= params.w0_m / 3.0:
        _emit(
            f"Pixelgroesse ({config.px_size_m * 1e9:.1f} nm) ist nicht deutlich "
            f"kleiner als das gefittete w0 ({params.w0_m * 1e9:.1f} nm; "
            f"Verhaeltnis px/w0 = {config.px_size_m / params.w0_m:.2f}). -> Die "
            "PSF ist vermutlich unterabgetastet; w0 (und damit sigma, h) kann "
            "verzerrt sein."
        )

    return msgs


# ── Gesamtpipeline ────────────────────────────────────────────────────────────


def analyze(config: MeasurementConfig) -> AnalysisResults:
    """Laedt Kacheln von der Festplatte (``load_tiles``) und fuehrt darauf
    die ICS-Pipeline aus. Duenner Wrapper um ``analyze_tiles`` - siehe dort
    fuer den vollstaendigen Ablauf.

    Fuer bereits im Speicher vorliegende Kacheln (z. B. aus einer GUI mit
    eigenem Ladeformat) stattdessen ``analyze_tiles(tiles, config)`` direkt
    verwenden.

    Parameters
    ----------
    config : MeasurementConfig
        Vollstaendige Messkonfiguration inkl. ``tile_dir``/``tile_glob``.

    Returns
    -------
    AnalysisResults

    Raises
    ------
    ValueError, FileNotFoundError
        Bei ungueltiger Konfiguration oder fehlenden/inkonsistenten Kacheln.
    ACFFitError
        Siehe ``analyze_tiles``.
    """
    tiles = load_tiles(config)
    return analyze_tiles(tiles, config)


def analyze_tiles(tiles: list[np.ndarray], config: MeasurementConfig) -> AnalysisResults:
    """Fuehrt die vollstaendige ICS-Pipeline auf bereits geladenen Kacheln aus:
    korrelieren, fitten, auswerten, Plausibilitaet pruefen.

    Nimmt Kacheln direkt entgegen, statt sie ueber ``config.tile_dir``/
    ``tile_glob`` von der Festplatte zu laden - fuer Aufrufer, die Bilder
    bereits ueber ein eigenes Format im Speicher haben (z. B. eine GUI).
    ``analyze(config)`` ist der datei-basierte duenne Wrapper darum.

    Ablauf
    ------
    1. Pro Kachel: raeumliche ACF berechnen (``spatial_acf``, inkl. Detrending).
    2. ACFs ueber alle Kacheln mitteln (Abschnitt 4, Punkt 3: bessere
       Statistik, lokale Beleuchtungsschwankungen mitteln sich heraus) und
       EINMAL fitten -> Ensemble-Ergebnis mit formalen Standardfehlern.
    3. Zusaetzlich jede Kachel EINZELN fitten, um die Kachel-zu-Kachel-
       Streuung als empirische Unsicherheit auszuweisen.
    4. Plausibilitaetspruefungen (Abschnitt 7) als Python-``warnings``.

    Parameters
    ----------
    tiles : list[np.ndarray]
        Bereits geladene 2D-Rasterkacheln (gleiche Shape vorausgesetzt).
    config : MeasurementConfig
        Messkonfiguration (``tile_dir``/``tile_glob`` werden hier nicht
        verwendet).

    Returns
    -------
    AnalysisResults

    Raises
    ------
    ValueError
        Wenn ``tiles`` leer ist.
    ACFFitError
        Wenn der Ensemble-Fit fehlschlaegt, oder wenn KEIN einziger
        Einzel-Kachel-Fit erfolgreich war (dann ist auch die empirische
        Kachelstreuung nicht bestimmbar).
    """
    n_tiles = len(tiles)
    if n_tiles == 0:
        raise ValueError("analyze_tiles() wurde mit einer leeren Kachelliste aufgerufen.")

    acfs = [
        spatial_acf(t, detrend=config.detrend, detrend_sigma_px=config.detrend_sigma_px)
        for t in tiles
    ]
    acf_mean = np.mean(np.stack(acfs, axis=0), axis=0)

    initial_w0_px = config.expected_w0_m / config.px_size_m
    ens_params, ens_errors = fit_acf(
        acf_mean, config.px_size_m, config.fit_radius_px, config.exclude_center,
        initial_w0_px=initial_w0_px,
    )

    sigma, sigma_rel_err, h, h_rel_err = _density_and_thickness(
        ens_params, ens_errors, config.conc_mol_per_L)

    per_tile_w0_m: list[float] = []
    per_tile_h_m: list[float] = []
    warnings_raised: list[str] = []
    for i, acf in enumerate(acfs):
        try:
            p, e = fit_acf(
                acf, config.px_size_m, config.fit_radius_px, config.exclude_center,
                initial_w0_px=initial_w0_px,
            )
            _, _, h_i, _ = _density_and_thickness(p, e, config.conc_mol_per_L)
        except ACFFitError as exc:
            msg = (f"Kachel {i}: Einzel-Kachel-Fit fehlgeschlagen ({exc}) - "
                   "aus Kachelstatistik ausgeschlossen.")
            warnings_raised.append(msg)
            warnings.warn(msg, RuntimeWarning)
            continue
        per_tile_w0_m.append(p.w0_m)
        per_tile_h_m.append(h_i)

    if not per_tile_h_m:
        raise ACFFitError(
            "Kein einziger Kachel-Einzel-Fit war erfolgreich - die kachelweise "
            "Streuung (empirische Unsicherheit) kann nicht bestimmt werden."
        )

    per_tile_w0_arr = np.array(per_tile_w0_m)
    per_tile_h_arr = np.array(per_tile_h_m)
    per_tile_h_mean = float(np.mean(per_tile_h_arr))
    per_tile_h_std = (
        float(np.std(per_tile_h_arr, ddof=1)) if len(per_tile_h_arr) > 1 else float("nan")
    )

    warnings_raised.extend(_run_plausibility_checks(config, ens_params, h))

    return AnalysisResults(
        config=config,
        n_tiles=n_tiles,
        example_tile=tiles[0],
        acf_mean=acf_mean,
        ensemble_params=ens_params,
        ensemble_errors=ens_errors,
        sigma_per_m2=sigma,
        sigma_rel_error=sigma_rel_err,
        h_m=h,
        h_rel_error=h_rel_err,
        per_tile_w0_m=per_tile_w0_arr,
        per_tile_h_m=per_tile_h_arr,
        per_tile_h_mean_m=per_tile_h_mean,
        per_tile_h_std_m=per_tile_h_std,
        warnings_raised=warnings_raised,
    )


# ── Report & Plot ──────────────────────────────────────────────────────────────


def format_report(results: AnalysisResults) -> str:
    """Formatiert ein menschenlesbares Textprotokoll aus ``AnalysisResults``.

    Benennt explizit, welche der zwei berichteten Unsicherheiten (formale
    Fit-Unsicherheit vs. empirische Kachelstreuung) was bedeutet
    (Abschnitt 5 der Spezifikation).

    Parameters
    ----------
    results : AnalysisResults

    Returns
    -------
    str
    """
    p, e = results.ensemble_params, results.ensemble_errors
    h_formal_err_m = results.h_m * results.h_rel_error
    lines = [
        "=" * 72,
        "ICS-Schichtdickenbestimmung - Ergebnisbericht",
        "=" * 72,
        f"Anzahl Kacheln:              {results.n_tiles}",
        f"Erfolgreiche Einzel-Fits:    {len(results.per_tile_h_m)} / {results.n_tiles}",
        "",
        "Ensemble-Fit (2D-Gauss-Fit an die ueber alle Kacheln gemittelte ACF):",
        f"  g0    = {p.g0:.5g}  +/- {e.g0:.2g}",
        f"  w0    = {p.w0_m * 1e9:.3f}  +/- {e.w0_m * 1e9:.3f} nm",
        f"  g_inf = {p.g_inf:.3g}  +/- {e.g_inf:.2g}",
        "",
        f"Flaechendichte sigma = {results.sigma_per_m2:.4g} 1/m^2 "
        f"(relative formale Unsicherheit: {results.sigma_rel_error * 100:.1f}%)",
        "",
        f"Schichtdicke h = {results.h_m * 1e9:.3f} nm",
        f"  - formale Unsicherheit (Fehlerfortpflanzung aus g0-/w0-Fitfehlern "
        f"des Ensemble-Fits): +/- {h_formal_err_m * 1e9:.3f} nm "
        f"({results.h_rel_error * 100:.1f}%)",
        f"  - empirische Kachel-zu-Kachel-Streuung (Std.-Abw. ueber "
        f"{len(results.per_tile_h_m)} unabhaengige Einzel-Kachel-Fits): "
        f"+/- {results.per_tile_h_std_m * 1e9:.3f} nm "
        f"(Kachelmittel: {results.per_tile_h_mean_m * 1e9:.3f} nm)",
        "",
        "Bedeutung der beiden Unsicherheiten:",
        "  * Die FORMALE Unsicherheit beschreibt, wie praezise der 2D-Gauss-Fit",
        "    an die GEMITTELTE ACF bestimmt ist (Guete des Fits selbst, aus der",
        "    Kovarianzmatrix von curve_fit).",
        "  * Die EMPIRISCHE Kachelstreuung beschreibt, wie stark das Ergebnis",
        "    zwischen unabhaengigen Kacheln tatsaechlich schwankt",
        "    (Probenhomogenitaet/Reproduzierbarkeit) - i. d. R. die",
        "    realistischere Gesamtunsicherheit, da sie reale Streuungsquellen",
        "    erfasst, die ein einzelner Fit an gemittelte Daten nicht sieht.",
        "",
    ]
    if results.warnings_raised:
        lines.append("Laufzeit-Warnungen:")
        lines.extend(f"  - {w}" for w in results.warnings_raised)
    else:
        lines.append("Keine Plausibilitaets-Warnungen ausgeloest.")
    lines.append("=" * 72)
    return "\n".join(lines)


def plot_results(results: AnalysisResults, output_path: Path | None = None):
    """Erstellt den 4-Panel-Report: (a) Beispielbild, (b) 2D-ACF-Heatmap,
    (c) radialer Schnitt mit Fitkurve, (d) Verteilung der kachelweisen h-Werte.

    Parameters
    ----------
    results : AnalysisResults
    output_path : Path | None, optional
        Wenn gesetzt, wird die Figure dorthin gespeichert (PNG, 150 dpi).

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    px = results.config.px_size_m
    r = results.config.fit_radius_px

    fig, ((ax_a, ax_b), (ax_c, ax_d)) = plt.subplots(2, 2, figsize=(11, 9))

    # (a) Beispielbild
    im_a = ax_a.imshow(results.example_tile, cmap="inferno")
    ax_a.set_title("(a) Beispielkachel (Rohdaten)")
    ax_a.set_xlabel("x [px]")
    ax_a.set_ylabel("y [px]")
    fig.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04)

    # (b) 2D-ACF-Heatmap (auf Fit-Fenster zugeschnitten)
    ny, nx = results.acf_mean.shape
    cy, cx = ny // 2, nx // 2
    crop = results.acf_mean[cy - r:cy + r + 1, cx - r:cx + r + 1]
    extent_nm = np.array([-r, r, -r, r]) * px * 1e9
    im_b = ax_b.imshow(crop, cmap="viridis", extent=extent_nm, origin="lower")
    ax_b.set_title("(b) Gemittelte 2D-ACF (Fit-Fenster)")
    ax_b.set_xlabel("xi [nm]")
    ax_b.set_ylabel("eta [nm]")
    fig.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)

    # (c) Radialer Schnitt mit Fitkurve
    r_px, g_r = _radial_profile(results.acf_mean, r)
    p = results.ensemble_params
    w0_px = p.w0_m / px
    fit_curve = p.g0 * np.exp(-(r_px.astype(np.float64) ** 2) / w0_px ** 2) + p.g_inf
    ax_c.plot(r_px * px * 1e9, g_r, "o", ms=4, label="Radial gemittelte ACF")
    ax_c.plot(r_px * px * 1e9, fit_curve, "-", lw=2, label="2D-Gauss-Fit")
    if results.config.exclude_center and len(r_px) > 0:
        ax_c.axvline(0, color="gray", ls="--", lw=1,
                     label="Lag 0 (vom Fit ausgeschlossen)")
    ax_c.set_xlabel("Radialer Lag r [nm]")
    ax_c.set_ylabel("g(r)")
    ax_c.set_title("(c) Radialer ACF-Schnitt mit Fit")
    ax_c.legend(fontsize=8)

    # (d) Verteilung der kachelweisen h-Werte
    n_bins = max(5, len(results.per_tile_h_m) // 3)
    ax_d.hist(results.per_tile_h_m * 1e9, bins=n_bins, color="C0", alpha=0.8)
    ax_d.axvline(results.per_tile_h_mean_m * 1e9, color="C1", ls="--",
                label=f"Kachelmittel = {results.per_tile_h_mean_m * 1e9:.2f} nm")
    ax_d.axvline(results.h_m * 1e9, color="C2", ls=":",
                label=f"Ensemble-Fit = {results.h_m * 1e9:.2f} nm")
    ax_d.set_xlabel("h pro Kachel [nm]")
    ax_d.set_ylabel("Anzahl Kacheln")
    ax_d.set_title("(d) Kachelweise Schichtdicken-Verteilung")
    ax_d.legend(fontsize=8)

    fig.suptitle(
        f"ICS-Report: h = ({results.h_m * 1e9:.2f} +/- "
        f"{results.h_m * results.h_rel_error * 1e9:.2f}) nm  (formal)  |  "
        f"+/- {results.per_tile_h_std_m * 1e9:.2f} nm  (Kachelstreuung, n="
        f"{len(results.per_tile_h_m)})"
    )
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


# ── CLI ─────────────────────────────────────────────────────────────────────


def _build_arg_parser() -> argparse.ArgumentParser:
    """Baut den argparse-Parser fuer die Kommandozeilen-Schnittstelle."""
    parser = argparse.ArgumentParser(
        prog="ics_thickness",
        description=(
            "Schichtdickenbestimmung duenner Farbstofffilme aus konfokalen "
            "Einzelmolekuel-Rasterbildern via Image Correlation Spectroscopy (ICS)."
        ),
    )
    parser.add_argument("--tile-dir", type=Path, required=True,
                        help="Verzeichnis mit den Rasterkacheln.")
    parser.add_argument("--tile-glob", type=str, required=True,
                        help="Glob-Muster innerhalb von --tile-dir, z. B. '*.tif'.")
    parser.add_argument("--px-size-m", type=float, required=True,
                        help="Pixelgroesse in Metern.")
    parser.add_argument("--conc-mol-per-l", type=float, required=True,
                        help="Fluorophorkonzentration in mol/L.")
    parser.add_argument("--expected-w0-m", type=float, required=True,
                        help="Erwarteter PSF-e^-2-Radius in Metern "
                             "(Fit-Startwert und Plausibilitaetspruefung).")
    parser.add_argument("--fit-radius-px", type=int, required=True,
                        help="Halbe Kantenlaenge des ACF-Fit-Fensters in Pixeln.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Zielverzeichnis fuer Report und Plot.")
    parser.add_argument("--exclude-center", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Lag (0,0) vom Fit ausschliessen (Schrotrauschen). "
                             "Default: an.")
    parser.add_argument("--detrend", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Gauss-Hintergrund vor der Korrelation abziehen. "
                             "Default: an.")
    parser.add_argument("--detrend-sigma-px", type=float, default=None,
                        help="Sigma des Gauss-Hintergrundfilters in Pixeln "
                             "(erforderlich, wenn --detrend aktiv ist).")
    parser.add_argument("--axial-psf-extent-m", type=float, default=None,
                        help="Axiale PSF-Ausdehnung in Metern, fuer "
                             "Plausibilitaetspruefung (optional).")
    parser.add_argument("--hdf5-dataset-key", type=str, default=None,
                        help="Dataset-Name innerhalb von HDF5-Kacheldateien "
                             "(falls verwendet).")
    return parser


def main(argv: Sequence[str] | None = None) -> AnalysisResults:
    """CLI-Einstiegspunkt: parst Argumente, fuehrt die Analyse aus, schreibt
    Report (stdout + Textdatei) und Plot in ``--output-dir``.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        Argumentliste; ``None`` verwendet ``sys.argv[1:]``.

    Returns
    -------
    AnalysisResults
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    config = MeasurementConfig(
        tile_dir=args.tile_dir,
        tile_glob=args.tile_glob,
        px_size_m=args.px_size_m,
        conc_mol_per_L=args.conc_mol_per_l,
        expected_w0_m=args.expected_w0_m,
        fit_radius_px=args.fit_radius_px,
        output_dir=args.output_dir,
        exclude_center=args.exclude_center,
        detrend=args.detrend,
        detrend_sigma_px=args.detrend_sigma_px,
        axial_psf_extent_m=args.axial_psf_extent_m,
        hdf5_dataset_key=args.hdf5_dataset_key,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    results = analyze(config)

    report = format_report(results)
    print(report)
    (config.output_dir / "ics_report.txt").write_text(report, encoding="utf-8")

    plot_results(results, output_path=config.output_dir / "ics_report.png")

    return results


if __name__ == "__main__":
    main()
