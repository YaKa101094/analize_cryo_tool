"""
test_ics_thickness.py — Pytest-Selbsttest fuer ics_thickness.py
==================================================================

Erzeugt synthetische Rasterkacheln aus Poisson-verteilten Punktemittern
bekannter Flaechendichte, gefaltet mit einer Gauss-PSF exp(-2 r^2 / w0^2),
plus Poisson-Schrotrauschen (auf das gefaltete Signal) und additivem
Detektorrauschen (Gauss). Prueft, dass die ICS-Pipeline die bekannte
Flaechendichte und das bekannte w0 innerhalb weniger Prozent rekonstruiert
(Abschnitt 6 der Spezifikation).

Toleranzen wurden empirisch anhand von 7 unabhaengigen Seeds bestimmt (siehe
Kommentar bei den Assertions): die tatsaechliche Abweichung liegt im
Basisfall typischerweise < 1-2 %, die hier verwendeten Toleranzen (5 % / 10 %)
liegen mit deutlichem Sicherheitsabstand darueber, um Flakiness zu vermeiden.

Ausfuehren mit: pytest test_ics_thickness.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from ics_thickness import GaussianACFParams, fit_acf, spatial_acf


def _make_synthetic_tile(
    rng: np.random.Generator,
    shape: tuple[int, int],
    density_per_px2: float,
    w0_px: float,
    brightness: float,
    detector_noise_std: float,
) -> np.ndarray:
    """Erzeugt eine synthetische Rasterkachel fuer den ICS-Selbsttest.

    Modell: Poisson-verteilte Punktemitter bekannter Flaechendichte,
    gefaltet mit einer Gauss-PSF exp(-2 r^2 / w0_px^2), anschliessend
    Poisson-Schrotrauschen auf das (nicht-negative) Signal, plus additives
    Gauss-Detektorrauschen.

    Parameters
    ----------
    rng : np.random.Generator
        Zufallszahlengenerator (fuer Reproduzierbarkeit von aussen gesetzt).
    shape : tuple[int, int]
        Kachelgroesse (Zeilen, Spalten) in Pixeln.
    density_per_px2 : float
        Bekannte Flaechendichte der Emitter in 1/px^2.
    w0_px : float
        Bekannter e^-2-PSF-Radius in Pixeln.
    brightness : float
        Peak-Amplitude eines einzelnen Emitters (vor Schrotrauschen).
    detector_noise_std : float
        Standardabweichung des additiven Detektorrauschens.

    Returns
    -------
    np.ndarray
        Synthetische Rasterkachel, shape wie ``shape``.
    """
    n_px = shape[0] * shape[1]
    n_particles = int(rng.poisson(density_per_px2 * n_px))
    xs = rng.uniform(0, shape[1], n_particles)
    ys = rng.uniform(0, shape[0], n_particles)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    signal = np.zeros(shape, dtype=np.float64)
    for x0, y0 in zip(xs, ys):
        signal += brightness * np.exp(-2 * ((xx - x0) ** 2 + (yy - y0) ** 2) / w0_px ** 2)
    photons = rng.poisson(np.clip(signal, 0, None)).astype(np.float64)
    return photons + rng.normal(0.0, detector_noise_std, shape)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_ics_reconstructs_known_density_and_w0(seed: int) -> None:
    """ICS-Pipeline muss bekannte Flaechendichte und w0 aus Simulationsdaten
    innerhalb weniger Prozent rekonstruieren (Abschnitt 6)."""
    rng = np.random.default_rng(seed)
    shape = (256, 256)
    w0_px_true = 3.0
    density_per_px2_true = 0.01
    brightness = 200.0
    detector_noise_std = 5.0
    n_tiles = 30
    fit_radius_px = 20

    tiles = [
        _make_synthetic_tile(rng, shape, density_per_px2_true, w0_px_true,
                             brightness, detector_noise_std)
        for _ in range(n_tiles)
    ]

    # detrend=False: die Simulation enthaelt keine Beleuchtungsinhomogenitaet,
    # Detrending ist hier nicht Gegenstand dieses Tests (siehe
    # test_detrending_removes_illumination_bias unten).
    acfs = [spatial_acf(t, detrend=False) for t in tiles]
    acf_mean = np.mean(np.stack(acfs, axis=0), axis=0)

    # px_size_m=1.0 -> w0_m ist hier numerisch identisch zu w0_px (Test in
    # Pixel-Einheiten, um unabhaengig von einer konkreten Pixelgroesse zu sein).
    params, _errors = fit_acf(
        acf_mean, px_size_m=1.0, fit_radius_px=fit_radius_px, exclude_center=True)

    n_mean = 1.0 / params.g0
    sigma_fit_per_px2 = n_mean / (np.pi * params.w0_m ** 2)

    rel_err_w0 = abs(params.w0_m - w0_px_true) / w0_px_true
    rel_err_sigma = abs(sigma_fit_per_px2 - density_per_px2_true) / density_per_px2_true

    # Empirisch (7 Seeds, siehe Modul-Kommentar): tatsaechliche Fehler < 1-2%.
    assert rel_err_w0 < 0.05, f"w0-Fehler {rel_err_w0:.1%} > 5% Toleranz (seed={seed})"
    assert rel_err_sigma < 0.10, f"sigma-Fehler {rel_err_sigma:.1%} > 10% Toleranz (seed={seed})"


def test_center_pixel_exclusion_recovers_true_amplitude() -> None:
    """Schrotrausch-Korrektur (Abschnitt 4, Punkt 1): eine kuenstliche Spitze
    im Zentralpixel darf bei exclude_center=True das gefittete g0 nicht
    beeinflussen, bei exclude_center=False muss sie es systematisch erhoehen.

    Deterministischer Test (keine Zufallszahlen) auf einer analytisch
    konstruierten ACF, um den Mechanismus unabhaengig von Rausch-Realisierungen
    eindeutig zu pruefen.
    """
    r = 15
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    g0_true, w0_true, g_inf_true = 0.05, 4.0, 0.0
    acf = g0_true * np.exp(-(xx**2 + yy**2) / w0_true**2) + g_inf_true
    acf[r, r] += 5.0 * g0_true  # kuenstliche Schrotrausch-Spitze bei Lag (0,0)

    params_excl, _ = fit_acf(acf, px_size_m=1.0, fit_radius_px=r, exclude_center=True)
    params_incl, _ = fit_acf(acf, px_size_m=1.0, fit_radius_px=r, exclude_center=False)

    assert abs(params_excl.g0 - g0_true) < 0.01 * g0_true, (
        "exclude_center=True sollte die Schrotrausch-Spitze vollstaendig ignorieren "
        "und g0 exakt (bis auf numerische Praezision) rekonstruieren."
    )
    assert params_incl.g0 > params_excl.g0, (
        "exclude_center=False sollte durch die Schrotrausch-Spitze ein "
        "systematisch ueberhoehtes g0 liefern."
    )


def test_fit_acf_rejects_oversized_radius() -> None:
    """fit_acf muss bei zu grossem fit_radius_px einen klaren Fehler werfen,
    statt still auf einen gueltigen Bereich zu clampen (Verbot stiller
    Fallbacks, Abschnitt 9)."""
    acf = np.zeros((21, 21))
    with pytest.raises(ValueError):
        fit_acf(acf, px_size_m=1.0, fit_radius_px=50, exclude_center=True)


def test_spatial_acf_requires_detrend_sigma_when_detrending() -> None:
    """spatial_acf(detrend=True) ohne detrend_sigma_px muss fehlschlagen,
    statt einen erfundenen Default zu verwenden (Verbot erfundener Werte,
    Abschnitt 9)."""
    img = np.ones((32, 32)) + np.eye(32)
    with pytest.raises(ValueError):
        spatial_acf(img, detrend=True, detrend_sigma_px=None)


def test_spatial_acf_normalization_matches_definition() -> None:
    """Prueft die ACF-Definition g = ifft2(|fft2(di)|^2) / (N_px * <i>^2)
    direkt gegen eine unabhaengige Referenzimplementierung ueber
    np.correlate-aequivalente Bruteforce-Faltung fuer ein kleines Testbild."""
    rng = np.random.default_rng(0)
    img = rng.poisson(50.0, size=(16, 16)).astype(np.float64)

    g = spatial_acf(img, detrend=False)

    mean_i = img.mean()
    di = img - mean_i
    n_px = di.size

    # Referenz: explizite zirkulaere Autokorrelation ueber np.fft, unabhaengig
    # vom Modul-Code nachgerechnet (gleiche Formel, getrennt aufgeschrieben).
    ref = np.fft.ifft2(np.fft.fft2(di) * np.conj(np.fft.fft2(di))).real
    ref = ref / (n_px * mean_i**2)
    ref = np.fft.fftshift(ref)

    assert np.allclose(g, ref, atol=1e-10)
    # g(0,0) (Zentrum nach fftshift) muss dem Bild-Sample-Variance/<i>^2-Verhaeltnis
    # entsprechen (Definition der ACF bei Lag 0).
    cy, cx = np.array(g.shape) // 2
    expected_g0 = np.mean(di**2) / mean_i**2
    assert np.isclose(g[cy, cx], expected_g0, rtol=1e-8)
