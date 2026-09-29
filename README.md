# analize_cryo_tool

This ist a Tool to analize the messurements at the cryo Setup at the institute of physics, university of lübeck. It contains tools for inspection, estimation of the film thickness and many more.

Tkinter-basiertes Desktop-Tool zum Laden, Visualisieren und Analysieren von Fluoreszenz-Bilddaten im `.img`-Format (scanbin_s4) sowie gängigen Rasterbildformaten.

---

## Voraussetzungen

- Python ≥ 3.9
- Erforderlich:

```
pip install numpy pillow
```

- Optional, je nach genutzter Analyse:

| Paket | Wird gebraucht für |
|---|---|
| `matplotlib` | Alle Plots in den Analyse-Tabs |
| `scipy` | Spot-Erkennung über zusammenhängende Regionen |
| [`photon_tools`](https://github.com/bgamari/photon-tools) | Natives `.img`- und HDF5-Laden; ohne das Paket greift ein eigener Fallback-Parser |
| `pycorrelate`, `lmfit` | FCS-Korrelation und Fit im Timetrace |
| `h5py`, `tifffile` | Ein-/Ausgabe der eigenständigen ICS-Pipeline |

---

## Starten

```bash
python Schichtdicke.py
```

Beim Start sucht das Programm automatisch nach einem Unterordner `img/` im selben Verzeichnis und lädt alle dort gefundenen Bilder.

---

## Unterstützte Dateiformate

| Format | Beschreibung |
|---|---|
| `.img` | scanbin_s4 – proprietäres Format mit zwei Detektorkanälen (detector0, detector1) |
| `.png`, `.jpg`, `.jpeg` | Standardrasterbilder |
| `.tif`, `.tiff` | TIFF |
| `.bmp`, `.webp` | Weitere Rasterformate |

Bei Standard-Rasterbildern wird der Grauwertkanal als `detector0` verwendet; `detector1` bleibt null.

---

## Oberfläche

Die Anwendung besteht aus einem Haupt-Notebook. Alle Werkzeuge sind ab dem Start als Tabs vorhanden, es gibt keinen separaten Auswahlbildschirm.

| Tab | Inhalt |
|---|---|
| **Bildanalyse** | Bildbetrachter, Navigation, Anzeigemodi, PNG-Export |
| **Analyse-Kasprzak** | Methodenauswahl: Timetrace / FCS, Schichtdicke (Zählen), Schichtdicke (ICS) |
| **Analyse-Krüger** | Methodenauswahl: Spot-Zähler (Methode A), ICS-Auswertung (Methode B), Spot-Simulation, Z-Stack (PSF) |

### Bildanalyse

| Steuerelement | Funktion |
|---|---|
| **◀ Prev / Next ▶** | Blättert durch Einzelbilder (nicht im Grid-Modus relevant) |
| **File** | Dropdown zur direkten Auswahl eines Bildes per Index |
| **Display** | Anzeigemodus (s. unten) |
| **Rows / Cols** | Raster-Dimensionen im Grid-Modus |
| **Threshold** | Relativer Schwellwert (0–1) bezogen auf I_max; beeinflusst Darstellung und Spot-Zählung |
| **Namen mitspeichern** | Schreibt den Dateinamen in das exportierte Bild |
| **Rasterbild** | Exportiert statt Einzelbildern ein Übersichtsgitter |
| **Dateinamen statt Nummern** | Beschriftet die Kacheln des Übersichtsgitters mit Dateinamen statt laufender Nummer |
| **Bilder speichern** | Exportiert nach `output/`, jeweils mit Datum und Uhrzeit im Dateinamen |

### Anzeigemodi

- `grid_sum` – Übersichtsgitter aller Bilder (Summenkanal), mit Spot-Zählung und Intensitätsskala
- `sum` – Einzelansicht Summenkanal
- `detector0` / `detector1` – Einzelansicht je Detektor
- `det0_det1` – Detektoren nebeneinander
- `all` – Alle drei Kanäle nebeneinander

Ein Klick auf ein Bild öffnet eine Vollbild-Detailansicht mit Zoom per Mausrad.

---

## Analyse-Funktionen

### Timetrace / FCS

Korreliert Photonenankunftszeiten aus HDF5-Dateien und fittet das Korrelationsmodell. Arbeitet unabhängig von den im Bildbrowser geladenen Datensätzen.

### Schichtdicke (Zählen)

Berechnet für jedes Bild die Schichtdicke aus der Spot-Dichte.

**Threshold-Tabelle:** Zeigt die Spot-Anzahl (oder den Mittelwert über alle Bilder) für Threshold-Werte von 0,00 bis 0,95 in 0,05-Schritten. Exportierbar als CSV.

**Schichtdickenberechnung:**

Eingaben:
- Konzentration *c* [nM]
- Bildfläche *A* [µm²]

Formel:

```
ρ  = c · 10⁻⁹ · N_A / 10¹⁵     [Moleküle/µm³]
d  = N_Spots / (ρ · A) · 1000   [nm]
```

Ausgabe: Schichtdicke pro Bild sowie Mittelwert ± Standardabweichung, exportierbar als CSV.

### Schichtdicke (ICS)

Image Correlation Spectroscopy über die Pipeline in `ics_thickness.py`, die sich auch eigenständig per CLI nutzen lässt.

### Analyse-Krüger

Vier Methoden, ursprünglich als eigenständige Jupyter-Notebooks entwickelt (Wilhelm Krüger) und hier als Untermenü-Punkte eines gemeinsamen Tabs zusammengefasst: Multi-Molekül-Spot-Zähler, ICS-Auswertung, Spot-Simulation und Z-Stack-Auswertung zur PSF-Bestimmung.

---

## Spot-Zählung

Ein Pixel gilt als Spot, wenn:

1. er das lokale Maximum in seiner 3×3-Nachbarschaft ist (strikt größer als alle 8 Nachbarn),
2. sein Wert ≥ `threshold × I_max` des jeweiligen Bildes.

---

## Verzeichnisstruktur

```
Schichtdicke.py           ← Einstiegspunkt
image_viewer.py           ← Hauptfenster, Navigation, Tabs
image_processing.py       ← Berechnungs- und Rendering-Funktionen
analysis_timetrace.py     ← Timetrace / FCS
analysis_schichtdicke.py  ← Schichtdicke über Spot-Zählung
analysis_ics.py           ← Schichtdicke über ICS
ics_thickness.py          ← Eigenständige ICS-Pipeline (CLI + Bibliothek)
analysis_spot_zaehler.py  ← Multi-Molekül-Spot-Zähler
analysis_krueger_*.py     ← ICS, Simulation und Z-Stack aus Analyse-Krüger
img/                      ← Eingabebilder (wird beim Start automatisch gesucht)
output/                   ← PNG- und CSV-Exporte (wird bei Bedarf angelegt)
```

---

## Menü

| Eintrag | Aktion |
|---|---|
| Datei → Pfad öffnen | Öffnet den `img/`-Ordner im Dateiexplorer |
| Datei → Bilder laden | Lädt Bilder neu aus `img/` |
| Datei → Beenden | Schließt die Anwendung |
| Hilfe → Info | Zeigt Versionsinformation |
| Hilfe → About | Zeigt Autoreninformation |

---

## Lizenz

Siehe [LICENSE](LICENSE).

---

Autor: Yannik Kasprzak, Institut für Physik, Universität zu Lübeck
