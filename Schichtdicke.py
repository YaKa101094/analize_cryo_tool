"""
Schichtdicke – Photon-Microscopy Image Browser & Analysis Tool
==============================================================
Einstiegspunkt der Anwendung. Startet direkt den ImageViewer im
Haupt-Notebook — alle Werkzeuge (Bildanalyse, Timetrace, Schichtdicke
Zählen/ICS) sind von Anfang an als Tabs vorhanden, kein separater
Auswahlbildschirm mehr nötig.

Modulstruktur
-------------
  Schichtdicke.py        – dieser Einstiegspunkt
  image_viewer.py        – ImageViewer-Hauptklasse (Bildbetrachter, Navigation, Tabs)
  image_processing.py    – Reine Berechnungs- und Rendering-Funktionen
  analysis_timetrace.py  – TimeraceDialog (FCS-Korrelation)
  analysis_schichtdicke.py – SchichtdickeDialog (Spot-Zählmethode)
  analysis_ics.py        – ICSDialog (Image Correlation Spectroscopy)
  ics_thickness.py        – Eigenständige, strenge ICS-Pipeline (CLI + Bibliothek)
  analysis_spot_zaehler.py – SpotZaehlerDialog (Multi-Molekül-Spot-Zähler,
                             portiert aus Spot_Zaehler_v7_patched.ipynb)

Autor: Yannik Kasprzak, Institut für Physik, Universität zu Lübeck
"""

import tkinter as tk

from image_viewer import ImageViewer

if __name__ == "__main__":
    root = tk.Tk()
    ImageViewer(root)
    root.mainloop()
