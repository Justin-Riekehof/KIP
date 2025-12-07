# debug_analyse.py
import os
import sys

# Wir importieren die bestehenden Funktionen & Konstanten aus deiner main.py
from main import (
    transcribe_whisper,
    diarize_speakers,
    match_speakers_with_text,
    normalize_speaker_labels,
    export_minutes_pdf,
    RECORDINGS_DIR,
)

def debug_analyse_audio(audio_filename=None):
    """
    Analysiert eine bestehende Audiodatei ohne GUI/Recording.
    - Führt Whisper-Transkription aus
    - Führt pyannote-Diarisierung aus
    - Matched Sprecher & Text
    - Normalisiert Sprecherlabels
    - Optional: Export als PDF
    """

    # 1) Dateiname bestimmen
    if audio_filename is None:
        # HIER kannst du einen Standard-Dateinamen hinterlegen:
        audio_filename = "A1_band_audition.wav"

    # Wenn der Pfad relativ ist, im RECORDINGS_DIR suchen
    if not os.path.isabs(audio_filename):
        audio_path = os.path.join(RECORDINGS_DIR, audio_filename)
    else:
        audio_path = audio_filename

    if not os.path.exists(audio_path):
        print(f"[FEHLER] Audiodatei nicht gefunden: {audio_path}")
        return

    print(f"== DEBUG-ANALYSE FÜR: {audio_path} ==")

    # 2) Transkription
    print("\n[1/4] Starte Whisper-Transkription...\n")
    transcript_segments = transcribe_whisper(audio_path)
    print("Transkript-Segmente:")
    for i, seg in enumerate(transcript_segments, start=1):
        print(f"  {i:03d} | {seg['start']:.2f}s - {seg['end']:.2f}s | {seg['text']}")

    # 3) Diarisierung
    print("\n[2/4] Starte pyannote-Diarisierung...\n")
    speaker_segments = diarize_speakers(audio_path)
    print("Sprecher-Segmente (rohe pyannote-Ausgabe):")
    for i, seg in enumerate(speaker_segments, start=1):
        print(
            f"  {i:03d} | {seg['start']:.2f}s - {seg['end']:.2f}s | Speaker-ID: {seg['speaker']}"
        )

    # 4) Matching Text <-> Sprecher
    print("\n[3/4] Matche Transkript-Segmente mit Sprecher-Segmenten...\n")
    combined = match_speakers_with_text(transcript_segments, speaker_segments)
    print("Gematchte Segmente (pyannote-Labels):")
    for i, item in enumerate(combined, start=1):
        print(f"  {i:03d} | {item['speaker']} | {item['text']}")

    # 5) Normalisierte Sprecherlabels
    print("\n[4/4] Normalisiere Sprecherlabels (Teilnehmer 1, Teilnehmer 2, ...)...\n")
    normalized = normalize_speaker_labels(combined)
    print("Normalisierte Segmente:")
    for i, item in enumerate(normalized, start=1):
        print(f"  {i:03d} | {item['speaker']} | {item['text']}")

    # 6) Optional: PDF-Export
    print("\n[OPTIONAL] Erzeuge Test-PDF mit dem Protokoll...")
    pdf_path = export_minutes_pdf(
        normalized,
        filename_prefix="DEBUG_protokoll",
        document_title="Debug-Protokoll",
    )
    print(f"PDF exportiert nach: {pdf_path}")

    print("\n== DEBUG-ANALYSE ABGESCHLOSSEN ==\n")


if __name__ == "__main__":
    # Nutzung:
    #   python debug_analyse.py                -> nutzt Standarddatei (gui_meeting.wav)
    #   python debug_analyse.py meine.wav     -> nutzt übergebenen Dateinamen
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = None

    debug_analyse_audio(filename)
