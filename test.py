import os
import sys
import queue
import wave
import threading
from datetime import datetime

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

import torch
from pyannote.audio.core.task import Specifications, Problem, Resolution
torch.serialization.add_safe_globals([Specifications, Problem, Resolution])

# PDF-Export
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# GUI
import tkinter as tk
from tkinter import ttk


RECORDINGS_DIR = "recordings"
EXPORTS_DIR = "exports"
os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs(EXPORTS_DIR, exist_ok=True)


# =========================
# AUDIO AUFNAHME (CLI)
# =========================
def record_audio(filename="meeting.wav", samplerate=16000, channels=1):
    """Blockierende Aufnahme für CLI-Modus (Strg+C zum Stoppen)."""
    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    filepath = os.path.join(RECORDINGS_DIR, filename)
    print("Aufnahme gestartet. Drücke STRG+C zum Stoppen...")

    frames = []
    with sd.InputStream(samplerate=samplerate, channels=channels, callback=callback):
        try:
            while True:
                frames.append(q.get())
        except KeyboardInterrupt:
            print("\nAufnahme beendet.")

    audio = np.concatenate(frames, axis=0)

    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())

    print(f"Audio gespeichert unter: {filepath}")
    return filepath


# =========================
# AUDIO AUFNAHME (GUI, threaded)
# =========================
recording_stop_flag = False  # wird von GUI gesetzt


def record_audio_to_path(filepath, samplerate=16000, channels=1):
    """Nicht-blockierende Aufnahme, beendet über recording_stop_flag."""
    global recording_stop_flag
    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    frames = []

    with sd.InputStream(samplerate=samplerate, channels=channels, callback=callback):
        while not recording_stop_flag:
            try:
                frames.append(q.get(timeout=0.1))
            except queue.Empty:
                pass

    if not frames:
        # Nichts aufgenommen
        return None

    audio = np.concatenate(frames, axis=0)

    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())

    return filepath


# =========================
# WHISPER TRANSKRIPTION
# =========================
def transcribe_whisper(audio_path, model_size="small", language="de", device="cpu"):
    print("Lade Whisper-Modell...")
    model = WhisperModel(model_size, device=device)

    print("Starte Transkription...")
    segments, info = model.transcribe(audio_path, language=language)

    result = [
        {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
        for seg in segments
    ]

    print("Transkription abgeschlossen.")
    return result


# =========================
# PYANNOTE DIARISIERUNG
# =========================
def diarize_speakers(audio_path):
    token = os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("HUGGINGFACE_TOKEN ist nicht gesetzt.")

    print("Lade pyannote-Pipeline (community-1)...")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=token,
    )

    # WAV manuell laden
    with wave.open(audio_path, "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        n_frames = wf.getnframes()
        audio_bytes = wf.readframes(n_frames)

    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0

    if n_channels > 1:
        audio_np = audio_np.reshape(-1, n_channels).T
        audio_np = audio_np[0:1, :]
    else:
        audio_np = audio_np[None, :]

    waveform = torch.from_numpy(audio_np)

    print("Starte Diarisierung...")
    output = pipeline({"waveform": waveform, "sample_rate": sample_rate})

    diarization = output.speaker_diarization

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": speaker,
        })

    print("Diarisierung abgeschlossen.")
    return segments


# =========================
# MATCHING TRANSKRIPT + SPRECHER
# =========================
def match_speakers_with_text(transcript_segments, speaker_segments):
    result = []

    for t in transcript_segments:
        t_start, t_end = t["start"], t["end"]
        best_speaker = None
        best_overlap = 0.0

        for s in speaker_segments:
            s_start, s_end = s["start"], s["end"]
            overlap = max(0.0, min(t_end, s_end) - max(t_start, s_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = s["speaker"]

        result.append({"speaker": best_speaker or "UNKNOWN", "text": t["text"]})

    return result


# =========================
# SPRECHER UMBENENNEN
# =========================
def normalize_speaker_labels(speaker_text_sequence):
    mapping = {}
    next_id = 1
    normalized = []

    for item in speaker_text_sequence:
        spk = item["speaker"]
        if spk not in mapping:
            mapping[spk] = f"Teilnehmer {next_id}"
            next_id += 1

        normalized.append({
            "speaker": mapping[spk],
            "text": item["text"],
        })

    return normalized


# =========================
# PDF EXPORT
# =========================
def export_minutes_pdf(
    speaker_text_sequence,
    filename_prefix="meeting_protokoll",
    document_title=None,
):
    """
    Exportiert ein PDF.

    - Dateiname: <filename_prefix>_YYYYMMDD_HHMMSS.pdf
    - Dokumenttitel & Überschrift:
        * wenn document_title gesetzt -> nur dieser Titel
        * sonst -> kompletter Dateititel ohne .pdf
    - Datum unten links, Seitenzahl unten rechts.
    """
    now = datetime.now()
    ts_str = now.strftime("%Y%m%d_%H%M%S")
    date_str = now.strftime("%Y-%m-%d")

    file_title = f"{filename_prefix}_{ts_str}"
    filename = f"{file_title}.pdf"
    filepath = os.path.join(EXPORTS_DIR, filename)

    if document_title:
        visible_title = document_title
        pdf_meta_title = document_title
    else:
        visible_title = file_title
        pdf_meta_title = file_title

    # Schriftart registrieren (optional)
    try:
        pdfmetrics.registerFont(TTFont("Roboto", "Roboto-Regular.ttf"))
        base_font = "Roboto"
    except Exception:
        base_font = "Helvetica"

    doc = SimpleDocTemplate(
        filepath,
        pagesize=A4,
        title=pdf_meta_title,  # Dokumenttitel in den PDF-Metadaten
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontName=base_font,
        fontSize=22,
        textColor=colors.HexColor("#333333"),
        spaceAfter=20,
    )

    speaker_style = ParagraphStyle(
        "Speaker",
        parent=styles["Heading4"],
        fontName=base_font,
        fontSize=14,
        textColor=colors.HexColor("#0077CC"),
        spaceBefore=12,
        spaceAfter=4,
    )

    text_style = ParagraphStyle(
        "Text",
        parent=styles["BodyText"],
        fontName=base_font,
        fontSize=12,
        leading=15,
    )

    story = []
    story.append(Paragraph(visible_title, title_style))
    story.append(Spacer(1, 12))

    current_speaker = None
    current_block = []

    def flush_block():
        if current_speaker and current_block:
            story.append(Paragraph(current_speaker, speaker_style))
            story.append(Paragraph(" ".join(current_block), text_style))
            story.append(Spacer(1, 6))

    for item in speaker_text_sequence:
        spk = item["speaker"]
        txt = item["text"]

        if spk != current_speaker:
            flush_block()
            current_speaker = spk
            current_block = [txt]
        else:
            current_block.append(txt)

    flush_block()

    def add_footer(canvas, doc_obj, base_font=base_font, date_text=date_str):
        canvas.setFont(base_font, 9)
        # Datum unten links
        canvas.drawString(2 * cm, 1.5 * cm, date_text)
        # Seite unten rechts
        canvas.drawRightString(A4[0] - 2 * cm, 1.5 * cm, f"Seite {doc_obj.page}")

    doc.build(story, onLaterPages=add_footer, onFirstPage=add_footer)

    print(f"PDF exportiert nach: {filepath}")
    return filepath


# =========================
# VERARBEITUNG EINER AUDIODATEI (CLI)
# =========================
def process_audio_file(audio_path, filename_prefix="meeting_protokoll", document_title=None):
    transcript_segments = transcribe_whisper(audio_path)
    speaker_segments = diarize_speakers(audio_path)

    combined = match_speakers_with_text(transcript_segments, speaker_segments)
    normalized = normalize_speaker_labels(combined)

    pdf_path = export_minutes_pdf(
        normalized,
        filename_prefix=filename_prefix,
        document_title=document_title,
    )
    return pdf_path


# =========================
# CLI MAIN
# =========================
def run_cli():
    audio_path = record_audio()
    filename_prefix = "meeting_protokoll"
    process_audio_file(audio_path, filename_prefix=filename_prefix, document_title=None)


# =========================
# TKINTER GUI
# =========================
def run_gui():
    root = tk.Tk()
    root.title("Meeting-Protokollierung")
    root.geometry("520x280")
    root.minsize(500, 260)

    # ttk-Style (etwas moderner)
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure("TLabel", font=("Segoe UI", 10))
    style.configure("TButton", font=("Segoe UI", 10), padding=5)
    style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"))

    main_frame = ttk.Frame(root, padding=15)
    main_frame.pack(fill="both", expand=True)

    title_label = ttk.Label(main_frame, text="Titel des Protokolls (optional):", style="Header.TLabel")
    title_label.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 5))

    title_entry = ttk.Entry(main_frame, width=45)
    title_entry.grid(row=1, column=0, columnspan=2, sticky="we", pady=(0, 10))

    status_label = ttk.Label(main_frame, text="Bereit.")
    status_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 8))

    start_button = ttk.Button(main_frame, text="Aufnahme starten")
    stop_button = ttk.Button(main_frame, text="Aufnahme stoppen & Protokoll erstellen", state=tk.DISABLED)

    start_button.grid(row=3, column=0, sticky="we", padx=(0, 5), pady=3)
    stop_button.grid(row=3, column=1, sticky="we", padx=(5, 0), pady=3)

    progress_label = ttk.Label(main_frame, text="Verarbeitung:", foreground="#555555")
    progress_label.grid(row=4, column=0, columnspan=2, sticky="w", pady=(10, 2))

    progress_bar = ttk.Progressbar(main_frame, orient="horizontal", mode="determinate", maximum=100, length=400)
    progress_bar.grid(row=5, column=0, columnspan=2, sticky="we")

    result_label = ttk.Label(main_frame, text="", foreground="blue")
    result_label.grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 0))

    main_frame.columnconfigure(0, weight=1)
    main_frame.columnconfigure(1, weight=1)

    # State
    gui_state = {
        "recording_thread": None,
        "audio_path": None,
        "filename_prefix": None,
        "document_title": None,
    }

    def set_status(text, color="black"):
        root.after(0, lambda: status_label.config(text=text, foreground=color))

    def set_progress(value):
        root.after(0, lambda: progress_bar.config(value=value))

    def set_result(text, color="blue"):
        root.after(0, lambda: result_label.config(text=text, foreground=color))

    def enable_buttons(start_enabled, stop_enabled):
        def _set():
            start_button.config(state=tk.NORMAL if start_enabled else tk.DISABLED)
            stop_button.config(state=tk.NORMAL if stop_enabled else tk.DISABLED)
        root.after(0, _set)

    def start_recording():
        set_result("", "blue")
        set_progress(0)

        raw_title = title_entry.get().strip()
        if raw_title:
            # User hat einen Titel eingegeben
            user_title = raw_title
            filename_prefix = raw_title
        else:
            # Kein Titel -> Standard
            user_title = None
            filename_prefix = "meeting_protokoll"

        gui_state["filename_prefix"] = filename_prefix
        gui_state["document_title"] = user_title

        audio_filename = "gui_meeting.wav"
        audio_path = os.path.join(RECORDINGS_DIR, audio_filename)
        gui_state["audio_path"] = audio_path

        global recording_stop_flag
        recording_stop_flag = False

        set_status("Aufnahme läuft...", "green")
        enable_buttons(False, True)

        def worker_rec():
            record_audio_to_path(audio_path)

        t = threading.Thread(target=worker_rec, daemon=True)
        gui_state["recording_thread"] = t
        t.start()

    def stop_and_process():
        global recording_stop_flag
        recording_stop_flag = True

        set_status("Beende Aufnahme...", "orange")
        enable_buttons(False, False)

        def worker():
            try:
                t = gui_state.get("recording_thread")
                if t is not None:
                    t.join()

                audio_path = gui_state.get("audio_path")
                if not audio_path or not os.path.exists(audio_path):
                    raise RuntimeError("Keine Audiodatei gefunden – war das Mikro stumm?")

                set_status("Verarbeite Audio (Transkription & Diarisierung)...", "orange")
                set_progress(5)

                # Verarbeitung mit Fortschritts-Updates
                # 1) Transkription
                set_progress(15)
                transcript_segments = transcribe_whisper(audio_path)
                set_progress(45)

                # 2) Diarisierung
                set_status("Diarisierung läuft...", "orange")
                set_progress(55)
                speaker_segments = diarize_speakers(audio_path)
                set_progress(75)

                # 3) Matching & PDF
                set_status("Erzeuge Protokoll-PDF...", "orange")
                combined = match_speakers_with_text(transcript_segments, speaker_segments)
                normalized = normalize_speaker_labels(combined)

                filename_prefix = gui_state.get("filename_prefix") or "meeting_protokoll"
                document_title = gui_state.get("document_title")

                pdf_path = export_minutes_pdf(
                    normalized,
                    filename_prefix=filename_prefix,
                    document_title=document_title,
                )
                set_progress(100)

                set_status("Fertig.", "blue")
                set_result(f"Protokoll erstellt: {os.path.basename(pdf_path)}", "blue")
            except Exception as e:
                set_status("Fehler aufgetreten.", "red")
                set_result(f"Fehler: {e}", "red")
            finally:
                enable_buttons(True, False)

        threading.Thread(target=worker, daemon=True).start()

    start_button.config(command=start_recording)
    stop_button.config(command=stop_and_process)

    root.mainloop()


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    # CLI-Modus explizit: python main.py cli
    if len(sys.argv) > 1 and sys.argv[1].lower() == "cli":
        run_cli()
    else:
        run_gui()
