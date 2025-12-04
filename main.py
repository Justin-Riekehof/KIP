import os
import sys
import queue
import wave
from datetime import datetime

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

import torch
from pyannote.audio.core.task import Specifications
torch.serialization.add_safe_globals([Specifications])


RECORDINGS_DIR = "recordings"
EXPORTS_DIR = "exports"
os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs(EXPORTS_DIR, exist_ok=True)


def record_audio(filename="meeting.wav", samplerate=16000, channels=1):
    """
    Nimmt Audio vom Mikrofon auf, bis STRG+C gedrückt wird.
    Speichert als WAV in ./recordings.
    """
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
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(samplerate)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())

    print(f"Audio gespeichert unter: {filepath}")
    return filepath


def transcribe_whisper(audio_path, model_size="small", language="de", device="cpu"):
    """
    Transkription mit faster-whisper.
    Gibt eine Liste von Segmenten zurück: [{start, end, text}, ...]
    """
    print("Lade Whisper-Modell...")
    model = WhisperModel(model_size, device=device)

    print("Starte Transkription...")
    segments, info = model.transcribe(audio_path, language=language)
    result = []
    for seg in segments:
        result.append(
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
            }
        )

    print("Transkription abgeschlossen.")
    return result


def diarize_speakers(audio_path):
    """
    Sprecherdiarisierung mit pyannote.audio 4.x (community pipeline).
    Nutzt: pyannote/speaker-diarization-community-1
    Erwartet, dass HUGGINGFACE_TOKEN gesetzt ist.
    Wir laden das Audio manuell und geben ein Dict mit
    {'waveform': Tensor (channels, time), 'sample_rate': int}
    an die Pipeline, um torchcodec/AudioDecoder zu umgehen.
    """
    token = os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("HUGGINGFACE_TOKEN ist nicht gesetzt.")

    # --- Fix für PyTorch 2.6+: pyannote-Checkpoint-Klassen erlauben ---
    import torch
    from pyannote.audio.core.task import Specifications, Problem, Resolution
    torch.serialization.add_safe_globals([Specifications, Problem, Resolution])
    # ------------------------------------------------------------------

    print("Lade pyannote-Pipeline (community-1)...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=token,
    )

    # --- WAV manuell laden (ohne torchcodec/AudioDecoder) ---
    with wave.open(audio_path, "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        n_frames = wf.getnframes()
        audio_bytes = wf.readframes(n_frames)

    # int16 -> float32, normalisiert auf [-1, 1]
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0

    if n_channels > 1:
        # (time * channels) -> (frames, channels) -> (channels, frames)
        audio_np = audio_np.reshape(-1, n_channels).T
        # hier: einfach nur den ersten Kanal verwenden
        audio_np = audio_np[0:1, :]
    else:
        # mono: (time,) -> (1, time)
        audio_np = audio_np[None, :]

    waveform = torch.from_numpy(audio_np)  # Shape: (channels, time)

    # --- Diarisierung mit vor-geladenem Audio ---
    print("Starte Diarisierung...")
    output = pipeline({"waveform": waveform, "sample_rate": sample_rate})

    diarization = output.speaker_diarization

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": speaker,
            }
        )

    print("Diarisierung abgeschlossen.")
    return segments

def match_speakers_with_text(transcript_segments, speaker_segments):
    """
    Für jedes Transkriptsegment den am besten überlappenden Sprecher finden.
    """
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

        if best_speaker is None:
            best_speaker = "UNKNOWN"

        result.append(
            {
                "speaker": best_speaker,
                "text": t["text"],
            }
        )

    return result


def normalize_speaker_labels(speaker_text_sequence):
    """
    Mappt pyannote-Speaker-Labels (z.B. SPEAKER_00) auf Teilnehmer 1, 2, ...
    """
    mapping = {}
    next_id = 1
    normalized = []

    for item in speaker_text_sequence:
        spk = item["speaker"]
        if spk not in mapping:
            mapping[spk] = f"Teilnehmer {next_id}"
            next_id += 1
        normalized.append(
            {
                "speaker": mapping[spk],
                "text": item["text"],
            }
        )

    return normalized


def export_minutes(speaker_text_sequence, filename_prefix="meeting_protokoll"):
    """
    Exportiert ein Protokoll in ./exports/<prefix>_YYYYmmdd_HHMMSS.txt
    Gruppiert zusammenhängende Blöcke pro Sprecher.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(EXPORTS_DIR, f"{filename_prefix}_{timestamp}.txt")

    lines = []
    current_speaker = None
    current_block = []

    def flush_block():
        if current_speaker is not None and current_block:
            lines.append(f"{current_speaker}:")
            lines.append('"' + " ".join(current_block).strip() + '"')
            lines.append("")

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

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Protokoll exportiert nach: {filepath}")
    return filepath


def main():
    # 1. Audio aufnehmen
    audio_path = record_audio()

    # 2. Transkribieren
    transcript_segments = transcribe_whisper(audio_path)

    # 3. Sprecherdiarisierung
    speaker_segments = diarize_speakers(audio_path)

    # 4. Matching + Normalisierung
    combined = match_speakers_with_text(transcript_segments, speaker_segments)
    normalized = normalize_speaker_labels(combined)

    # 5. Export
    export_minutes(normalized)


if __name__ == "__main__":
    main()
