import os
import glob
import whisper
import torch
import soundfile as sf
import librosa
import pandas as pd

from nemo.collections.asr.models import ASRModel
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from speechbrain.pretrained import EncoderDecoderASR
from jiwer import wer

# -------- Helper functions -------- #
def load_audio(filepath, target_sr=16000):
    audio, sr = sf.read(filepath)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return audio, sr

def load_ref(cha_file):
    if not os.path.exists(cha_file):
        return None
    text_lines = []
    with open(cha_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("*PAR:") or line.startswith("*INV:") or line.startswith("*CHI:"):
                text_lines.append(line.split(":", 1)[-1].strip())
    return " ".join(text_lines)

# -------- Whisper -------- #
def transcribe_whisper(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]

# -------- NeMo -------- #
def transcribe_nemo(audio_path):
    model = ASRModel.from_pretrained("stt_en_conformer_ctc_small")
    return model.transcribe([audio_path])[0]

# -------- HuggingFace -------- #
def transcribe_hf(audio_path):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to("cuda")
    model.eval()
    speech, sr = load_audio(audio_path)
    inputs = processor(speech, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values.to("cuda")).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.decode(predicted_ids[0])

# -------- SpeechBrain -------- #
def transcribe_speechbrain(audio_path):
    asr_model = EncoderDecoderASR.from_hparams(
        source="speechbrain/asr-transformer-transformerlm-librispeech",
        savedir="pretrained_models/asr-transformer-transformerlm-librispeech"
    )
    return asr_model.transcribe_file(audio_path)

# -------- Main loop -------- #
if __name__ == "__main__":
    data_dir = datadir
    out_dir = output dir
    os.makedirs(out_dir, exist_ok=True)

    mp4_files = sorted(glob.glob(os.path.join(data_dir, "*.mp4")))
    results = []

    for mp4 in mp4_files:
        base = os.path.splitext(os.path.basename(mp4))[0]
        cha_file = os.path.join(data_dir, f"{base}.cha")
        ref_text = load_ref(cha_file)

        print(f"\nProcessing {base}...")

        try:
            hyp_whisper = transcribe_whisper(mp4)
            hyp_nemo = transcribe_nemo(mp4)
            hyp_hf = transcribe_hf(mp4)
            hyp_sb = transcribe_speechbrain(mp4)
        except Exception as e:
            print(f"Error with {base}: {e}")
            continue

        # Save transcripts individually
        with open(os.path.join(out_dir, f"{base}_whisper.txt"), "w") as f:
            f.write(hyp_whisper)
        with open(os.path.join(out_dir, f"{base}_nemo.txt"), "w") as f:
            f.write(hyp_nemo)
        with open(os.path.join(out_dir, f"{base}_hf.txt"), "w") as f:
            f.write(hyp_hf)
        with open(os.path.join(out_dir, f"{base}_speechbrain.txt"), "w") as f:
            f.write(hyp_sb)

        # Collect results for CSV
        row = {
            "File": base,
            "Whisper": hyp_whisper,
            "NeMo": hyp_nemo,
            "HuggingFace": hyp_hf,
            "SpeechBrain": hyp_sb,
        }

        if ref_text:
            row["WER_Whisper"] = wer(ref_text, hyp_whisper)
            row["WER_NeMo"] = wer(ref_text, hyp_nemo)
            row["WER_HF"] = wer(ref_text, hyp_hf)
            row["WER_SB"] = wer(ref_text, hyp_sb)

        results.append(row)

    # Save big summary CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, "asr_benchmark_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to {csv_path}")
