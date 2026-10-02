#!/usr/bin/env python3
"""
Local Gradio UI for Chatterbox voice cloning.

Two modes share one reference voice:

  Convert   -- re-speak an existing audio clip in the reference voice
               (speech-to-speech, the chatterbox_vc pipeline).
  Speak     -- synthesize typed text in the reference voice
               (zero-shot TTS, via chatterbox.tts).

Both models load lazily on first use, so startup is instant and the
~1 GB TTS checkpoint is only downloaded if you actually open that tab.

Usage:
    .venv/bin/python app.py
    .venv/bin/python app.py --share       # temporary public link
"""

import argparse
import tempfile

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import torch

VC_SR = 24000
# The model reads at most 10s of reference audio (DEC_COND_LEN at 24kHz).
REF_MAX_SAMPLES = 240000


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device()

_vc = None
_tts = None


def get_vc():
    """Load the voice-conversion model once, on first use."""
    global _vc
    if _vc is None:
        from chatterbox.vc import ChatterboxVC

        _vc = ChatterboxVC.from_pretrained(DEVICE)
    return _vc


def get_tts():
    """Load the TTS model once, on first use (~1 GB download initially)."""
    global _tts
    if _tts is None:
        from chatterbox.tts import ChatterboxTTS

        _tts = ChatterboxTTS.from_pretrained(DEVICE)
    return _tts


def speaker_embedding(path: str) -> np.ndarray:
    """CAMPPlus x-vector for a clip, used to score speaker similarity."""
    model = get_vc()
    wav, _ = librosa.load(path, sr=VC_SR)
    ref = model.s3gen.embed_ref(wav[:REF_MAX_SAMPLES], VC_SR, device=DEVICE)
    return ref["embedding"].squeeze().detach().cpu().numpy()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def guard_clipping(wav: np.ndarray) -> np.ndarray:
    """Scale back output that exceeds full scale, which would clip on save."""
    peak = np.abs(wav).max()
    if peak > 1.0:
        wav = wav / peak * 0.99
    return wav


def similarity_note(score: float) -> str:
    if score >= 0.80:
        return "strong match"
    if score >= 0.60:
        return "recognizable"
    return "weak -- try a cleaner reference"


def check_reference(path):
    """Report duration and flag references likely to give poor results."""
    if not path:
        return ""
    info = sf.info(path)
    msgs = [f"**{info.duration:.1f}s**, {info.samplerate} Hz, {info.channels} ch"]
    if info.duration < 3:
        msgs.append("Too short -- 6-10s gives a far more stable voice.")
    elif info.duration > 10:
        msgs.append("Only the first 10s is used; the rest is ignored.")
    return "  \n".join(msgs)


def convert(reference, source, takes, progress=gr.Progress()):
    """Re-speak `source` in the voice of `reference`, keeping the best take."""
    if not reference:
        raise gr.Error("Add a reference voice first.")
    if not source:
        raise gr.Error("Add the audio you want re-spoken.")

    progress(0, desc="Loading model")
    model = get_vc()
    model.set_target_voice(reference)
    ref_emb = speaker_embedding(reference)

    best_wav, best_score, scores = None, -1.0, []
    for i in range(int(takes)):
        progress(i / takes, desc=f"Take {i + 1} of {int(takes)}")
        wav = model.generate(audio=source).squeeze().cpu().numpy().astype(np.float32)
        wav = guard_clipping(wav)

        # Score this take by writing it out -- the encoder reads from disk.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, wav, VC_SR)
            score = cosine(ref_emb, speaker_embedding(tmp.name))

        scores.append(score)
        if score > best_score:
            best_wav, best_score = wav, score

    report = [
        f"### Similarity to your reference: **{best_score:.3f}** ({similarity_note(best_score)})",
        "",
        "Scored with the model's own speaker encoder. Above 0.80 is a strong "
        "match, 0.60-0.80 recognizable, below 0.50 a different speaker.",
    ]
    if len(scores) > 1:
        ranked = "  ".join(f"`{s:.3f}`" for s in scores)
        report += ["", f"All {len(scores)} takes: {ranked} -- returning the best."]

    return (VC_SR, best_wav), "\n".join(report)


def speak(reference, text, exaggeration, cfg_weight, temperature, progress=gr.Progress()):
    """Synthesize `text` in the voice of `reference`."""
    if not reference:
        raise gr.Error("Add a reference voice first.")
    if not text or not text.strip():
        raise gr.Error("Type something for the voice to say.")

    progress(0, desc="Loading TTS model (first run downloads ~1 GB)")
    model = get_tts()

    progress(0.5, desc="Synthesizing")
    wav = model.generate(
        text.strip(),
        audio_prompt_path=reference,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=temperature,
    )
    wav = guard_clipping(wav.squeeze().cpu().numpy().astype(np.float32))

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, wav, model.sr)
        score = cosine(speaker_embedding(reference), speaker_embedding(tmp.name))

    return (model.sr, wav), (
        f"### Similarity to your reference: **{score:.3f}** ({similarity_note(score)})"
    )


def build_ui():
    with gr.Blocks(title="Voice Cloning") as demo:
        gr.Markdown(
            f"""
            # Voice Cloning
            Give it a reference voice, then either re-speak an existing clip in
            that voice or type something new for it to say.
            Running on **{DEVICE}**.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Reference voice")
                reference = gr.Audio(
                    label="The voice to clone",
                    sources=["upload", "microphone"],
                    type="filepath",
                )
                ref_info = gr.Markdown()
                gr.Markdown(
                    "6-10 seconds of clean speech, one speaker, no music or "
                    "background noise. Varied words beat one repeated phrase. "
                    "Only the first 10s is used."
                )

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("Convert existing audio"):
                        source = gr.Audio(
                            label="Audio to re-speak (its words and timing are kept)",
                            sources=["upload", "microphone"],
                            type="filepath",
                        )
                        takes = gr.Slider(
                            1, 8, value=3, step=1,
                            label="Takes to generate",
                            info="Each take is a fresh draw; the closest match to "
                                 "your reference is returned. More takes, better "
                                 "odds, proportionally longer.",
                        )
                        convert_btn = gr.Button("Convert", variant="primary")
                        convert_out = gr.Audio(label="Result", type="numpy")
                        convert_report = gr.Markdown()

                    with gr.Tab("Speak typed text"):
                        text = gr.Textbox(
                            label="What should the voice say?",
                            placeholder="Type a sentence or two...",
                            lines=4,
                        )
                        with gr.Accordion("Generation settings", open=False):
                            exaggeration = gr.Slider(
                                0.0, 1.0, value=0.5, step=0.05,
                                label="Expressiveness",
                                info="Higher is more animated, lower is flatter.",
                            )
                            cfg_weight = gr.Slider(
                                0.0, 1.0, value=0.5, step=0.05,
                                label="Guidance",
                                info="Higher sticks closer to the reference voice.",
                            )
                            temperature = gr.Slider(
                                0.1, 1.5, value=0.8, step=0.05,
                                label="Temperature",
                                info="Higher is more varied, lower more predictable.",
                            )
                        speak_btn = gr.Button("Generate", variant="primary")
                        speak_out = gr.Audio(label="Result", type="numpy")
                        speak_report = gr.Markdown()
                        gr.Markdown(
                            "_First use downloads the TTS model (~1 GB). "
                            "Later runs start immediately._"
                        )

        reference.change(check_reference, reference, ref_info)
        convert_btn.click(
            convert, [reference, source, takes], [convert_out, convert_report]
        )
        speak_btn.click(
            speak,
            [reference, text, exaggeration, cfg_weight, temperature],
            [speak_out, speak_report],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--share", action="store_true", help="create a public link")
    parser.add_argument(
        "--port", type=int, default=None,
        help="default: first free port from 7860 upward",
    )
    args = parser.parse_args()

    build_ui().launch(
        share=args.share, server_port=args.port, theme=gr.themes.Soft()
    )
