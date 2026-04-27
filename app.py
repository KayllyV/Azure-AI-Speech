from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, render_template
import azure.cognitiveservices.speech as speechsdk
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import os
import uuid
import threading
import json
import base64
import subprocess
from azure.monitor.opentelemetry import configure_azure_monitor

def init_telemetry():
    connection_string = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if not connection_string:
        print("Warning: No Application Insights connection string found")
        return
    configure_azure_monitor(connection_string=connection_string)
    print("Application Insights initialized.")

init_telemetry()

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

language_client = TextAnalyticsClient(
    endpoint=os.environ.get("AZURE_LANGUAGE_ENDPOINT"),
    credential=AzureKeyCredential(os.environ.get("AZURE_LANGUAGE_KEY"))
)

def transcribe_audio(filepath: str) -> dict:
    ext = filepath.rsplit(".", 1)[-1].lower()

    speech_config = speechsdk.SpeechConfig(
        subscription=os.environ.get("AZURE_SPEECH_KEY"),
        region=os.environ.get("AZURE_SPEECH_REGION")
    )
    speech_config.request_word_level_timestamps()
    speech_config.output_format = speechsdk.OutputFormat.Detailed

    if ext in ("mp3", "ogg", "webm"):
        wav_path = filepath.rsplit(".", 1)[0] + ".wav"

        result = subprocess.run(
            ["ffmpeg", "-y", "-i", filepath, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path],
            capture_output=True,
            text=True
        )

        if result.returncode != 0 or not os.path.exists(wav_path):
            raise RuntimeError(f"ffmpeg failed: {result.stderr}")

        audio_config = speechsdk.audio.AudioConfig(filename=wav_path)
        converted_path = wav_path
    else:
        audio_config = speechsdk.audio.AudioConfig(filename=filepath)
        converted_path = None

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config
    )

    done = threading.Event()
    result_holder = {}

    def on_recognized(evt):
        result_holder["result"] = evt.result
        done.set()

    def on_canceled(evt):
        reason = evt.result.cancellation_details.reason
        if str(reason) == "CancellationReason.EndOfStream":
            done.set()
        else:
            result_holder["error"] = str(reason)
            done.set()

    recognizer.recognized.connect(on_recognized)
    recognizer.canceled.connect(on_canceled)
    recognizer.start_continuous_recognition()
    done.wait(timeout=30)
    recognizer.stop_continuous_recognition()

    if converted_path and os.path.exists(converted_path):
        os.remove(converted_path)

    if "error" in result_holder:
        raise RuntimeError(result_holder["error"])
    if "result" not in result_holder:
        raise RuntimeError("Transcription timed out — no result received within 30 s")

    result = result_holder["result"]
    detail = json.loads(result.json) if result.json else {}
    best = detail.get("NBest", [{}])[0]

    words = [
        {
            "word": w.get("Word", ""),
            "offset": round(w.get("Offset", 0) / 10_000_000, 3),
            "duration": round(w.get("Duration", 0) / 10_000_000, 3),
            "confidence": round(w.get("Confidence", 0), 4),
        }
        for w in best.get("Words", [])
    ]

    return {
        "transcript": result.text,
        "language": "en-US",
        "duration_seconds": round(detail.get("Duration", 0) / 10_000_000, 2),
        "confidence": round(best.get("Confidence", 0), 4),
        "words": words,
    }

def analyze_text(text: str) -> dict:
    documents = [text]

    key_phrases_result = language_client.extract_key_phrases(documents)
    key_phrases = (
        key_phrases_result[0].key_phrases
        if not key_phrases_result[0].is_error
        else []
    )

    ner_result = language_client.recognize_entities(documents)
    entities = [
        {
            "text": e.text,
            "category": e.category,
            "confidence": round(e.confidence_score, 4),
        }
        for e in (ner_result[0].entities if not ner_result[0].is_error else [])
    ]

    sentiment_result = language_client.analyze_sentiment(documents)
    s = sentiment_result[0]
    sentiment = {
        "label": s.sentiment,
        "scores": {
            "positive": round(s.confidence_scores.positive, 4),
            "neutral": round(s.confidence_scores.neutral, 4),
            "negative": round(s.confidence_scores.negative, 4),
        },
    }

    linked_result = language_client.recognize_linked_entities(documents)
    linked_entities = [
        {"name": e.name, "url": e.url, "data_source": e.data_source}
        for e in (linked_result[0].entities if not linked_result[0].is_error else [])
    ]

    return {
        "key_phrases": key_phrases,
        "entities": entities,
        "sentiment": sentiment,
        "linked_entities": linked_entities,
    }

def save_audio_upload(audio_file) -> tuple[str, str]:
    audio_file.seek(0, 2)
    size = audio_file.tell()
    audio_file.seek(0)

    if size > 25 * 1024 * 1024:
        raise ValueError("File too large — max 25 MB")

    name = audio_file.filename.lower()

    if name.endswith(".wav"):
        ext = "wav"
    elif name.endswith(".mp3"):
        ext = "mp3"
    elif name.endswith(".ogg"):
        ext = "ogg"
    elif name.endswith(".webm"):
        ext = "webm"
    elif name.endswith((".m4a", ".aac")):
        raise IOError("AAC/M4A not supported — please convert to WAV first")
    else:
        raise IOError("Unsupported format — use WAV, MP3, OGG, or WEBM")

    os.makedirs("temp_audio", exist_ok=True)
    filepath = f"temp_audio/{uuid.uuid4()}.{ext}"
    audio_file.save(filepath)
    return filepath, ext

def build_summary(analysis_result):
    key_phrases = analysis_result.get("key_phrases", [])
    entities = analysis_result.get("entities", [])
    sentiment = analysis_result.get("sentiment", {}).get("label", "unknown")

    key_phrase_count = len(key_phrases)
    entity_count = len(entities)

    if key_phrases:
        topics_text = ", ".join(key_phrases[:3])
    else:
        topics_text = "no major topics"

    entity_categories = {}
    for entity in entities:
        category = entity.get("category", "Unknown")
        entity_categories[category] = entity_categories.get(category, 0) + 1

    if entity_categories:
        entity_parts = [f"{count} {category}" for category, count in entity_categories.items()]
        entity_text = ", ".join(entity_parts)
    else:
        entity_text = "no named entities"

    return (
        f"Hey! Your memo mentions {key_phrase_count} key topic"
        f"{'s' if key_phrase_count != 1 else ''}: {topics_text}. "
        f"The overall tone is {sentiment}. "
        f"I also detected {entity_count} named entit"
        f"{'ies' if entity_count != 1 else 'y'}, including {entity_text}."
    )

def synthesize_summary(summary_text: str) -> dict:
    speech_config = speechsdk.SpeechConfig(
        subscription=os.environ.get("AZURE_SPEECH_KEY"),
        region=os.environ.get("AZURE_SPEECH_REGION")
    )

    speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
    )

    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config,
        audio_config=None
    )

    result = synthesizer.speak_text_async(summary_text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        audio_base64 = base64.b64encode(result.audio_data).decode("utf-8")
        return {
            "summary_text": summary_text,
            "audio_base64": audio_base64,
            "char_count": len(summary_text),
            "voice": "en-US-JennyNeural"
        }

    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation = result.cancellation_details
        raise RuntimeError(
            f"TTS canceled: {cancellation.reason} - {cancellation.error_details}"
        )

    else:
        raise RuntimeError("TTS failed for an unknown reason")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio_file = request.files.get("audio")
    if not audio_file:
        return jsonify({"error": "No audio file provided"}), 400

    try:
        filepath, _ = save_audio_upload(audio_file)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except IOError as e:
        return jsonify({"error": str(e)}), 415

    try:
        result = transcribe_audio(filepath)
        return jsonify(result)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Request body must include a 'text' field"}), 400

    try:
        result = analyze_text(data["text"])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/process", methods=["POST"])
def process():
    audio_file = request.files.get("audio")
    if not audio_file:
        return jsonify({"error": "No audio file provided"}), 400

    try:
        filepath, _ = save_audio_upload(audio_file)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except IOError as e:
        return jsonify({"error": str(e)}), 415

    try:
        stt_result = transcribe_audio(filepath)
        language_result = analyze_text(stt_result["transcript"])
        summary = build_summary(language_result)
        tts_result = synthesize_summary(summary)

        return jsonify({
            **stt_result,
            **language_result,
            "summary": summary,
            "tts": tts_result
        })

    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route("/telemetry-summary", methods=["GET"])
def telemetry_summary():
    return jsonify({
        "status": "ok",
        "message": "Telemetry is being collected in Application Insights.",
        "tracked_metrics": [
            "stt_confidence",
            "stage.speech_to_text.latency_ms",
            "stage.language_analysis.latency_ms",
            "stage.text_to_speech.latency_ms"
        ],
        "tracked_spans": [
            "stage.speech_to_text",
            "stage.language_analysis",
            "stage.text_to_speech"
        ]
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)