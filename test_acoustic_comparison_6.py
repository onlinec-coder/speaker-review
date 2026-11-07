import os
import json
import time
from pathlib import Path
from google import genai
from google.genai import types

PROJECT = Path(r"C:\Users\chinzorig\SRT-Translator")
DESKTOP = Path(r"C:\Users\chinzorig\Desktop")

AUDIO_FILES = [
    {
        "name": "normal_original",
        "path": DESKTOP / "Husbands_test_01_normal_original.wav",
        "start": "00:00:00",
        "cues": {
            66: "00:00.458 - 00:02.000",
            67: "00:02.000 - 00:03.666",
            68: "00:05.300 - 00:06.500",
            69: "00:06.900 - 00:08.300",
            70: "00:08.300 - 00:09.900",
        },
    },
    {
        "name": "normal_speech",
        "path": DESKTOP / "Husbands_test_01_normal_speech.wav",
        "start": "00:00:00",
        "cues": {
            66: "00:00.458 - 00:02.000",
            67: "00:02.000 - 00:03.666",
            68: "00:05.300 - 00:06.500",
            69: "00:06.900 - 00:08.300",
            70: "00:08.300 - 00:09.900",
        },
    },
    {
        "name": "emotional_original",
        "path": DESKTOP / "Husbands_test_02_emotional_original.wav",
        "start": "00:00:00",
        "cues": {
            698: "00:00.400 - 00:01.958",
            699: "00:02.000 - 00:02.958",
            700: "00:03.200 - 00:04.166",
            701: "00:05.200 - 00:06.500",
            702: "00:06.600 - 00:07.749",
        },
    },
    {
        "name": "emotional_speech",
        "path": DESKTOP / "Husbands_test_02_emotional_speech.wav",
        "start": "00:00:00",
        "cues": {
            698: "00:00.400 - 00:01.958",
            699: "00:02.000 - 00:02.958",
            700: "00:03.200 - 00:04.166",
            701: "00:05.200 - 00:06.500",
            702: "00:06.600 - 00:07.749",
        },
    },
    {
        "name": "quiet_original",
        "path": DESKTOP / "Husbands_test_03_quiet_original.wav",
        "start": "00:00:00",
        "cues": {
            9: "00:01.000 - 00:02.125",
            10: "00:02.583 - 00:03.600",
            11: "00:05.374 - 00:06.749",
            12: "00:06.800 - 00:07.541",
            13: "00:08.541 - 00:09.800",
            15: "00:12.600 - 00:13.500",
            16: "00:13.600 - 00:14.958",
            17: "00:15.000 - 00:16.374",
            18: "00:16.500 - 00:18.166",
            19: "00:18.200 - 00:20.000",
        },
    },
    {
        "name": "quiet_speech",
        "path": DESKTOP / "Husbands_test_03_quiet_speech.wav",
        "start": "00:00:00",
        "cues": {
            9: "00:01.000 - 00:02.125",
            10: "00:02.583 - 00:03.600",
            11: "00:05.374 - 00:06.749",
            12: "00:06.800 - 00:07.541",
            13: "00:08.541 - 00:09.800",
            15: "00:12.600 - 00:13.500",
            16: "00:13.600 - 00:14.958",
            17: "00:15.000 - 00:16.374",
            18: "00:16.500 - 00:18.166",
            19: "00:18.200 - 00:20.000",
        },
    },
]

OUTPUT_DIR = PROJECT / "acoustic_comparison_6_results_v2"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load API key without displaying it.
env_path = PROJECT / ".env"

if not env_path.exists():
    raise FileNotFoundError(f".env файл олдсонгүй: {env_path}")

for line in env_path.read_text(encoding="utf-8-sig").splitlines():
    line = line.strip()
    if line.startswith("GEMINI_API_KEY="):
        value = line.split("=", 1)[1].strip().strip('"').strip("'")
        if value:
            os.environ["GEMINI_API_KEY"] = value
        break

if not os.environ.get("GEMINI_API_KEY"):
    raise RuntimeError("GEMINI_API_KEY олдсонгүй.")

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def build_prompt(item):
    cue_lines = "\\n".join(
        f"Cue {cue_id}: {interval}"
        for cue_id, interval in item["cues"].items()
    )

    return f"""
Analyze the acoustic properties of the HUMAN VOICE in the attached audio.

The attached audio is a short clip extracted from a movie.
The clip begins at movie time {item["start"]}.

Analyze ONLY these exact intervals, measured from the beginning
of the attached clip:

{cue_lines}

IMPORTANT ANALYSIS RULES:

- Analyze only audible acoustic evidence from the human voice.
- Do NOT infer emotion or meaning from the words.
- Do NOT use subtitles, character identities, plot, or visual information.
- Do NOT compare this recording with another recording.
- Distinguish the human voice from background music and sound effects.
- Do NOT attribute musical intensity, instruments, impacts, or sound
  effects to the speaker's vocal characteristics.
- Evaluate perceived vocal loudness, not the overall loudness of the mix.
- Do NOT assume that loudness means anger or strong emotion.
- Do NOT assume that high pitch means excitement or fear.
- Do NOT invent vocal qualities that are not clearly audible.
- If music, sound effects, or audio processing make a feature uncertain,
  explicitly report that uncertainty.
- If a cue is too short to assess reliably, say so.
- Report "unclear" rather than guessing.
- Breathing analysis is NOT required. Do not identify or classify
  breathing sounds.

FOR EACH CUE, ANALYZE THESE AREAS:

1. emotion:
   Identify the emotion conveyed by the voice alone.
   Possible values: neutral, anger, sadness, fear, happiness,
   excitement, disgust, surprise, mixed, or unclear.
   Do not force an emotion when acoustic evidence is insufficient.

2. emotion_intensity:
   low, moderate, high, or unclear.
   This is the intensity of the emotional expression, NOT recording volume.

3. vocal_delivery:
   Describe how the speaker produces the voice.
   Examples: conversational, restrained, emphatic, forceful,
   shouting, whispering, crying, laughing, tense, or controlled.
   Use only qualities supported by audible evidence.

4. loudness_and_pitch:
   Describe perceived vocal loudness separately from pitch.
   Include pitch contour where audible: rising, falling, level,
   or variable. Do not invent precise numerical pitch measurements.

5. voice_quality_and_rhythm:
   Describe observable voice quality, such as clear, breathy, raspy,
   strained, pressed, dry, or muffled. Describe speaking rate and
   rhythm as slow, moderate, fast, measured, or uneven.

6. vocal_stability:
   stable, unstable, or unclear, with brief audible evidence.

IMPORTANT DISTINCTIONS:

- Forceful delivery does not automatically indicate anger.
- Strained or pressed phonation does not automatically indicate
  emotional distress.
- A loud voice can be emotionally neutral.
- A quiet voice can express strong emotion.
- Do not interpret background music as evidence of vocal emotion.
- Do not use silence or pauses as evidence of emotion by themselves.

Return valid JSON only.

Use this structure:
{{
  "Cue 66": {{
    "emotion": "",
    "emotion_intensity": "",
    "vocal_delivery": "",
    "loudness_and_pitch": "",
    "voice_quality_and_rhythm": "",
    "vocal_stability": "",
    "clearly_audible_features": [],
    "uncertain_features": []
  }},
  "comparison": {{
    "overall_voice_quality": "",
    "recording_limitations": [],
    "uncertain_observations": []
  }}
}}

Replace "Cue 66" with each actual cue label exactly as listed above.
Include every requested cue exactly once.
Do not add extra cue entries.
"""

def process_audio(item):
    audio_path = item["path"]
    output_path = OUTPUT_DIR / f"{item['name']}.json"

    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                print(f"\n[SKIP] Өмнө хадгалсан: {item['name']}")
                return True
        except Exception:
            print(f"\n[WARNING] Өмнөх JSON буруу байна. Дахин шинжилнэ.")

    if not audio_path.exists():
        print(f"\n[ERROR] Аудио олдсонгүй: {audio_path}")
        return False

    print(f"\n{'=' * 60}")
    print(f"Шинжилж байна: {item['name']}")
    print(f"Файл: {audio_path.name}")
    print(f"Хэмжээ: {audio_path.stat().st_size:,} bytes")
    print(f"Cue тоо: {len(item['cues'])}")
    print("Gemini хүсэлт илгээж байна...")

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=[
                build_prompt(item),
                types.Part.from_bytes(
                    data=audio_path.read_bytes(),
                    mime_type="audio/wav",
                ),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0,
            ),
        )

        result = json.loads(response.text)

        if not isinstance(result, dict):
            raise ValueError("Gemini JSON object буцаасангүй.")

        expected_keys = {
            f"Cue {cue_id}" for cue_id in item["cues"]
        }
        missing_keys = expected_keys - set(result.keys())

        if missing_keys:
            print(f"[WARNING] Дутуу cue: {sorted(missing_keys)}")

        temp_path = output_path.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(output_path)

        print("[SUCCESS] Шинжилгээ дууслаа.")
        print(f"Хадгалсан: {output_path}")

        return True

    except Exception as exc:
        print(f"[ERROR] {item['name']}: {type(exc).__name__}: {exc}")
        return False


def main():
    # First validate the revised prompt on the emotional A/B pair only.
    # The original six-result folder is preserved; new outputs go to v2.
    pilot_names = {"emotional_original", "emotional_speech"}
    pilot_files = [item for item in AUDIO_FILES if item["name"] in pilot_names]

    print("HUSBANDS — REVISED ACOUSTIC PROMPT PILOT")
    print("Model: gemini-3.5-flash-lite")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Pilot files: {len(pilot_files)} (emotional Original + Speech)")

    results = []

    for index, item in enumerate(pilot_files, start=1):
        print(f"\nФайл {index}/{len(AUDIO_FILES)}")

        success = process_audio(item)
        results.append((item["name"], success))

        if index < len(pilot_files):
            time.sleep(3)

    print("\n" + "=" * 60)
    print("ТУРШИЛТЫН НЭГТГЭЛ")

    for name, success in results:
        status = "SUCCESS" if success else "FAILED"
        print(f"{status}: {name}")

    successful = sum(1 for _, success in results if success)
    print(f"\nАмжилттай: {successful}/{len(results)}")
    print(f"Үр дүнгийн хавтас: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
