
import json
import os
import shutil
import time
from pathlib import Path

import srt
from dotenv import load_dotenv


# ==================================================
# 1. ТОХИРГОО
# ==================================================

SUBTITLE_FILE = None  # Set from user input in main()
CHARACTER_FILE = "Husbands_characters_reconciled.json"

PROMPT_DIRECTORY = Path("speaker_prompts")
RESULT_DIRECTORY = Path("speaker_results")

ANALYSIS_INPUT_FILE = Path("speaker_analysis_input.json")
ASSIGNMENTS_FILE = Path("speaker_assignments.json")
REVIEW_REPORT_FILE = Path("speaker_review_report.json")

BATCH_SIZE = 40
CONTEXT_SIZE = 5
RETRY_ONLY_BATCHES = []  # Бүх багцыг боловсруулна.

VALID_SPEAKER_TYPES = [
    "character",
    "inner_monologue",
    "narrator",
    "unknown"
]

VALID_CONFIDENCE_LEVELS = [
    "high",
    "medium",
    "low"
]
def create_ai_client(provider, api_key):

    if provider == "openai":

        from openai import OpenAI

        return OpenAI(api_key=api_key)

    elif provider == "gemini":

        from google import genai

        return genai.Client(api_key=api_key)

    else:

        raise ValueError(
            f"Дэмжигдээгүй provider: {provider}"
        )


def load_ai_config():

    load_dotenv()

    provider = os.getenv(
        "AI_PROVIDER",
        "gemini"
    ).strip().lower()

    if provider == "openai":

        api_key = os.getenv("OPENAI_API_KEY")

        model = os.getenv(
            "OPENAI_MODEL",
            "gpt-5.5"
        )

    elif provider == "gemini":

        api_key = os.getenv("GEMINI_API_KEY")

        model = os.getenv(
            "GEMINI_MODEL",
            "gemini-3.8-flash"
        )

    else:

        raise ValueError(
            f"Дэмжигдээгүй provider: {provider}"
        )

    if not api_key:

        raise ValueError(
            f"{provider.upper()} API key .env файлд олдсонгүй."
        )

    client = create_ai_client(
        provider,
        api_key
    )

    return provider, model, client
def call_ai(
    client,
    provider,
    model,
    user_prompt,
    system_prompt
):

    max_retries = 5

    for attempt in range(max_retries):

        try:

            if provider == "openai":

                response = client.responses.create(
                    model=model,
                    instructions=system_prompt,
                    input=user_prompt
                )

                return response.output_text

            elif provider == "gemini":

                from google.genai import types

                response = client.models.generate_content(
                    model=model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        response_mime_type="application/json",
                        temperature=0.2
                    )
                )

                return response.text

            raise ValueError(
                f"Буруу provider: {provider}"
            )

        except Exception as error:

            error_text = str(error)

            temporary_errors = [
                "503",
                "UNAVAILABLE",
                "429",
                "RESOURCE_EXHAUSTED",
                "500",
                "INTERNAL",
                "502",
                "504"
            ]

            is_temporary = any(
                code in error_text
                for code in temporary_errors
            )

            if not is_temporary:
                raise

            if attempt == max_retries - 1:
                raise

            wait_seconds = 2 ** (attempt + 1)

            print(
                f"\nТүр алдаа гарлаа: {error_text[:150]}"
            )

            print(
                f"{wait_seconds} секунд хүлээгээд "
                f"дахин оролдоно... "
                f"({attempt + 1}/{max_retries})"
            )

            time.sleep(wait_seconds)

def process_single_batch(
    batch,
    characters,
    client,
    provider,
    model
):

    batch_number = batch["batch_number"]

    result_file = (
        RESULT_DIRECTORY
        / f"batch_{batch_number:02d}.json"
    )

    prompt_file = (
        PROMPT_DIRECTORY
        / f"batch_{batch_number:02d}.txt"
    )

    if not prompt_file.exists():

        raise FileNotFoundError(
            f"Prompt файл олдсонгүй: {prompt_file}"
        )

    user_prompt = prompt_file.read_text(
        encoding="utf-8"
    )

    # Өмнөх үр дүн байгаа бол эхлээд шалгана.
    if result_file.exists():

        try:

            existing_result = json.loads(
                result_file.read_text(
                    encoding="utf-8"
                )
            )

            existing_errors = validate_speaker_results(
                existing_result,
                batch["subtitles"],
                characters
            )

            if not existing_errors:

                print(
                    f"\nБагц {batch_number}: "
                    "Өмнөх үр дүн хүчинтэй тул алгаслаа."
                )

                return False

            print(
                f"\nБагц {batch_number}: "
                "Өмнөх үр дүн шалгалтад тэнцсэнгүй. "
                "Дахин боловсруулна."
            )

        except (
            json.JSONDecodeError,
            OSError,
            TypeError
        ):

            print(
                f"\nБагц {batch_number}: "
                "Өмнөх JSON файл гэмтсэн байна. "
                "Дахин боловсруулна."
            )

    system_prompt = (
        "Чи киноны хадмалын яригчийг "
        "нарийн тодорхойлдог шинжээч. "
        "Зааврыг яг дагаж, зөвхөн шаардсан JSON "
        "бүтцийг буцаа."
    )

    print(
        f"\nБагц {batch_number}: "
        f"AI руу хүсэлт илгээж байна..."
    )

    ai_response = call_ai(
        client=client,
        provider=provider,
        model=model,
        user_prompt=user_prompt,
        system_prompt=system_prompt
    )

    try:

        result = json.loads(ai_response)

    except json.JSONDecodeError as error:

        raise ValueError(
            f"Багц {batch_number}: "
            f"AI JSON буцаасангүй. {error}"
        )

    batch_subtitles = batch["subtitles"]

    validation_errors = validate_speaker_results(
        result,
        batch_subtitles,
        characters
    )

    if validation_errors:

        raise ValueError(
            f"Багц {batch_number}: "
            "AI-ийн үр дүн шалгалтад тэнцсэнгүй:\n"
            + "\n".join(validation_errors)
        )

    save_json(
        result_file,
        result
    )

    print(
        f"Багц {batch_number}: "
        "Амжилттай хадгаллаа."
    )

    return True

# ==================================================
# 2. ФАЙЛ УНШИХ
# ==================================================

def load_subtitles(file_path):

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Файл олдсонгүй: {path}"
        )

    content = path.read_text(
        encoding="utf-8-sig"
    )

    subtitles = list(srt.parse(content))

    if not subtitles:
        raise ValueError(
            "SRT файл хоосон байна."
        )

    return subtitles


def load_characters(file_path):

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Файл олдсонгүй: {path}"
        )

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    characters = data.get("characters", [])

    if not characters:
        raise ValueError(
            "Дүрийн бүртгэл хоосон байна."
        )

    return characters


# ==================================================
# 3. ӨГӨГДӨЛ БЭЛТГЭХ
# ==================================================

def prepare_speaker_data(subtitles, characters):

    character_list = []

    for character in characters:

        character_list.append({
            "id": character["id"],
            "name": character["canonical_name"],
            "aliases": character.get("aliases", []),
            "gender": character.get("gender"),
            "relationships": character.get(
                "relationships", []
            ),
            "speech_style": character.get(
                "speech_style", []
            )
        })

    subtitle_list = []

    for sub in subtitles:

        subtitle_list.append({
            "subtitle_id": sub.index,
            "start": str(sub.start),
            "end": str(sub.end),
            "text": sub.content
        })

    return {
        "characters": character_list,
        "subtitles": subtitle_list
    }


# ==================================================
# 4. ХАДМАЛЫГ БАГЦЛАХ
# ==================================================

def create_subtitle_batches(
    subtitles,
    batch_size=BATCH_SIZE,
    context_size=CONTEXT_SIZE
):

    if batch_size < 1:
        raise ValueError(
            "batch_size 1-ээс бага байж болохгүй."
        )

    if context_size < 0:
        raise ValueError(
            "context_size сөрөг байж болохгүй."
        )

    batches = []

    for start in range(
        0,
        len(subtitles),
        batch_size
    ):

        end = min(
            start + batch_size,
            len(subtitles)
        )

        context_start = max(
            0,
            start - context_size
        )

        context_end = min(
            len(subtitles),
            end + context_size
        )

        batches.append({
            "batch_number": len(batches) + 1,

            "subtitles": subtitles[start:end],

            "context_before": subtitles[
                context_start:start
            ],

            "context_after": subtitles[
                end:context_end
            ]
        })

    return batches



# ==================================================
# 5. SPEAKER PROMPT ҮҮСГЭХ
# ==================================================

def build_speaker_prompt(batch, characters):

    target_subtitles = batch["subtitles"]

    output_template = {
        "results": [
            {
                "subtitle_id": item["subtitle_id"],
                "speaker_id": None,
                "speaker_name": None,
                "speaker_type": "unknown",
                "confidence": "low",
                "evidence": "Шийдвэрийн нотолгоо"
            }
            for item in target_subtitles
        ]
    }

    # Husbands-ийн тохиролцсон voice identity:
    # Narration болон Vivienne-ийн inner monologue хоёулаа Vivienne-ийн
    # нэг voice identity (C001)-д оноогдоно. Narrator тусдаа voice биш.
    vivienne = next(
        (
            character
            for character in characters
            if character.get("name") == "Vivienne"
        ),
        None
    )

    if not vivienne:
        raise ValueError(
            "Дүрийн бүртгэлээс Vivienne олдсонгүй. "
            "Narration болон inner_monologue-ийг C001-д оноох боломжгүй."
        )

    vivienne_id = vivienne["id"]

    # Narrator нэртэй хуучин бүртгэлийг AI-д дүр болгон харуулахгүй.
    # Энэ нь тусдаа Narrator voice үүсгэхээс сэргийлнэ.
    prompt_characters = [
        character
        for character in characters
        if character.get("name") != "Narrator"
    ]

    narrator_instruction = f"""
VOICE IDENTITY-ИЙН ЗААВАЛ МӨРДӨХ ДҮРЭМ:

- Киноны narration / хүүрнэлийг тусдаа Narrator дүрд оноож болохгүй.
- Narration-ийн speaker_id = "{vivienne_id}"
- Narration-ийн speaker_name = "Vivienne"
- Narration-ийн speaker_type = "narrator"

- Vivienne-ийн дотоод бодол, мэдрэмж, дурсамжийг:
  speaker_id = "{vivienne_id}"
  speaker_name = "Vivienne"
  speaker_type = "inner_monologue"

- Narration болон Vivienne-ийн inner_monologue нь dubbing хийхэд
  НЭГ Л VOICE IDENTITY буюу Vivienne байна.
- "Narrator" нэртэй тусдаа speaker_id-г хэзээ ч бүү ашигла.
- Narrator бүртгэл дүрийн жагсаалтад байсан ч түүнийг оноолтод бүү ашигла.
- narration ба inner_monologue-ийн ялгааг зөвхөн speaker_type-аар хадгал.
"""

    return f"""
Чи мэргэжлийн киноны диалог болон хүүрнэлийн шинжээч.

ҮҮРЭГ:
Шинжлэх хадмал бүрийн яригч болон өгүүлэмжийн төрлийг
контекстэд үндэслэн тодорхойл.

ДҮРҮҮД:
{json.dumps(
    prompt_characters,
    ensure_ascii=False,
    indent=2
)}

ӨМНӨХ КОНТЕКСТ:
{json.dumps(
    batch["context_before"],
    ensure_ascii=False,
    indent=2
)}

ШИНЖЛЭХ ХАДМАЛ:
{json.dumps(
    target_subtitles,
    ensure_ascii=False,
    indent=2
)}

ДАРААХ КОНТЕКСТ:
{json.dumps(
    batch["context_after"],
    ensure_ascii=False,
    indent=2
)}

==================================================
1. SPEAKER_TYPE АНГИЛАЛ
==================================================

Зөвхөн дараах гурван төрлийг ялгаж ойлго.

A. character

Дүрийн амнаас гарч буй шууд яриа.

Жишээ:
- You betrayed me.
- Where are you going?
- I never loved you.

Шууд яриа нь тухайн дүрийн ID, нэртэй байна.

B. inner_monologue

Дүрийн дотоод бодол, мэдрэмж, дурсамж,
өөртөө хэлж буй үг, субъектив дотоод өгүүлэмж.

Жишээ:
- I couldn't believe what I had heard.
- Why did he do this to me?
- I knew I had to leave.
- I didn't want to see him again.

АНХААР:
Эхний биеийн I, me, my, myself ашигласан нь
дангаараа inner_monologue гэж ангилах хангалттай нотолгоо биш.

Дотоод бодлын эзнийг өгүүлэмжийн контекст,
дүрийн POV болон өмнөх, дараах мөрүүдээр тогтоо.

C. narrator

Киноны үйл явдлыг гаднын өгүүлэгчийн байр сууринаас
тайлбарлаж буй хүүрнэл.

Жишээ:
- The sun was setting.
- Lucian walked toward the door.
- The room was completely silent.
- She looked at him without saying a word.

Narrator нь дүрийн дотоод бодол биш,
үйл явдлыг хүүрнэж буй өгүүлэгч юм.

==================================================
2. NARRATOR БА INNER_MONOLOGUE-ИЙН ЯЛГАА
==================================================

Энэ бол хамгийн чухал дүрэм.

1. Эхний биеэр өгүүлсэн бүх мөрийг narrator гэж үзэж болохгүй.

2. Эхний биеэр өгүүлсэн бүх мөрийг
   inner_monologue гэж үзэж болохгүй.

3. Хэрэв өгүүлэмжид дүрийн хувийн бодол,
   мэдрэмж, дурсамж, айдас, хүсэл,
   шийдвэр эсвэл субъектив үнэлгээ илэрч байвал
   inner_monologue гэж ангил.

4. Хэрэв өгүүлэмж зөвхөн үйл явдал,
   орчин, бусад дүрийн үйлдлийг
   гаднын байр сууринаас тайлбарлаж байвал
   narrator гэж ангил.

5. Lucian, Mother, Aria зэрэг дүрийн үйлдлийг
   тайлбарласан нь автоматаар narrator болно.
   Гэхдээ тухайн өгүүлбэр дотоод бодол байвал
   POV дүрийн inner_monologue байна.

6. Ярианы агуулга, POV болон өгүүлэмжийн
   үргэлжлэлийг хамтад нь шинжил.

==================================================
3. INNER_MONOLOGUE-ИЙН ЭЗНИЙГ ТОГТООХ
==================================================

1. Дотоод бодлын эзнийг characters жагсаалтаас ол.

2. Дотоод бодлын эзнийг тогтоохдоо:
   - Өмнөх болон дараах хадмал
   - POV-ийн үргэлжлэл
   - Дүрүүдийн харилцаа
   - Өгүүлэмжийн нөхцөл
   - Дүрийн тухайн үеийн байр суурь
   зэргийг ашигла.

3. Vivienne өөрийн бодол, мэдрэмжийг
   хүүрнэж байгаа нь контекстоор батлагдсан бол:

   speaker_id = "C001"
   speaker_name = "Vivienne"
   speaker_type = "inner_monologue"

4. Зөвхөн I, me, my гэсэн үгсээр
   Vivienne гэж таамаглаж болохгүй.

5. Narrator бүртгэл байгаа гэдгээр
   inner_monologue-ийг Narrator-д оноож болохгүй.

6. Дотоод бодлын эзнийг тогтоох боломжгүй бол:

   speaker_id = null
   speaker_name = null
   speaker_type = "inner_monologue"

==================================================
4. NARRATION БА VOICE IDENTITY
==================================================

{narrator_instruction}

Энэ киноны баталгаажсан dubbing шийдвэрийг мөрд:
Narration нь тусдаа Narrator voice биш, Vivienne-ийн voice identity.
Гаднын хүүрнэлийн төрлийг speaker_type = "narrator" гэж хадгалж болно,
гэхдээ speaker_id болон speaker_name нь Vivienne байна.

Narration ба inner_monologue-ийн speaker identity-г хэзээ ч салгаж болохгүй.

==================================================
5. UNKNOWN АНГИЛАЛ
==================================================

Яригчийг найдвартай тогтоох боломжгүй бол
speaker_type = "unknown" ашигла.

Энэ үед:

speaker_id = null
speaker_name = null
speaker_type = "unknown"

Unknown болон inner_monologue-ийг ялга:

- Дотоод бодол гэдэг нь тодорхой боловч эзэн
  тодорхойгүй бол inner_monologue.
- Ямар төрлийн өгүүлэмж болох нь ч тодорхойгүй
  бол unknown.

==================================================
6. НЭМЭЛТ ШИНЖИЛГЭЭНИЙ ДҮРЭМ
==================================================

1. Нэр дурдагдсан дүрийг шууд яригч гэж бүү үз.
2. Ярианы ээлжийг харгалз.
3. Өмнөх болон дараах контекстыг ашигла.
4. Дүрүүдийн харилцааг харгалз.
5. Дүрүүдийн ярианы хэв маягийг харгалз.
6. Нэр болон ID хоорондоо яг таарч байх ёстой.
7. Таамгаар дүрийн ID бүү оноо.
8. Бага confidence шаардлагатай үед ашигла.
9. Өгүүлбэрийн утгыг бус, бодит speaker-ийг тогтоо.
10. Нэг мөрийн шийдвэрийг бусад мөрийн шийдвэртэй
    зөрчилдөхгүй байлга.

==================================================
7. ГАРАЛТЫН ШААРДЛАГА
==================================================

1. Бүх хадмалын subtitle_id-г хэвээр хадгал.
2. Нэг ч хадмалыг алгасаж болохгүй.
3. Зөвхөн ШИНЖЛЭХ ХАДМАЛ хэсгийн мөрүүдийг буцаа.
4. Контекстийн мөрүүдийг буцааж болохгүй.
5. subtitle_id бүр яг нэг удаа байх ёстой.
6. Бүх үр дүнг JSON object хэлбэрээр буцаа.
7. Нэмэлт тайлбар, Markdown бүү оруул.
8. evidence нь тухайн шийдвэрийн товч,
   бодит нотолгоо байна.
9. Бүх мөрийн speaker_type зөвшөөрөгдсөн
   утгын нэг байх ёстой.

ЗӨВШӨӨРӨГДСӨН speaker_type:
{VALID_SPEAKER_TYPES}

ЗӨВШӨӨРӨГДСӨН confidence:
{VALID_CONFIDENCE_LEVELS}

ГАРАЛТЫН БҮТЭЦ:
{json.dumps(
    output_template,
    ensure_ascii=False,
    indent=2
)}

JSON-оос өөр зүйл буцааж болохгүй.
"""



# ==================================================
# 6. SPEAKER CONFIDENCE АНГИЛАХ
# ==================================================

def classify_speaker_confidence(results):

    review = {
        "high": [],
        "medium": [],
        "low": []
    }

    for item in results:

        confidence = item.get("confidence")

        if confidence in review:

            review[confidence].append(item)

    return review


# ==================================================
# 7. SPEAKER ҮР ДҮНГ ШАЛГАХ
# ==================================================

def validate_speaker_results(
    ai_response,
    batch_subtitles,
    characters
):

    errors = []

    if not isinstance(ai_response, dict):

        return [
            "AI-ийн хариу JSON object биш байна."
        ]

    results = ai_response.get("results")

    if not isinstance(results, list):

        return [
            "AI-ийн хариунд results жагсаалт алга."
        ]

    valid_speaker_ids = {
        character["id"]
        for character in characters
    }

    speaker_name_by_id = {
        character["id"]: character["canonical_name"]
        for character in characters
    }

    expected_ids = {
       (
            subtitle["subtitle_id"]
            if isinstance(subtitle, dict)
            else subtitle.index
        )
        for subtitle in batch_subtitles
    }

    received_ids = []

    for item in results:

        if not isinstance(item, dict):

            errors.append(
                "results дотор буруу бүтэцтэй элемент байна."
            )

            continue

        subtitle_id = item.get("subtitle_id")

        if type(subtitle_id) is not int:

            errors.append(
                f"Буруу subtitle_id: {subtitle_id}"
            )

            continue

        received_ids.append(subtitle_id)

        speaker_id = item.get("speaker_id")
        speaker_name = item.get("speaker_name")
        speaker_type = item.get("speaker_type")
        confidence = item.get("confidence")

        if speaker_id is not None:

            if not isinstance(speaker_id, str):

                errors.append(
                    f"[{subtitle_id}] speaker_id текст биш байна."
                )

            elif speaker_id not in valid_speaker_ids:

                errors.append(
                    f"[{subtitle_id}] Байхгүй дүрийн ID: "
                    f"{speaker_id}"
                )

            elif speaker_name != speaker_name_by_id[speaker_id]:

                errors.append(
                    f"[{subtitle_id}] Speaker нэр таарахгүй: "
                    f"{speaker_id} нь "
                    f"{speaker_name_by_id[speaker_id]} байх ёстой."
                )

        elif speaker_name is not None:

            errors.append(
                f"[{subtitle_id}] speaker_id null боловч "
                "speaker_name null биш байна."
            )

        if speaker_type not in VALID_SPEAKER_TYPES:

            errors.append(
                f"[{subtitle_id}] Буруу speaker_type: "
                f"{speaker_type}"
            )

        if confidence not in VALID_CONFIDENCE_LEVELS:

            errors.append(
                f"[{subtitle_id}] Буруу confidence: "
                f"{confidence}"
            )

        if not isinstance(item.get("evidence"), str) or not item.get("evidence").strip():

            errors.append(
                f"[{subtitle_id}] evidence хоосон байна."
            )

        if speaker_type == "unknown":

            if speaker_id is not None or speaker_name is not None:

                errors.append(
                    f"[{subtitle_id}] unknown үед "
                    "speaker_id болон speaker_name null байх ёстой."
                )

        if speaker_type == "character" and speaker_id is None:

            errors.append(
                f"[{subtitle_id}] character төрөлтэй боловч "
                "speaker_id алга."
            )

        if speaker_type == "narrator" and speaker_id is None:

            errors.append(
                f"[{subtitle_id}] narrator төрөлтэй боловч "
                "speaker_id алга."
            )

        if (
            speaker_type == "inner_monologue"
            and speaker_id is None
            and speaker_name is not None
        ):

            errors.append(
                f"[{subtitle_id}] inner_monologue эзэнгүй боловч "
                "speaker_name null биш байна."
            )

    if len(received_ids) != len(set(received_ids)):

        errors.append(
            "Давхардсан subtitle_id байна."
        )

    received_set = set(received_ids)

    missing_ids = expected_ids - received_set
    extra_ids = received_set - expected_ids

    if missing_ids:

        errors.append(
            f"Орхигдсон хадмалын дугаар: "
            f"{sorted(missing_ids)}"
        )

    if extra_ids:

        errors.append(
            f"Багцад байхгүй хадмалын дугаар: "
            f"{sorted(extra_ids)}"
        )

    return errors


# ==================================================
# 8. НЭГДСЭН ҮР ДҮН ҮҮСГЭХ
# ==================================================

def build_speaker_output(subtitles, results):

    return {
        "movie": {
            "subtitle_file": SUBTITLE_FILE,
            "total_subtitles": len(subtitles)
        },

        "assignments": results,

        "review": classify_speaker_confidence(
            results
        )
    }


# ==================================================
# 9. JSON ФАЙЛ ХАДГАЛАХ
# ==================================================

def save_json(file_path, data):

    path = Path(file_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=4
        ),
        encoding="utf-8"
    )

    temporary_path.replace(path)


# ==================================================
# 10. БАГЦЫН ҮР ДҮН УНШИХ
# ==================================================

def load_batch_result(batch_number):

    result_file = (
        RESULT_DIRECTORY
        / f"batch_{batch_number:02d}.json"
    )

    if not result_file.exists():

        return None, [
            f"Үр дүнгийн файл олдсонгүй: {result_file}"
        ]

    try:

        response = json.loads(
            result_file.read_text(
                encoding="utf-8-sig"
            )
        )

    except json.JSONDecodeError as error:

        return None, [
            f"{result_file} JSON алдаатай: {error}"
        ]

    return response, []


# ==================================================
# 11. БҮХ БАГЦЫН ҮР ДҮНГ НЭГТГЭХ
# ==================================================

def collect_batch_results(batches, characters):

    all_results = []
    errors = []

    for batch in batches:

        batch_number = batch["batch_number"]

        response, file_errors = load_batch_result(
            batch_number
        )

        if file_errors:

            errors.extend(file_errors)
            continue

        batch_errors = validate_speaker_results(
            response,
            batch["subtitles"],
            characters
        )

        if batch_errors:

            for error in batch_errors:

                errors.append(
                    f"[Багц {batch_number}] {error}"
                )

            continue

        all_results.extend(
            response["results"]
        )

    return all_results, errors


# ==================================================
# 12. ЭЦСИЙН БҮТЭН БАЙДАЛ ШАЛГАХ
# ==================================================

def validate_complete_assignments(
    results,
    subtitles
):

    errors = []

    expected_ids = {
        (
        str(subtitle["subtitle_id"])
        if isinstance(subtitle, dict)
        else str(subtitle.index)
    )
    for subtitle in subtitles
    }

    received_ids = [
         str(item["subtitle_id"])
    for item in results
    ]

    if len(received_ids) != len(set(received_ids)):

        errors.append(
            "Нэгдсэн үр дүнд давхардсан subtitle_id байна."
        )

    received_set = set(received_ids)

    missing_ids = expected_ids - received_set
    extra_ids = received_set - expected_ids

    if missing_ids:

        errors.append(
            f"Нэгдсэн үр дүнд дутуу хадмал: "
            f"{sorted(missing_ids)}"
        )

    if extra_ids:

        errors.append(
            f"Нэгдсэн үр дүнд илүүдэл хадмал: "
            f"{sorted(extra_ids)}"
        )

    return errors


# ==================================================
# 13. ҮР ДҮН БОЛОВСРУУЛАХ
# ==================================================

def process_results(
    subtitles,
    batches,
    characters
):

    print()
    print("Speaker үр дүн шалгаж эхэллээ.")

    all_results, errors = collect_batch_results(
        batches,
        characters
    )

    if errors:

        save_json(
            REVIEW_REPORT_FILE,
            {
                "status": "needs_review",
                "errors": errors
            }
        )

        print("Үр дүнг бүрэн баталгаажуулж чадсангүй.")
        print("Алдааны тоо:", len(errors))
        print("Тайлан:", REVIEW_REPORT_FILE)

        return False

    complete_errors = validate_complete_assignments(
        all_results,
        subtitles
    )

    if complete_errors:

        save_json(
            REVIEW_REPORT_FILE,
            {
                "status": "needs_review",
                "errors": complete_errors
            }
        )

        print("Нэгдсэн үр дүн бүрэн биш байна.")

        for error in complete_errors:
            print(error)

        return False

    all_results.sort(
        key=lambda item: item["subtitle_id"]
    )

    output = build_speaker_output(
        subtitles,
        all_results
    )

    # Шинэ бүрэн үр дүнгээр солихоос өмнө хуучин assignment-ийг хадгална.
    if ASSIGNMENTS_FILE.exists():
        backup_file = ASSIGNMENTS_FILE.with_name(
            f"{ASSIGNMENTS_FILE.stem}_backup_{time.strftime('%Y%m%d_%H%M%S')}{ASSIGNMENTS_FILE.suffix}"
        )

        suffix = 1
        while backup_file.exists():
            backup_file = ASSIGNMENTS_FILE.with_name(
                f"{ASSIGNMENTS_FILE.stem}_backup_{time.strftime('%Y%m%d_%H%M%S')}_{suffix}{ASSIGNMENTS_FILE.suffix}"
            )
            suffix += 1

        shutil.copy2(
            ASSIGNMENTS_FILE,
            backup_file
        )

        print(f"Өмнөх assignment backup: {backup_file}")

    save_json(
        ASSIGNMENTS_FILE,
        output
    )

    if REVIEW_REPORT_FILE.exists():
        REVIEW_REPORT_FILE.unlink()

    print("Бүх багц амжилттай баталгаажлаа.")
    print("Нийт оноолт:", len(all_results))
    print("High:", len(output["review"]["high"]))
    print("Medium:", len(output["review"]["medium"]))
    print("Low:", len(output["review"]["low"]))
    print("Эцсийн файл:", ASSIGNMENTS_FILE)

    return True


# ==================================================
# 14. PROMPT БОЛОН INPUT ФАЙЛ ҮҮСГЭХ
# ==================================================

def prepare_analysis_files(
    speaker_data,
    batches
):

    PROMPT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    for batch in batches:

        prompt = build_speaker_prompt(
            batch,
            speaker_data["characters"]
        )

        prompt_file = (
            PROMPT_DIRECTORY
            / f"batch_{batch['batch_number']:02d}.txt"
        )

        prompt_file.write_text(
            prompt,
            encoding="utf-8"
        )

    result = {
        "characters": speaker_data["characters"],
        "batches": batches
    }

    save_json(
        ANALYSIS_INPUT_FILE,
        result
    )

    print("Бүх багцын Prompt бэлэн боллоо.")
    print("Prompt хавтас:", PROMPT_DIRECTORY)
    print("Үүсгэсэн Prompt:", len(batches))
    print("Үр дүнгийн хавтас:", RESULT_DIRECTORY)


# ==================================================
# 15. MAIN
# ==================================================


# ==================================================
# 15. MAIN
# ==================================================

def get_subtitle_file_path():

    while True:

        raw_path = input(
            "\nEnter the SRT file path to process: "
        ).strip().strip('"').strip("'")

        if not raw_path:
            print("The file path is empty. Please try again.")
            continue

        subtitle_path = Path(raw_path).expanduser()

        if not subtitle_path.is_file():
            print(f"File not found: {subtitle_path}")
            print("Please enter a valid file path.")
            continue

        return subtitle_path.resolve()


def main():

    global SUBTITLE_FILE, RESULT_DIRECTORY

    import time

    subtitle_path = get_subtitle_file_path()
    SUBTITLE_FILE = str(subtitle_path)

    subtitles = load_subtitles(
        subtitle_path
    )

    characters = load_characters(
        CHARACTER_FILE
    )

    speaker_data = prepare_speaker_data(
        subtitles,
        characters
    )

    batches = create_subtitle_batches(
        speaker_data["subtitles"],
        batch_size=BATCH_SIZE,
        context_size=CONTEXT_SIZE
    )

    prepare_analysis_files(
        speaker_data,
        batches
    )

    provider, model, client = load_ai_config()

    # Өмнөх багцын үр дүнг backup хийнэ. Ингэснээр өөр SRT-ийн
    # хуучин үр дүн шинэ боловсруулалттай холилдохгүй.
    if RESULT_DIRECTORY.exists():
        backup_directory = RESULT_DIRECTORY.with_name(
            f"{RESULT_DIRECTORY.name}_backup_{time.strftime('%Y%m%d_%H%M%S')}"
        )

        suffix = 1
        while backup_directory.exists():
            backup_directory = RESULT_DIRECTORY.with_name(
                f"{RESULT_DIRECTORY.name}_backup_{time.strftime('%Y%m%d_%H%M%S')}_{suffix}"
            )
            suffix += 1

        shutil.move(
            str(RESULT_DIRECTORY),
            str(backup_directory)
        )

        print(f"Өмнөх үр дүнгийн backup: {backup_directory}")

    RESULT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n========== SPEAKER ENGINE ==========")

    print(f"Provider: {provider.upper()}")
    print(f"Model: {model}")

    print(f"Нийт хадмал: {len(subtitles)}")
    print(f"Нийт дүр: {len(characters)}")
    print(f"Нийт багц: {len(batches)}")

    print("\nAI боловсруулалт эхэллээ.")

    successful = 0
    failed = 0
    skipped = 0

    for index, batch in enumerate(batches):

        batch_number = batch["batch_number"]

        print(
            f"\n---------- "
            f"БАГЦ {batch_number}/{len(batches)} "
            f"----------"
        )

        try:

            result = process_single_batch(
                batch=batch,
                characters=characters,
                client=client,
                provider=provider,
                model=model
            )

            if result is False:
                skipped += 1
            else:
                successful += 1

        except Exception as error:

            failed += 1

            print(
                f"\nБагц {batch_number} алдаатай: {error}"
            )

        if index < len(batches) - 1:

            print(
                "\nДараагийн багцын өмнө "
                "5 секунд хүлээнэ."
            )

            time.sleep(5)

    print("\n========== AI БОЛОВСРУУЛАЛТЫН ТАЙЛАН ==========")

    print(f"Амжилттай: {successful}")
    print(f"Алгассан: {skipped}")
    print(f"Алдаатай: {failed}")

    print("\nSpeaker үр дүн шалгаж эхэллээ.")

    process_results(
        subtitles,
        batches,
        characters
    )



# ==================================================
# 16. OFFLINE ТЕСТҮҮД
# ==================================================

def test_speaker_confidence_classification():

    results = [
        {"subtitle_id": 1, "confidence": "high"},
        {"subtitle_id": 2, "confidence": "medium"},
        {"subtitle_id": 3, "confidence": "low"},
        {"subtitle_id": 4, "confidence": "high"}
    ]

    classified = classify_speaker_confidence(
        results
    )

    assert len(classified["high"]) == 2
    assert len(classified["medium"]) == 1
    assert len(classified["low"]) == 1

    print("Speaker confidence тест: PASS")


def test_subtitle_batch_context():

    subtitles = [
        {"subtitle_id": i}
        for i in range(1, 101)
    ]

    batches = create_subtitle_batches(
        subtitles,
        batch_size=40,
        context_size=5
    )

    assert len(batches) == 3

    assert len(batches[0]["subtitles"]) == 40
    assert len(batches[0]["context_before"]) == 0
    assert len(batches[0]["context_after"]) == 5

    assert len(batches[1]["subtitles"]) == 40
    assert len(batches[1]["context_before"]) == 5
    assert len(batches[1]["context_after"]) == 5

    assert len(batches[2]["subtitles"]) == 20
    assert len(batches[2]["context_before"]) == 5
    assert len(batches[2]["context_after"]) == 0

    print("Багцын контекст тест: PASS")


def test_speaker_validation():

    characters = [
        {
            "id": "C001",
            "canonical_name": "Vivienne"
        }
    ]

    subtitles = [
        {"subtitle_id": 1}
    ]

    correct_response = {
        "results": [
            {
                "subtitle_id": 1,
                "speaker_id": "C001",
                "speaker_name": "Vivienne",
                "speaker_type": "character",
                "confidence": "high",
                "evidence": "Тест"
            }
        ]
    }

    errors = validate_speaker_results(
        correct_response,
        subtitles,
        characters
    )

    assert not errors, f"Complete assignment errors: {errors}"

    print("Зөв Speaker тест: PASS")

    unknown_response = {
        "results": [
            {
                "subtitle_id": 1,
                "speaker_id": None,
                "speaker_name": None,
                "speaker_type": "unknown",
                "confidence": "low",
                "evidence": "Тогтоох боломжгүй"
            }
        ]
    }

    errors = validate_speaker_results(
        unknown_response,
        subtitles,
        characters
    )

    assert not errors

    print("Unknown Speaker тест: PASS")

    inner_monologue_response = {
        "results": [
            {
                "subtitle_id": 1,
                "speaker_id": None,
                "speaker_name": None,
                "speaker_type": "inner_monologue",
                "confidence": "low",
                "evidence": "Бодлын эзнийг тогтоох боломжгүй"
            }
        ]
    }

    errors = validate_speaker_results(
        inner_monologue_response,
        subtitles,
        characters
    )

    assert not errors

    print("Unknown inner_monologue тест: PASS")

    wrong_name_response = {
        "results": [
            {
                "subtitle_id": 1,
                "speaker_id": "C001",
                "speaker_name": "Lucian",
                "speaker_type": "character",
                "confidence": "high",
                "evidence": "Тест"
            }
        ]
    }

    errors = validate_speaker_results(
        wrong_name_response,
        subtitles,
        characters
    )

    assert any(
        "Speaker нэр таарахгүй" in error
        for error in errors
    )

    print("Буруу нэрийн тест: PASS")


def test_complete_assignments():

    subtitles = [
        {"subtitle_id": 1},
        {"subtitle_id": 2},
        {"subtitle_id": 3}
    ]

    results = [
        {"subtitle_id": 1},
        {"subtitle_id": 2},
        {"subtitle_id": 3}
    ]

    errors = validate_complete_assignments(
        results,
        subtitles
    )

    assert not errors

    print("Бүрэн хадмалын тест: PASS")

    incomplete_results = [
        {"subtitle_id": 1},
        {"subtitle_id": 3}
    ]

    errors = validate_complete_assignments(
        incomplete_results,
        subtitles
    )

    assert any(
        "дутуу хадмал" in error
        for error in errors
    )

    print("Дутуу хадмалын тест: PASS")


def run_all_tests():

    print()
    print("========== OFFLINE TEST ==========")

    test_speaker_confidence_classification()
    test_subtitle_batch_context()
    test_speaker_validation()
    test_complete_assignments()

    print("==================================")
    print("Бүх offline тест амжилттай дууслаа.")


# ==================================================
# 17. АЖИЛЛУУЛАХ
# ==================================================

if __name__ == "__main__":

    run_all_tests()

    print()
    print("========== SPEAKER ENGINE ==========")

    main()
