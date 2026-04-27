from dotenv import load_dotenv
import os, re, json, hashlib, logging
from collections import OrderedDict
from typing import Optional, List, Dict, Any

import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing. Check your .env file")

OPENAI_HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json",
}
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
MODEL      = "gpt-4o-mini"

# ── OpenAI gpt-4o-transcribe via REST API ────────────────────────
import asyncio

OPENAI_TRANSCRIBE_URL = "https://api.openai.com/v1/audio/transcriptions"

TRANSCRIPTION_PROMPTS = {
    "ar": (
        "أعاني من أعراض مرضية. "
        "لدي حمى وزكام وسعال وألم في الجسم وصداع وألم في الصدر وألم في البطن وألم في الساق. "
        "أعاني من إسهال وغثيان وتقيؤ وإمساك وحرقة معدة. "
        "منذ يوم يومين ثلاثة أيام أسبوع. "
        "ألم شديد متوسط خفيف. أحتاج إلى دواء."
    ),
    "hi": (
        "मुझे बुखार, खांसी, जुकाम, सिरदर्द, पेट दर्द, सीने में दर्द, पैर दर्द, "
        "उल्टी, दस्त, थकान, शरीर दर्द है। "
        "एक दिन, दो दिन, तीन दिन से। तेज बुखार, हल्का दर्द।"
    ),
    "en": (
        "I have fever, cold, cough, headache, chest pain, stomach pain, leg pain, "
        "body ache, diarrhea, nausea, vomiting, fatigue, sore throat, runny nose. "
        "Since one day, two days, three days, one week. Mild, moderate, severe."
    ),
    "ta": (
        "எனக்கு காய்ச்சல், இருமல், சளி, தலைவலி, வயிற்று வலி, "
        "மார்பு வலி, கால் வலி, வாந்தி, வயிற்றுப்போக்கு இருக்கிறது. "
        "ஒரு நாள், இரண்டு நாட்கள், மூன்று நாட்களாக."
    ),
    "te": (
        "నాకు జ్వరం, దగ్గు, జలుబు, తలనొప్పి, కడుపు నొప్పి, "
        "ఛాతీ నొప్పి, కాలు నొప్పి, వాంతి, విరేచనాలు ఉన్నాయి. "
        "ఒక రోజు, రెండు రోజులు, మూడు రోజులుగా."
    ),
}

async def transcribe_audio(audio_bytes: bytes, lang: str = "auto") -> tuple[str, str]:
    """Transcribe audio using OpenAI gpt-4o-transcribe.
    Returns (transcript, detected_language).
    gpt-4o-transcribe auto-detects the language when none is specified.
    A language-specific medical prompt is always included to prevent hallucination.
    """
    # Build multipart form data
    files = {
        "file": ("audio.webm", audio_bytes, "audio/webm"),
        "model": (None, "gpt-4o-transcribe"),
    }
    # Pass explicit language hint only when the caller knows the language
    if lang and lang != "auto":
        # OpenAI uses ISO-639-1 codes (en, hi, ar, ta, te)
        files["language"] = (None, lang)

    # Add a medical vocabulary prompt to prevent hallucination.
    # For "auto", use English prompt as fallback — Whisper still benefits from it.
    prompt_lang = lang if lang in TRANSCRIPTION_PROMPTS else "en"
    files["prompt"] = (None, TRANSCRIPTION_PROMPTS[prompt_lang])
    logger.info(f"Using transcription prompt for lang={prompt_lang!r}")

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(OPENAI_TRANSCRIBE_URL, headers=headers, files=files)
            r.raise_for_status()
            data = r.json()

        transcript = data.get("text", "").strip()
        logger.info(f"gpt-4o-transcribe result: {transcript!r}")

        detected_lang = detect_language(transcript) if transcript else "en"
        return transcript, detected_lang

    except httpx.HTTPStatusError as e:
        logger.error(f"OpenAI transcription error: {e.response.status_code} — {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"transcribe_audio failed: {e}")
        raise

async def normalize_to_english(text: str) -> str:
    """Converts any language/mixed speech (Hindi, Arabic, Hinglish, Arabizi)
    into a literal English translation. Returns original text on failure.
    NOTE: Never skip romanized text (Hinglish/Arabizi) — it still needs translation."""
    if not text or not text.strip():
        return text
    # Do NOT skip based on ASCII/script detection.
    # Romanized Hindi ("mujhe pair mein dard hai") has no non-ASCII chars but
    # is NOT English — it must still be translated.
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Translate the text to English. "
                    "Be strictly literal — do NOT rephrase, expand, restructure, or add any words. "
                    "Translate only what is said, exactly as said. "
                    "Handle Hindi, Arabic, Hinglish, Arabizi, Tamil, Telugu, and romanized scripts. "
                    "Output only the translated text — no explanations, no preamble."
                ),
            },
            {"role": "user", "content": text},
        ],
        "temperature": 0,
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(OPENAI_URL, headers=OPENAI_HEADERS, json=payload)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning(f"normalize_to_english failed: {e} — using raw transcript")
        return text




# Load Cipla dataset
with open("cipla_data.json") as f:
    CIPLA_DATA = json.load(f)


async def get_medicine_type(user_input: str) -> str:
    """Use GPT to classify the symptom into a medicine type keyword."""
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a medical assistant. "
                    "Given a symptom, return ONLY the type of medicine needed. "
                    "Examples:\n"
                    "fever → paracetamol\n"
                    "cold → antihistamine\n"
                    "stomach pain → antacid\n"
                    "diarrhea → ors\n"
                    "vomiting → domperidone\n"
                    "cough → bromhexine\n"
                    "body pain → ibuprofen\n"
                    "allergy → cetirizine\n"
                    "headache → paracetamol\n"
                    "Return only ONE word, lowercase."
                ),
            },
            {"role": "user", "content": user_input},
        ],
        "temperature": 0,
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(OPENAI_URL, headers=OPENAI_HEADERS, json=payload)
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"].strip().lower()
            # Clean: take only the first word to handle unexpected multi-word output
            return raw.split()[0]
    except Exception as e:
        logger.warning(f"get_medicine_type failed: {e} — returning empty string")
        return ""


async def smart_search_cipla(user_input: str) -> list:
    """Classify the symptom with GPT, then search CIPLA_DATA by generic name."""
    medicine_type = await get_medicine_type(user_input)
    if not medicine_type:
        return []

    results = []
    for item in CIPLA_DATA:
        generic = item.get("generic", "").lower()
        if medicine_type in generic:
            results.append(item)

    return results[:3]


async def get_gpt_medicines(user_input: str) -> list:
    """Ask GPT for up to 3 OTC medicines when Cipla dataset comes up short."""
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Suggest exactly 3 OTC medicines available in India for the given symptom. "
                    "Return ONLY a JSON array — no markdown, no preamble. "
                    "Format: [{\"name\": \"...\", \"generic\": \"...\", \"note\": \"...\"}]"
                ),
            },
            {"role": "user", "content": user_input},
        ],
        "temperature": 0,
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(OPENAI_URL, headers=OPENAI_HEADERS, json=payload)
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"]
            # Strip accidental markdown fences before parsing
            raw = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
    except Exception as e:
        logger.warning(f"get_gpt_medicines failed: {e} — returning empty list")
        return []


async def get_final_medicines(user_input: str) -> list:
    """
    Priority logic:
      1. Try Cipla dataset first (GPT-classified search).
      2. Fill remaining slots with GPT suggestions.
      3. Total = exactly 3 medicines (or fewer if nothing found).
    """
    cipla_results = await smart_search_cipla(user_input)

    final_results = [
        {
            "name": item["brand"],
            "generic": item["generic"],
            "note": "From Cipla dataset",
        }
        for item in cipla_results
    ]

    # Fill up to 3 with GPT if Cipla didn't return enough
    if len(final_results) < 3:
        gpt_results = await get_gpt_medicines(user_input)
        existing_names = {m["name"] for m in final_results}
        for med in gpt_results:
            if len(final_results) >= 3:
                break
            if med.get("name") not in existing_names:
                final_results.append(med)

    return final_results[:3]


def format_cipla_response(cipla_results, message, lang, days):
    return {
        "symptom_detected": message,
        "severity": "mild",
        "duration_note": duration_note(days, lang),
        "home_remedies": ["Rest and stay hydrated"],
        "medications": [
            {
                "name": item["brand"],
                "generic": item["generic"],
                "note": "From Cipla dataset"
            }
            for item in cipla_results
        ],
        "diet_tips": ["Drink plenty of fluids"],
        "warning": "Consult doctor if symptoms persist",
        "red_flag": False,
        "_source": "cipla"
    }


# ────────────────────────────────────────────────────────────────
#  SYSTEM PROMPT
# ────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a helpful India-based healthcare assistant.

Rules:
- Reply ONLY with valid JSON, no markdown, no preamble.
- ALWAYS reply in English
- Only suggest OTC medicines available in India.
- Include India-specific brand names (e.g., Dolo 650, Crocin, D-Cold Total).
- Always include at least 1 medicine for non-red-flag cases.
- Keep advice general and safe.
- If symptom duration > 3 days or red flags present, increase severity.

Return EXACTLY this JSON shape:
{
  "symptom_detected": "",
  "severity": "mild|moderate|see_doctor_now",
  "duration_note": "",
  "home_remedies": [],
  "medications": [
    {"name": "", "generic": "", "note": ""}
  ],
  "diet_tips": [],
  "warning": "",
  "red_flag": false
}
"""

# ────────────────────────────────────────────────────────────────
#  LANGUAGE DETECTION  (auto — no manual selection needed)
# ────────────────────────────────────────────────────────────────
def detect_language(text: str) -> str:
    """Detect language from Unicode script ranges."""
    if re.search(r"[\u0600-\u06FF]", text or ""):
        return "ar"   # Arabic
    if re.search(r"[\u0900-\u097F]", text or ""):
        return "hi"   # Hindi (Devanagari)
    if re.search(r"[\u0B80-\u0BFF]", text or ""):
        return "ta"   # Tamil
    if re.search(r"[\u0C00-\u0C7F]", text or ""):
        return "te"   # Telugu
    return "en"       # Default: English

def normalize_arabic(text: str) -> str:
    """Normalize Arabic hamza/alef variants so STT output matches rule keywords.
    Speech-to-text produces plain alef (ا 0x627) but rules use hamza forms
    like أ (0x623) and إ (0x625) — this makes both match the same way.
    """
    # Alef variants → plain alef
    text = re.sub(r"[\u0623\u0625\u0622\u0671]", "\u0627", text)
    # Remove Arabic diacritics (tashkeel / harakat)
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    # Teh marbuta → heh (optional, helps some words)
    text = text.replace("\u0629", "\u0647")
    return text

# ────────────────────────────────────────────────────────────────
#  HEALTH CONTEXT DETECTION  (FIX #1 — "Hospital" bug)
# ────────────────────────────────────────────────────────────────
SYMPTOM_WORDS = {
    # English
    "fever", "temperature", "hot body", "cold", "cough", "pain", "ache",
    "vomit", "nausea", "diarrhea", "diarrhoea", "headache", "throat",
    "allergy", "rash", "itching", "itchy", "acidity", "heartburn", "sick",
    "ill", "hurt", "burning", "swelling", "breathe", "dizzy", "tired",
    "weak", "infection", "sore", "runny", "sneeze", "loose motion",
    "stool", "phlegm", "mucus", "bloating", "gas", "indigestion",
    "body ache", "muscle", "joint", "back pain", "leg pain", "migraine",
    "stomach", "chest", "breathing", "congestion", "sneezing",
    # Hindi
    "बुखार", "दर्द", "खांसी", "जुकाम", "उल्टी", "दस्त", "सिरदर्द",
    "गला", "एलर्जी", "एसिडिटी", "बीमार", "तकलीफ", "मतली", "बदन",
    "सर्दी", "बलगम", "पेट", "छाती", "सांस",
    # Arabic
    "ألم", "حمى", "سعال", "زكام", "قيء", "إسهال", "صداع", "حساسية",
    "حموضة", "مرض", "سخونة", "حرارة", "كحة", "بلغم", "غثيان",
    # Tamil
    "காய்ச்சல்", "வலி", "இருமல்", "சளி", "தலைவலி", "வாந்தி",
    # Telugu
    "జ్వరం", "నొప్పి", "దగ్గు", "జలుబు", "తలనొప్పి", "వాంతి",
}

def has_health_context(text: str) -> bool:
    """Returns True only if the message contains actual health/symptom words."""
    t = text.lower()
    t_ar = normalize_arabic(t)   # normalized form for Arabic hamza-variant matching
    for w in SYMPTOM_WORDS:
        wl = w.lower()
        if wl in t or normalize_arabic(wl) in t_ar:
            return True
    return False

# ────────────────────────────────────────────────────────────────
#  DURATION EXTRACTION
# ────────────────────────────────────────────────────────────────
_DURATION_PATTERNS = [
    # English
    (r"\b(\d+)\s*day[s]?\b",               lambda m: int(m.group(1))),
    (r"\bsince\s+yesterday\b",              lambda m: 1),
    (r"\bsince\s+morning\b",               lambda m: 0),
    (r"\bfor\s+(\d+)\s*day[s]?\b",         lambda m: int(m.group(1))),
    (r"\bfrom\s+(\d+)\s*day[s]?\b",        lambda m: int(m.group(1))),
    (r"\blast\s+(\d+)\s*day[s]?\b",        lambda m: int(m.group(1))),
    (r"\bsince\s+(\d+)\s*day[s]?\b",       lambda m: int(m.group(1))),
    (r"\b(\d+)\s*hour[s]?\b",              lambda m: 0),
    # Hindi transliteration
    (r"\b(\d+)\s*din\s*se\b",              lambda m: int(m.group(1))),
    (r"\bkal\s*se\b",                      lambda m: 1),
    # Arabic
    (r"منذ\s+يومين",                       lambda m: 2),
    (r"منذ\s+ثلاثة\s+أيام",               lambda m: 3),
    (r"منذ\s+أسبوع",                       lambda m: 7),
    (r"منذ\s+يوم",                         lambda m: 1),
    (r"منذ\s+(\d+)\s*أيام",               lambda m: int(m.group(1))),
    (r"منذ\s+(\d+)\s*يوم",                lambda m: int(m.group(1))),
    (r"من\s+يومين",                        lambda m: 2),
    (r"من\s+(\d+)\s*أيام",                lambda m: int(m.group(1))),
    (r"من\s+(\d+)\s*يوم",                 lambda m: int(m.group(1))),
    (r"يومان",                             lambda m: 2),
    (r"ثلاثة\s+أيام",                     lambda m: 3),
    (r"أسبوع",                             lambda m: 7),
    # Arabic-Indic numerals
    (r"[١-٩٠]+\s*أيام",                   lambda m: _parse_arabic_numeral(m.group(0))),
    (r"[١-٩٠]+\s*يوم",                    lambda m: _parse_arabic_numeral(m.group(0))),
]

_AR_NUMERAL_MAP = {'٠':'0','١':'1','٢':'2','٣':'3','٤':'4',
                   '٥':'5','٦':'6','٧':'7','٨':'8','٩':'9'}

def _parse_arabic_numeral(text: str) -> int:
    digits = ''.join(_AR_NUMERAL_MAP.get(c, c) for c in text if c in _AR_NUMERAL_MAP or c.isdigit())
    try:
        return int(digits)
    except ValueError:
        return 1

def extract_duration(text: str) -> Optional[int]:
    txt = text.lower()
    for pattern, extractor in _DURATION_PATTERNS:
        m = re.search(pattern, txt)
        if m:
            return extractor(m)
    return None

def duration_note(days: Optional[int], lang: str = "en") -> str:
    # 🔒 ALWAYS return English — output language is forced to English
    if days is None:
        return ""
    return f"Duration detected: {days} day{'s' if days != 1 else ''}"

# ────────────────────────────────────────────────────────────────
#  RED FLAG DETECTION
# ────────────────────────────────────────────────────────────────
_REDFLAG_PATTERNS_EN = [
    r"\bvery high fever\b", r"\bhigh fever\b", r"\bfever above 103\b",
    r"\bcan't breathe\b", r"\bbreathing difficult\b", r"\bshortness of breath\b",
    r"\bchest pain\b", r"\bchest tightness\b",
    r"\bblood in\b", r"\bbleeding\b",
    r"\bunconscious\b", r"\bfaint\b", r"\bcollapsed\b",
    r"\bseizure\b", r"\bconvulsion\b",
    r"\bsevere stomach pain\b",
    r"\bstiff neck\b",
]
_REDFLAG_PATTERNS_HI = [
    r"बहुत तेज़ बुखार", r"सांस नहीं", r"छाती में दर्द",
    r"खून आ", r"बेहोश",
]
_REDFLAG_PATTERNS_AR = [
    r"حمى شديدة", r"ضيق التنفس", r"ألم شديد في الصدر",
    r"نزيف", r"فقدان الوعي",
]

def is_red_flag(text: str) -> bool:
    t = text.lower()
    for pat in _REDFLAG_PATTERNS_EN:
        if re.search(pat, t):
            return True
    for pat in _REDFLAG_PATTERNS_HI + _REDFLAG_PATTERNS_AR:
        if re.search(pat, text):
            return True
    return False

# ────────────────────────────────────────────────────────────────
#  SYMPTOM RULES  (India-specific, EN + HI + AR)
# ────────────────────────────────────────────────────────────────
SYMPTOM_RULES: List[Dict[str, Any]] = [
    {
        "keywords_en": ["fever", "temperature", "hot body", "pyrexia"],
        "keywords_hi": ["बुखार", "तेज़ बुखार", "बदन गर्म"],
        "keywords_ar": ["حمى", "حرارة", "سخونة"],
        "symptom_detected_en": "Fever",
        "symptom_detected_hi": "बुखार",
        "symptom_detected_ar": "حمى",
        "severity": "mild",
        "medications_en": [
            {"name": "Dolo 650",    "generic": "Paracetamol 650 mg",   "note": "India's most prescribed fever tablet. 1 tablet every 6–8 hrs. Do not exceed 4/day."},
            {"name": "Crocin 650",  "generic": "Paracetamol 650 mg",   "note": "Alternative to Dolo. Available at all medical stores."},
        ],
        "medications_hi": [
            {"name": "डोलो 650",   "generic": "पेरासिटामोल 650 मि.ग्रा.", "note": "हर 6–8 घंटे में एक गोली। दिन में 4 से अधिक न लें।"},
            {"name": "क्रोसिन",   "generic": "पेरासिटामोल",              "note": "डोलो का विकल्प।"},
        ],
        "medications_ar": [
            {"name": "دولو 650",   "generic": "باراسيتامول 650 مجم",   "note": "قرص كل 6-8 ساعات. لا تتجاوز 4 أقراص يومياً."},
        ],
        "remedies_en": ["Stay hydrated — drink ORS or nimbu paani", "Cold wet cloth on forehead", "Rest well", "Lukewarm sponge bath if fever > 103°F"],
        "remedies_hi": ["ORS या नींबू पानी पियें", "माथे पर ठंडी पट्टी रखें", "आराम करें"],
        "remedies_ar": ["اشرب الكثير من السوائل", "ضع قطعة قماش باردة على الجبهة", "خذ قسطًا من الراحة"],
        "diet_en": ["Warm khichdi or daliya", "Coconut water or ORS", "Avoid spicy food"],
        "diet_hi": ["खिचड़ी या दलिया खाएं", "नारियल पानी या ORS पिएं", "मसालेदार भोजन से बचें"],
        "diet_ar": ["شوربة دافئة", "ماء جوز الهند", "تجنب الطعام الحار"],
        "warning_en": "Seek medical care if fever > 103°F, lasts > 3 days, comes with breathing difficulty, or occurs in a child under 5 or person over 60.",
        "warning_hi": "यदि बुखार 3 दिन से अधिक हो, बहुत तेज़ हो, या सांस लेने में तकलीफ हो तो तुरंत डॉक्टर से मिलें।",
        "warning_ar": "راجع الطبيب إذا استمرت الحمى أكثر من 3 أيام أو تجاوزت 39.5 درجة أو صاحبها ضيق في التنفس.",
    },
    {
        "keywords_en": ["cold", "runny nose", "sneezing", "blocked nose", "stuffy nose", "nasal congestion"],
        "keywords_hi": ["जुकाम", "नाक बह", "छींक", "सर्दी"],
        "keywords_ar": ["زكام", "رشح", "عطاس", "انسداد الأنف"],
        "symptom_detected_en": "Common Cold",
        "symptom_detected_hi": "जुकाम / सर्दी",
        "symptom_detected_ar": "زكام",
        "severity": "mild",
        "medications_en": [
            {"name": "D-Cold Total",   "generic": "Paracetamol + Phenylephrine + Cetirizine", "note": "Popular India combo for cold. 1 tablet at night (causes drowsiness)."},
            {"name": "Nasivion nasal spray", "generic": "Oxymetazoline 0.05%", "note": "Relieves nasal congestion fast. Max 3 days use."},
        ],
        "medications_hi": [
            {"name": "डी-कोल्ड टोटल", "generic": "पेरासिटामोल + फिनाइलफ्रिन", "note": "रात में एक गोली। नींद आ सकती है।"},
            {"name": "नेसीवियन नेज़ल स्प्रे", "generic": "Oxymetazoline", "note": "बंद नाक के लिए। 3 दिन से अधिक न लें।"},
        ],
        "medications_ar": [
            {"name": "سيتريزين",        "generic": "Cetirizine 10 mg",           "note": "يساعد في العطاس وسيلان الأنف."},
        ],
        "remedies_en": ["Steam inhalation with Vicks/Tulsi", "Haldi milk (turmeric milk) at night", "Ginger-honey-lemon tea"],
        "remedies_hi": ["भाप लें (विक्स डालकर)", "रात को हल्दी वाला दूध पिएं", "अदरक–शहद–नींबू चाय"],
        "remedies_ar": ["استنشاق البخار مع فيكس", "شاي الزنجبيل مع العسل"],
        "diet_en": ["Warm soup or kadha", "Avoid cold drinks and ice cream", "Tulsi-ginger tea"],
        "diet_hi": ["काढ़ा पिएं", "ठंडी चीज़ें न खाएं", "तुलसी–अदरक की चाय"],
        "diet_ar": ["شوربة دافئة", "تجنب المشروبات الباردة"],
        "warning_en": "See a doctor if cold lasts > 7 days, fever develops, or breathing becomes difficult.",
        "warning_hi": "7 दिन से अधिक जुकाम रहे, बुखार आए या सांस में तकलीफ हो तो डॉक्टर से मिलें।",
        "warning_ar": "راجع الطبيب إذا استمر الزكام أكثر من 7 أيام أو ظهرت حمى.",
    },
    {
        "keywords_en": ["headache", "head pain", "migraine", "eye strain", "head ache"],
        "keywords_hi": ["सिरदर्द", "सिर दर्द", "माइग्रेन"],
        "keywords_ar": ["صداع", "ألم الرأس", "شقيقة"],
        "symptom_detected_en": "Headache",
        "symptom_detected_hi": "सिरदर्द",
        "symptom_detected_ar": "صداع",
        "severity": "mild",
        "medications_en": [
            {"name": "Saridon",   "generic": "Paracetamol + Propyphenazone + Caffeine", "note": "Popular India headache tablet. 1 tablet with water."},
            {"name": "Crocin 500", "generic": "Paracetamol 500 mg", "note": "Mild headache relief."},
        ],
        "medications_hi": [
            {"name": "सेरिडॉन",   "generic": "पेरासिटामोल + कैफीन", "note": "सिरदर्द की आम दवाई।"},
        ],
        "medications_ar": [
            {"name": "باراسيتامول", "generic": "Paracetamol 500 mg", "note": "استخدمه حسب الإرشادات."},
        ],
        "remedies_en": ["Rest in a dark, quiet room", "Cold compress on forehead", "Reduce screen time", "Peppermint oil on temples"],
        "remedies_hi": ["अंधेरे शांत कमरे में आराम करें", "माथे पर ठंडी पट्टी रखें", "स्क्रीन टाइम कम करें"],
        "remedies_ar": ["ارتَح في غرفة هادئة ومعتمة", "كمادة باردة على الجبهة"],
        "diet_en": ["Drink plenty of water", "Avoid skipping meals", "Reduce caffeine"],
        "diet_hi": ["खूब पानी पिएं", "भोजन न छोड़ें", "चाय/कॉफी कम करें"],
        "diet_ar": ["اشرب الكثير من الماء", "لا تتخطى وجباتك"],
        "warning_en": "Seek emergency care for sudden severe ('thunderclap') headache, headache with stiff neck, fever, confusion, or vision changes.",
        "warning_hi": "अचानक बहुत तेज़ सिरदर्द, गर्दन अकड़न या बुखार के साथ हो तो तुरंत डॉक्टर के पास जाएं।",
        "warning_ar": "اطلب الطوارئ فوراً إذا كان الصداع مفاجئاً وشديداً أو مصاحباً لتصلب الرقبة أو حمى.",
    },
    {
        "keywords_en": ["diarrhoea", "diarrhea", "loose motions", "stomach upset", "loose stool", "watery stool"],
        "keywords_hi": ["दस्त", "लूज मोशन", "पेट खराब", "पेचिश"],
        "keywords_ar": ["إسهال", "براز سائل", "انتفاخ"],
        "symptom_detected_en": "Diarrhoea / Loose Motions",
        "symptom_detected_hi": "दस्त / लूज़ मोशन",
        "symptom_detected_ar": "إسهال",
        "severity": "moderate",
        "medications_en": [
            {"name": "Electral ORS",  "generic": "Oral Rehydration Salts", "note": "Most important — prevents dehydration. Mix 1 sachet in 1L water."},
            {"name": "Sporlac DS",    "generic": "Lactobacillus (Probiotic)", "note": "Restores gut flora. Safe for all ages."},
        ],
        "medications_hi": [
            {"name": "इलेक्ट्राल ORS", "generic": "Oral Rehydration Salts", "note": "1 लीटर पानी में एक पैकेट घोलकर पिएं।"},
            {"name": "स्पोर्लैक",       "generic": "प्रोबायोटिक",           "note": "आंत को ठीक करने में मदद करता है।"},
        ],
        "medications_ar": [
            {"name": "محلول الإماهة الفموية ORS", "generic": "ORS", "note": "أساسي لتعويض السوائل. كيس في لتر ماء."},
        ],
        "remedies_en": ["Drink ORS every 30 min", "BRAT diet (Banana, Rice, Applesauce, Toast)", "Avoid dairy and oily food"],
        "remedies_hi": ["हर आधे घंटे में ORS पिएं", "केला, चावल, दही खाएं", "तेल-मसाला बंद करें"],
        "remedies_ar": ["تناول ORS كل 30 دقيقة", "حمية BRAT (موز، أرز، تفاح، خبز محمص)"],
        "diet_en": ["Khichdi (rice+dal)", "Curd/yoghurt", "Banana", "Coconut water"],
        "diet_hi": ["खिचड़ी (चावल + दाल)", "दही", "केला", "नारियल पानी"],
        "diet_ar": ["أرز سادة", "زبادي", "موز", "ماء جوز الهند"],
        "warning_en": "See a doctor if blood in stool, high fever, severe cramps, vomiting or dehydration signs (sunken eyes, no urination).",
        "warning_hi": "मल में खून, तेज़ बुखार, उल्टी या पेशाब बंद हो जाए तो तुरंत डॉक्टर के पास जाएं।",
        "warning_ar": "راجع الطبيب فوراً إذا وُجد دم في البراز أو حمى أو علامات جفاف.",
    },
    {
        "keywords_en": ["vomiting", "nausea", "vomit", "throwing up", "feel like vomiting"],
        "keywords_hi": ["उल्टी", "मतली", "जी मचलाना", "मितली"],
        "keywords_ar": ["قيء", "غثيان", "إقياء"],
        "symptom_detected_en": "Nausea / Vomiting",
        "symptom_detected_hi": "उल्टी / मतली",
        "symptom_detected_ar": "غثيان وقيء",
        "severity": "mild",
        "medications_en": [
            {"name": "Perinorm (Metoclopramide)", "generic": "Metoclopramide 10 mg", "note": "Relieves nausea. 1 tablet 30 min before meals. Max 3/day."},
            {"name": "Emetrol",                  "generic": "Fructose-based syrup",   "note": "OTC syrup for nausea. Safe for most adults."},
        ],
        "medications_hi": [
            {"name": "पेरिनॉर्म",  "generic": "मेटोक्लोप्रामाइड", "note": "खाने से पहले एक गोली।"},
        ],
        "medications_ar": [
            {"name": "ميتوكلوبراميد", "generic": "Metoclopramide", "note": "قرص قبل الأكل بـ30 دقيقة."},
        ],
        "remedies_en": ["Sip cold water or ORS slowly", "Ginger tea with honey", "Eat small frequent meals", "Avoid lying down after eating"],
        "remedies_hi": ["ठंडा पानी या ORS धीरे-धीरे पिएं", "अदरक की चाय", "थोड़ा-थोड़ा खाएं"],
        "remedies_ar": ["اشرب الماء البارد ببطء", "شاي الزنجبيل بالعسل", "وجبات صغيرة ومتكررة"],
        "diet_en": ["Plain rice or crackers", "Banana", "Cold nimbu soda (lime soda)", "Avoid fatty or spicy food"],
        "diet_hi": ["सादा चावल या बिस्कुट", "केला", "ठंडा नींबू पानी"],
        "diet_ar": ["أرز سادة", "موز", "تجنب الطعام الدهني"],
        "warning_en": "See a doctor if vomiting is persistent (> 24 hrs), blood present, severe abdominal pain, or signs of dehydration.",
        "warning_hi": "24 घंटे से अधिक उल्टी हो, खून आए या तेज़ पेट दर्द हो तो डॉक्टर से मिलें।",
        "warning_ar": "راجع الطبيب إذا استمر القيء أكثر من 24 ساعة أو كان فيه دم.",
    },
    {
        "keywords_en": ["sore throat", "throat pain", "pain while swallowing", "throat infection"],
        "keywords_hi": ["गले में दर्द", "गला खराब", "गले में खराश", "निगलने में दर्द"],
        "keywords_ar": ["التهاب الحلق", "ألم الحلق", "ألم عند البلع"],
        "symptom_detected_en": "Sore Throat",
        "symptom_detected_hi": "गला दर्द / खराश",
        "symptom_detected_ar": "التهاب الحلق",
        "severity": "mild",
        "medications_en": [
            {"name": "Strepsils",        "generic": "Amylmetacresol lozenge", "note": "Soothing throat lozenge. 1 every 3–4 hrs."},
            {"name": "Betadine Gargle",  "generic": "Povidone Iodine 1%",    "note": "Dilute 1:3 with water. Gargle twice daily."},
        ],
        "medications_hi": [
            {"name": "स्ट्रेप्सिल्स", "generic": "Throat lozenge",        "note": "हर 3–4 घंटे में एक।"},
            {"name": "बेटाडीन गार्गल", "generic": "Povidone Iodine",     "note": "पानी में मिलाकर दिन में दो बार गरारा करें।"},
        ],
        "medications_ar": [
            {"name": "ستريبسلز", "generic": "Throat lozenge", "note": "قرص كل 3-4 ساعات."},
        ],
        "remedies_en": ["Warm salt-water gargle 3x/day", "Honey with warm water", "Turmeric milk at night"],
        "remedies_hi": ["गरम नमक पानी से गरारा करें", "गरम पानी में शहद मिलाएं", "रात को हल्दी वाला दूध"],
        "remedies_ar": ["غرغرة بماء دافئ ومملح 3 مرات/يوم", "عسل مع ماء دافئ"],
        "diet_en": ["Warm soup or kadha", "Soft foods", "Warm herbal tea (tulsi, ginger)"],
        "diet_hi": ["गरम सूप या काढ़ा", "मुलायम खाना", "तुलसी–अदरक की चाय"],
        "diet_ar": ["شوربة دافئة", "أطعمة لينة"],
        "warning_en": "See a doctor if throat pain is severe, difficulty breathing, swollen tonsils with white patches, or high fever.",
        "warning_hi": "गला बहुत ज़्यादा दर्द हो, सांस में तकलीफ हो या बुखार हो तो डॉक्टर के पास जाएं।",
        "warning_ar": "راجع الطبيب إذا كان ألم الحلق شديداً أو كان هناك صعوبة في التنفس.",
    },
    {
        "keywords_en": ["acidity", "acid reflux", "heartburn", "burning chest", "gas", "bloating", "indigestion"],
        "keywords_hi": ["एसिडिटी", "जलन", "गैस", "अपच", "खट्टी डकार"],
        "keywords_ar": ["حموضة", "ارتجاع", "حرقة", "انتفاخ"],
        "symptom_detected_en": "Acidity / Heartburn",
        "symptom_detected_hi": "एसिडिटी / जलन",
        "symptom_detected_ar": "حموضة",
        "severity": "mild",
        "medications_en": [
            {"name": "Digene Gel",    "generic": "Aluminium Hydroxide + Magnesium", "note": "India's #1 antacid. 2 tsp after meals."},
            {"name": "Pan-D",         "generic": "Pantoprazole + Domperidone",      "note": "For frequent acidity. 1 tablet before breakfast."},
        ],
        "medications_hi": [
            {"name": "डाइजीन जेल", "generic": "एल्युमीनियम हाइड्रॉक्साइड", "note": "खाने के बाद 2 चम्मच लें।"},
            {"name": "पैन-डी",      "generic": "Pantoprazole",                "note": "नाश्ते से पहले एक गोली।"},
        ],
        "medications_ar": [
            {"name": "ديجين",    "generic": "Antacid gel",   "note": "ملعقتان بعد الأكل."},
        ],
        "remedies_en": ["Cold milk or lassi", "Ajwain (carom seeds) with warm water", "Avoid lying down right after eating"],
        "remedies_hi": ["ठंडा दूध या लस्सी", "अजवाइन गरम पानी के साथ", "खाने के तुरंत बाद न लेटें"],
        "remedies_ar": ["حليب بارد أو لبن", "يانسون مع ماء دافئ"],
        "diet_en": ["Avoid spicy, oily, fried food", "Eat smaller meals", "Avoid tea/coffee on empty stomach"],
        "diet_hi": ["मसालेदार, तला भोजन बंद करें", "छोटे-छोटे भोजन लें", "खाली पेट चाय/कॉफी न पिएं"],
        "diet_ar": ["تجنب الطعام الحار والمقلي", "وجبات صغيرة", "تجنب القهوة على معدة فارغة"],
        "warning_en": "See a doctor if chest burning is severe, persistent, or mimics chest pain — could be cardiac.",
        "warning_hi": "सीने में जलन बहुत तेज़ हो या लगातार बनी रहे तो डॉक्टर से मिलें।",
        "warning_ar": "راجع الطبيب إذا كانت حرقة الصدر شديدة أو مستمرة.",
    },
    {
        "keywords_en": ["body pain", "muscle pain", "joint pain", "back pain", "leg pain", "body ache", "arthritis"],
        "keywords_hi": ["बदन दर्द", "मांसपेशियों में दर्द", "जोड़ों में दर्द", "कमर दर्द", "पैर में दर्द"],
        "keywords_ar": ["ألم الجسم", "ألم العضلات", "ألم المفاصل", "آلام الظهر"],
        "symptom_detected_en": "Body / Muscle Pain",
        "symptom_detected_hi": "बदन / मांसपेशी दर्द",
        "symptom_detected_ar": "آلام الجسم",
        "severity": "mild",
        "medications_en": [
            {"name": "Combiflam",     "generic": "Ibuprofen 400 mg + Paracetamol 325 mg", "note": "Popular India pain tablet. Take after food. Max 3/day."},
            {"name": "Volini Spray",  "generic": "Diclofenac topical spray",              "note": "Apply on painful area. Fast local relief."},
        ],
        "medications_hi": [
            {"name": "कॉम्बीफ्लेम",   "generic": "आइबुप्रोफेन + पेरासिटामोल", "note": "खाने के बाद एक गोली।"},
            {"name": "वोलिनी स्प्रे", "generic": "Diclofenac spray",       "note": "दर्द वाली जगह पर लगाएं।"},
        ],
        "medications_ar": [
            {"name": "كومبيفلام", "generic": "Ibuprofen + Paracetamol", "note": "قرص بعد الأكل."},
        ],
        "remedies_en": ["Warm compress or hot water bottle on area", "Gentle stretching/massage", "Rest and hydration"],
        "remedies_hi": ["गरम पानी की बोतल लगाएं", "हल्की मालिश करें", "आराम करें और पानी पिएं"],
        "remedies_ar": ["كمادة دافئة", "تدليك لطيف", "راحة وترطيب"],
        "diet_en": ["Anti-inflammatory foods: turmeric milk, ginger tea", "Stay well hydrated"],
        "diet_hi": ["हल्दी वाला दूध", "अदरक की चाय", "खूब पानी पिएं"],
        "diet_ar": ["حليب الكركم", "شاي الزنجبيل", "اشرب الكثير من الماء"],
        "warning_en": "See a doctor if pain is severe, persistent, comes with swelling, redness or fever.",
        "warning_hi": "दर्द बहुत तेज़ हो, सूजन हो या बुखार के साथ हो तो डॉक्टर से मिलें।",
        "warning_ar": "راجع الطبيب إذا كان الألم شديداً أو مصاحباً لتورم أو حمى.",
    },
    {
        "keywords_en": ["cough", "dry cough", "wet cough", "phlegm", "mucus", "productive cough"],
        "keywords_hi": ["खांसी", "सूखी खांसी", "कफ वाली खांसी", "बलगम"],
        "keywords_ar": ["سعال", "كحة", "بلغم"],
        "symptom_detected_en": "Cough",
        "symptom_detected_hi": "खांसी",
        "symptom_detected_ar": "سعال",
        "severity": "mild",
        "medications_en": [
            {"name": "Benadryl (dry cough)",  "generic": "Diphenhydramine syrup",         "note": "For dry tickly cough. 2 tsp at bedtime."},
            {"name": "Alex (wet cough)",      "generic": "Bromhexine + Guaifenesin",       "note": "For cough with phlegm. 2 tsp 3x/day."},
            {"name": "Honitus Syrup",         "generic": "Tulsi-based ayurvedic cough syrup","note": "Natural option. 2 tsp 3x/day."},
        ],
        "medications_hi": [
            {"name": "बेनाड्रिल",    "generic": "Diphenhydramine", "note": "सूखी खांसी के लिए, रात में 2 चम्मच।"},
            {"name": "एलेक्स",       "generic": "Bromhexine",      "note": "बलगम वाली खांसी के लिए।"},
        ],
        "medications_ar": [
            {"name": "بيناد ريل",    "generic": "Diphenhydramine", "note": "للسعال الجاف. ملعقتان عند النوم."},
        ],
        "remedies_en": ["Honey + ginger juice", "Steam inhalation with Vicks / tulsi leaves", "Warm water with turmeric and pepper"],
        "remedies_hi": ["शहद + अदरक का रस पिएं", "विक्स के साथ भाप लें", "गरम पानी में हल्दी और काली मिर्च"],
        "remedies_ar": ["عسل وزنجبيل", "استنشاق البخار مع فيكس"],
        "diet_en": ["Warm fluids throughout day", "Avoid cold drinks and ice cream", "Tulsi-ginger-honey tea"],
        "diet_hi": ["दिनभर गरम पानी/काढ़ा पिएं", "ठंडी चीज़ें बंद करें"],
        "diet_ar": ["مشروبات دافئة طوال اليوم", "تجنب المشروبات الباردة"],
        "warning_en": "See a doctor if cough persists > 2 weeks, blood in phlegm, chest pain, or breathlessness.",
        "warning_hi": "2 हफ़्ते से ज़्यादा खांसी हो, बलगम में खून हो या सांस में तकलीफ हो तो डॉक्टर के पास जाएं।",
        "warning_ar": "راجع الطبيب إذا استمر السعال أكثر من أسبوعين أو كان معه دم.",
    },
    {
        "keywords_en": ["allergy", "itching", "hives", "skin rash", "rash", "itchy skin"],
        "keywords_hi": ["एलर्जी", "खुजली", "चकत्ते", "रैश", "त्वचा पर दाने"],
        "keywords_ar": ["حساسية", "حكة", "طفح جلدي", "شرى"],
        "symptom_detected_en": "Allergy / Skin Rash",
        "symptom_detected_hi": "एलर्जी / खुजली",
        "symptom_detected_ar": "حساسية",
        "severity": "mild",
        "medications_en": [
            {"name": "Cetirizine (Zyrtec/Cetzine)", "generic": "Cetirizine 10 mg", "note": "1 tablet at night (causes drowsiness). Effective for allergies and hives."},
            {"name": "Calamine Lotion",              "generic": "Calamine",          "note": "Apply on rash/hives for soothing relief. Safe for skin."},
        ],
        "medications_hi": [
            {"name": "सेटीरिज़ीन (सेटज़ीन)", "generic": "Cetirizine 10 mg", "note": "रात में एक गोली।"},
            {"name": "कैलामाइन लोशन",         "generic": "Calamine",         "note": "रैश पर लगाएं।"},
        ],
        "medications_ar": [
            {"name": "سيتريزين",    "generic": "Cetirizine 10 mg", "note": "قرص في الليل."},
            {"name": "لوشن كلامين", "generic": "Calamine",         "note": "ضعه على الطفح الجلدي."},
        ],
        "remedies_en": ["Cool compress on affected area", "Avoid scratching", "Identify and avoid trigger (dust, food, pollen)"],
        "remedies_hi": ["ठंडी पट्टी लगाएं", "खुजलाएं मत", "एलर्जन से बचें (धूल, परागकण, भोजन)"],
        "remedies_ar": ["كمادة باردة", "لا تحك", "تجنب مسببات الحساسية"],
        "diet_en": ["Drink plenty of water", "Avoid known trigger foods (shellfish, peanuts, dairy if allergic)"],
        "diet_hi": ["खूब पानी पिएं", "जिन चीज़ों से एलर्जी हो उन्हें न खाएं"],
        "diet_ar": ["اشرب الكثير من الماء", "تجنب الأطعمة المثيرة"],
        "warning_en": "EMERGENCY: Swelling of lips/tongue/throat, difficulty breathing after exposure = anaphylaxis. Call 108 immediately.",
        "warning_hi": "आपातकाल: होंठ, जीभ या गले में सूजन हो और सांस लेने में तकलीफ हो तो तुरंत 108 कॉल करें।",
        "warning_ar": "طوارئ: تورم الشفاه أو اللسان مع ضيق التنفس = صدمة تحسسية. اتصل بالإسعاف فوراً.",
    },
]

# ────────────────────────────────────────────────────────────────
#  CACHE
# ────────────────────────────────────────────────────────────────
class Cache:
    def __init__(self, max_size: int = 200):
        self.store    = OrderedDict()
        self.max_size = max_size

    def key(self, text: str):
        return hashlib.md5(text.lower().encode()).hexdigest()

    def get(self, text: str):
        return self.store.get(self.key(text))

    def set(self, text: str, value: Any):
        k = self.key(text)
        self.store[k] = value
        if len(self.store) > self.max_size:
            self.store.popitem(last=False)

cache = Cache()

# ────────────────────────────────────────────────────────────────
#  RULE MATCHING
# ────────────────────────────────────────────────────────────────
def match_rule(message: str) -> Optional[Dict[str, Any]]:
    msg = message.lower()
    msg_ar = normalize_arabic(msg)   # normalized for Arabic hamza-variant matching
    for rule in SYMPTOM_RULES:
        if (any(k.lower() in msg for k in rule.get("keywords_en", []))
                or any(k in message for k in rule.get("keywords_hi", []))
                or any(normalize_arabic(k) in msg_ar for k in rule.get("keywords_ar", []))):
            return rule
    return None

def build_rule_response(message: str, lang: str, days: Optional[int] = None) -> Dict[str, Any]:
    # FIX: Do NOT override lang — use the passed-in language
    rule = match_rule(message)

    def maybe_upgrade_severity(base: str) -> str:
        if days is not None and days >= 3 and base == "mild":
            return "moderate"
        if days is not None and days >= 5:
            return "see_doctor_now"
        return base

    if not rule:
        # FIX: No rule match + no health context → ask user to describe symptoms
        if not has_health_context(message):
            no_symptom_msg = {
                "en": "Please describe your symptoms (e.g. fever, cold, headache, cough) to receive medicine suggestions.",
                "hi": "कृपया अपने लक्षण बताएं (जैसे बुखार, खांसी, सिरदर्द) ताकि हम सही सुझाव दे सकें।",
                "ar": "يرجى وصف أعراضك (مثل الحمى، السعال، الصداع) للحصول على التوصيات.",
                "ta": "உங்கள் அறிகுறிகளை விவரிக்கவும் (காய்ச்சல், இருமல், தலைவலி போன்றவை).",
                "te": "దయచేసి మీ లక్షణాలను వివరించండి (జ్వరం, దగ్గు, తలనొప్పి వంటివి).",
            }
            return {
                "symptom_detected": "No symptoms detected",
                "severity": "mild",
                "duration_note": "",
                "home_remedies": [],
                "medications": [],
                "diet_tips": [],
                "warning": no_symptom_msg.get(lang, no_symptom_msg["en"]),
                "red_flag": False,
            }

        # Has health context but no specific rule → generic fallback WITH meds
        generic_label = {
            "en": "General Symptoms", "hi": "सामान्य लक्षण",
            "ar": "أعراض عامة",       "ta": "பொது அறிகுறிகள்",
            "te": "సాధారణ లక్షణాలు",
        }
        generic_meds = {
            "en": [{"name": "Dolo 650", "generic": "Paracetamol 650 mg", "note": "Take 1 tablet every 6–8 hrs as needed."}],
            "hi": [{"name": "डोलो 650", "generic": "पेरासिटामोल 650 मि.ग्रा.", "note": "आवश्यकतानुसार 6-8 घंटे में एक।"}],
            "ar": [{"name": "دولو 650", "generic": "باراسيتامول 650 مجم", "note": "قرص كل 6-8 ساعات حسب الحاجة."}],
            "ta": [{"name": "Dolo 650", "generic": "Paracetamol 650 mg", "note": "6–8 மணி நேரத்திற்கு ஒரு மாத்திரை."}],
            "te": [{"name": "Dolo 650", "generic": "Paracetamol 650 mg", "note": "6–8 గంటలకు ఒక మాత్ర తీసుకోండి."}],
        }
        generic_remedies = {
            "en": ["Rest and stay hydrated"],
            "hi": ["आराम करें और पानी पिएं"],
            "ar": ["الراحة وشرب السوائل"],
            "ta": ["ஓய்வெடுங்கள், நீர் அருந்துங்கள்"],
            "te": ["విశ్రాంతి తీసుకోండి మరియు నీరు తాగండి"],
        }
        generic_diet = {
            "en": ["Drink plenty of water"],
            "hi": ["खूब पानी पिएं"],
            "ar": ["اشرب الكثير من الماء"],
            "ta": ["நிறைய தண்ணீர் குடிக்கவும்"],
            "te": ["నీటిని పుష్కలంగా తాగండి"],
        }
        generic_warning = {
            "en": "If symptoms worsen or persist > 3 days, consult a doctor.",
            "hi": "यदि लक्षण बिगड़ें या 3 दिन से अधिक रहें तो डॉक्टर से मिलें।",
            "ar": "إذا ساءت الأعراض أو استمرت أكثر من 3 أيام راجع الطبيب.",
            "ta": "அறிகுறிகள் மோசமாகினால் அல்லது 3 நாட்களுக்கு மேல் நீடித்தால் மருத்துவரை அணுகவும்.",
            "te": "లక్షణాలు తీవ్రమైతే లేదా 3 రోజులకంటే ఎక్కువగా కొనసాగితే వైద్యుడిని సంప్రదించండి.",
        }
        return {
            "symptom_detected": generic_label.get(lang, "General Symptoms"),
            "severity": maybe_upgrade_severity("mild"),
            "duration_note": duration_note(days, lang),
            "home_remedies": generic_remedies.get(lang, generic_remedies["en"]),
            "medications": generic_meds.get(lang, generic_meds["en"]),
            "diet_tips": generic_diet.get(lang, generic_diet["en"]),
            "warning": generic_warning.get(lang, generic_warning["en"]),
            "red_flag": False,
        }

    # Pick the right language suffix; Tamil/Telugu fall back to English rules
    suffix = "ar" if lang == "ar" else ("hi" if lang == "hi" else "en")
    meds      = rule.get(f"medications_{suffix}", rule.get("medications_en", []))
    remedies  = rule.get(f"remedies_{suffix}",    rule.get("remedies_en", []))
    diet      = rule.get(f"diet_{suffix}",        rule.get("diet_en", []))
    symptom_d = rule.get(f"symptom_detected_{suffix}", rule.get("symptom_detected_en", "Symptoms"))
    warning   = rule.get(f"warning_{suffix}",     rule.get("warning_en", ""))

    return {
        "symptom_detected": symptom_d,
        "severity": maybe_upgrade_severity(rule["severity"]),
        "duration_note": duration_note(days, lang),
        "home_remedies": remedies,
        "medications": meds,
        "diet_tips": diet,
        "warning": warning,
        "red_flag": False,
    }

# ────────────────────────────────────────────────────────────────
#  LLM CALL (fallback + Tamil/Telugu translation)
# ────────────────────────────────────────────────────────────────
async def call_llm(message: str, lang: str) -> Dict[str, Any]:
    payload = {
        "model": MODEL,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Language: {lang}\nUser message: {message}"},
        ],
    }
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(OPENAI_URL, headers=OPENAI_HEADERS, json=payload)
        r.raise_for_status()
    return json.loads(r.json()["choices"][0]["message"]["content"])

async def call_llm_translate(rule_result: Dict[str, Any], message: str, lang: str) -> Dict[str, Any]:
    """Ask LLM to translate an existing rule result into Tamil/Telugu."""
    lang_name = "Tamil" if lang == "ta" else "Telugu"
    payload = {
        "model": MODEL,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Language: {lang} ({lang_name})\n"
                    f"User's original message: {message}\n"
                    f"Translate ALL text fields of this JSON into {lang_name} script. "
                    f"Keep medicine names (Dolo 650, Crocin etc.) and dosage numbers as-is. "
                    f"Return the same JSON structure with translated text:\n"
                    f"{json.dumps(rule_result, ensure_ascii=False)}"
                ),
            },
        ],
    }
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(OPENAI_URL, headers=OPENAI_HEADERS, json=payload)
        r.raise_for_status()
    return json.loads(r.json()["choices"][0]["message"]["content"])

def normalize_result(result: Dict[str, Any], lang: str, days: Optional[int] = None) -> Dict[str, Any]:
    if not isinstance(result, dict):
        result = {}
    result.setdefault("symptom_detected", "Symptoms")
    result.setdefault("severity", "mild")
    result.setdefault("duration_note", duration_note(days, lang))
    result.setdefault("home_remedies", [])
    result.setdefault("medications", [])
    result.setdefault("diet_tips", [])
    result.setdefault("warning", "")
    result.setdefault("red_flag", False)

    if result["severity"] not in ["mild", "moderate", "see_doctor_now"]:
        result["severity"] = "mild"
    if not isinstance(result["medications"], list):
        result["medications"] = []
    # FIX: Removed auto-injection of fever meds when no medications exist.
    # If no meds are returned, leave it empty — the UI will show "No medicines — consult a doctor"
    return result


# ────────────────────────────────────────────────────────────────
#  PROCESS REQUEST
# ────────────────────────────────────────────────────────────────
async def process(message: str, lang: str) -> Dict[str, Any]:
    cache_key = f"{lang}:{message.lower()}"
    cached = cache.get(cache_key)
    if cached:
        cached["_cached"] = True
        return cached

    days = extract_duration(message)

    # STEP 1: RED FLAG FIRST
    if is_red_flag(message):
        # lang is always "en" for voice input (normalized upstream).
        # Keep the dict for /symptom endpoint edge-cases but default to English.
        warnings = {
            "en": "⚠️ RED FLAG: Your symptoms may require emergency care. Please call 108 (India) or visit the nearest hospital immediately.",
            "hi": "⚠️ रेड फ्लैग: आपके लक्षण आपातकालीन देखभाल का संकेत देते हैं। कृपया तुरंत 108 पर कॉल करें या नजदीकी अस्पताल जाएं।",
            "ar": "⚠️ تحذير عاجل: أعراضك قد تستدعي رعاية طارئة.",
            "ta": "⚠️ எச்சரிக்கை: உங்கள் அறிகுறிகளுக்கு அவசர சிகிச்சை தேவைப்படலாம்.",
            "te": "⚠️ హెచ్చరిక: మీ లక్షణాలకు అత్యవసర సంరక్షణ అవసరం కావచ్చు.",
        }
        result = {
            "symptom_detected": "Red Flag Symptoms",
            "severity": "see_doctor_now",
            "duration_note": duration_note(days, "en"),   # 🔒 always English
            "home_remedies": [],
            "medications": [],
            "diet_tips": [],
            "warning": warnings.get(lang, warnings["en"]),
            "red_flag": True,
            "_cached": False,
        }
        cache.set(cache_key, result)
        return result

    # STEP 2: BASE RULE RESPONSE (respects language)
    rule_result = build_rule_response(message, lang, days)

    # STEP 3: If no symptoms detected, return immediately (no meds needed)
    if rule_result.get("symptom_detected") == "No symptoms detected":
        rule_result["_cached"] = False
        cache.set(cache_key, rule_result)
        return rule_result

    # STEP 4: For Tamil/Telugu, translate via LLM
    if lang in ("ta", "te"):
        try:
            translated = await call_llm_translate(rule_result, message, lang)
            translated = normalize_result(translated, lang, days)
            translated["_cached"] = False
            cache.set(cache_key, translated)
            return translated
        except Exception:
            pass  # fall through to rule result

    # STEP 5: GPT-CLASSIFIED CIPLA SEARCH → FILL WITH GPT (total = 3)
    final_meds = await get_final_medicines(message)

    # If get_final_medicines returned nothing, fall back to rule-based meds
    if not final_meds:
        final_meds = rule_result.get("medications", [])

    rule_result["medications"] = final_meds[:3]
    rule_result["_cached"] = False
    cache.set(cache_key, rule_result)
    return rule_result


# ────────────────────────────────────────────────────────────────
#  FASTAPI APP
# ────────────────────────────────────────────────────────────────
app = FastAPI(title="Healthcare Voice Assistant API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Req(BaseModel):
    message:  str
    age:      Optional[int]  = None
    gender:   Optional[str]  = None
    language: Optional[str]  = None  # auto-detect if omitted

@app.get("/")
def home():
    return FileResponse("healthcare_index.html")

@app.get("/favicon.ico")
def icon():
    return Response(status_code=204)

@app.post("/voice")
async def voice_input(file: UploadFile = File(...), lang: str = "auto"):
    """
    Accepts multipart audio + optional ?lang=hi|ar|ta|te|en|auto query param.
    Uses OpenAI gpt-4o-transcribe for multi-language speech recognition.
    Passing the correct language improves transcription accuracy.
    """

    # Validate lang
    valid_langs = ("en", "hi", "ar", "ta", "te", "auto")
    if lang not in valid_langs:
        lang = "auto"

    try:
        audio_bytes = await file.read()
        logger.info(f"Received audio: {len(audio_bytes)} bytes, requested lang={lang!r}")

        # transcribe_audio now returns (transcript, detected_lang) tuple
        transcript, detected_lang = await transcribe_audio(audio_bytes, lang)

        if not transcript.strip():
            no_speech_msgs = {
                "en": "No speech detected. Please speak clearly and try again.",
                "hi": "कोई वाणी नहीं मिली। कृपया स्पष्ट रूप से बोलें।",
                "ar": "لم يتم اكتشاف أي كلام. يرجى التحدث بوضوح.",
                "ta": "பேச்சு கண்டறியப்படவில்லை. தெளிவாக பேசவும்.",
                "te": "మాట గుర్తించబడలేదు. స్పష్టంగా మాట్లాడండి.",
            }
            detected = detected_lang or "en"
            return {
                "symptom_detected": "No speech detected",
                "severity": "mild",
                "duration_note": "",
                "home_remedies": [],
                "medications": [],
                "diet_tips": [],
                "warning": no_speech_msgs.get(detected, no_speech_msgs["en"]),
                "red_flag": False,
                "_transcript": "",
                "_lang": detected,
            }

        logger.info(f"Processing transcript as lang={detected_lang!r}")

        # 🔒 ALWAYS FORCE ENGLISH OUTPUT
        # Normalize transcript to English BEFORE hitting medical rules.
        # This ensures build_rule_response always picks _en fields regardless
        # of what language the user spoke.
        english_transcript = await normalize_to_english(transcript)
        if english_transcript != transcript:
            logger.info(f"Normalized to English: {english_transcript!r}")

        result = await process(english_transcript, "en")   # ← always "en"
        result["_transcript"]       = english_transcript   # show English on UI
        result["_original_speech"]  = transcript           # keep raw for debugging
        result["_lang"]             = "en"                 # tell frontend: always English
        return result

    except httpx.HTTPStatusError as e:
        logger.error(f"OpenAI Transcription API error: {e.response.status_code} — {e.response.text}")
        raise HTTPException(status_code=502, detail=f"OpenAI transcription error: {e.response.text}")
    except Exception as e:
        logger.error(f"Voice endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/symptom")
async def symptom(req: Req):
    msg = req.message.strip()
    if not msg:
        raise HTTPException(400, "Empty message")

    # Language: prefer explicit from client, else auto-detect from text
    valid_langs = ("en", "hi", "ar", "ta", "te")
    lang = req.language if req.language in valid_langs else detect_language(msg)

    if req.age:
        msg += f" (age: {req.age})"
    if req.gender:
        msg += f" (gender: {req.gender})"

    try:
        # For Arabic/Hindi/Tamil/Telugu input via the /symptom endpoint,
        # normalize to English first so build_rule_response uses _en fields
        # and the LLM fallback gets clean English input.
        if lang in ("ar", "hi", "ta", "te"):
            english_msg = msg  # default fallback = original text
            try:
                english_msg = await normalize_to_english(msg)
                logger.info(f"/symptom normalized [{lang}→en]: {english_msg!r}")
            except Exception as e:
                logger.warning(f"normalize_to_english failed in /symptom: {e} — using raw")
            result = await process(english_msg, "en")
            result["_original_text"] = msg
            result["_english_text"] = english_msg  # always set for UI bubble
            result["_lang"] = lang
            return result
        # English: process directly but still set _english_text so the frontend
        # always has a reliable field — it equals exactly what the user said.
        result = await process(msg, lang)
        result["_english_text"] = msg   # exact transcript, no modifications
        result["_lang"] = lang
        return result
    except httpx.HTTPStatusError:
        raise HTTPException(502, "LLM API error")
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/health")
def health():
    return {
        "status": "ok",
        "supported_languages": ["en", "hi", "ar", "ta", "te"],
        "language_detection": "automatic",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("healthcare_main:app", host="0.0.0.0", port=8001, reload=True)