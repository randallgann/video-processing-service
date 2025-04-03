Install Replicate’s Python client library:
pip install replicate

Copy
Set the REPLICATE_API_TOKEN environment variable:
export REPLICATE_API_TOKEN=r8_cqP**********************************

Visibility

Copy
This is your Default API token. Keep it to yourself.

Import the client:
import replicate

Copy
Run openai/whisper using Replicate’s API. Check out the model's schema for an overview of inputs and outputs.

output = replicate.run(
    "openai/whisper:8099696689d249cf8b122d833c36ac3f75505c666a395ca40ef26f68e7d3d16e",
    input={
        "audio": "https://replicate.delivery/mgxm/e5159b1b-508a-4be4-b892-e1eb47850bdc/OSR_uk_000_0050_8k.wav",
        "language": "auto",
        "translate": False,
        "temperature": 0,
        "transcription": "plain text",
        "suppress_tokens": "-1",
        "logprob_threshold": -1,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": True,
        "compression_ratio_threshold": 2.4,
        "temperature_increment_on_fallback": 0.2
    }
)
print(output)

# Input Schema
```json
{
  "type": "object",
  "title": "Input",
  "required": [
    "audio"
  ],
  "properties": {
    "audio": {
      "type": "string",
      "title": "Audio",
      "format": "uri",
      "x-order": 0,
      "description": "Audio file"
    },
    "language": {
      "enum": [
        "auto",
        "af",
        "am",
        "ar",
        "as",
        "az",
        "ba",
        "be",
        "bg",
        "bn",
        "bo",
        "br",
        "bs",
        "ca",
        "cs",
        "cy",
        "da",
        "de",
        "el",
        "en",
        "es",
        "et",
        "eu",
        "fa",
        "fi",
        "fo",
        "fr",
        "gl",
        "gu",
        "ha",
        "haw",
        "he",
        "hi",
        "hr",
        "ht",
        "hu",
        "hy",
        "id",
        "is",
        "it",
        "ja",
        "jw",
        "ka",
        "kk",
        "km",
        "kn",
        "ko",
        "la",
        "lb",
        "ln",
        "lo",
        "lt",
        "lv",
        "mg",
        "mi",
        "mk",
        "ml",
        "mn",
        "mr",
        "ms",
        "mt",
        "my",
        "ne",
        "nl",
        "nn",
        "no",
        "oc",
        "pa",
        "pl",
        "ps",
        "pt",
        "ro",
        "ru",
        "sa",
        "sd",
        "si",
        "sk",
        "sl",
        "sn",
        "so",
        "sq",
        "sr",
        "su",
        "sv",
        "sw",
        "ta",
        "te",
        "tg",
        "th",
        "tk",
        "tl",
        "tr",
        "tt",
        "uk",
        "ur",
        "uz",
        "vi",
        "yi",
        "yo",
        "yue",
        "zh",
        "Afrikaans",
        "Albanian",
        "Amharic",
        "Arabic",
        "Armenian",
        "Assamese",
        "Azerbaijani",
        "Bashkir",
        "Basque",
        "Belarusian",
        "Bengali",
        "Bosnian",
        "Breton",
        "Bulgarian",
        "Burmese",
        "Cantonese",
        "Castilian",
        "Catalan",
        "Chinese",
        "Croatian",
        "Czech",
        "Danish",
        "Dutch",
        "English",
        "Estonian",
        "Faroese",
        "Finnish",
        "Flemish",
        "French",
        "Galician",
        "Georgian",
        "German",
        "Greek",
        "Gujarati",
        "Haitian",
        "Haitian Creole",
        "Hausa",
        "Hawaiian",
        "Hebrew",
        "Hindi",
        "Hungarian",
        "Icelandic",
        "Indonesian",
        "Italian",
        "Japanese",
        "Javanese",
        "Kannada",
        "Kazakh",
        "Khmer",
        "Korean",
        "Lao",
        "Latin",
        "Latvian",
        "Letzeburgesch",
        "Lingala",
        "Lithuanian",
        "Luxembourgish",
        "Macedonian",
        "Malagasy",
        "Malay",
        "Malayalam",
        "Maltese",
        "Mandarin",
        "Maori",
        "Marathi",
        "Moldavian",
        "Moldovan",
        "Mongolian",
        "Myanmar",
        "Nepali",
        "Norwegian",
        "Nynorsk",
        "Occitan",
        "Panjabi",
        "Pashto",
        "Persian",
        "Polish",
        "Portuguese",
        "Punjabi",
        "Pushto",
        "Romanian",
        "Russian",
        "Sanskrit",
        "Serbian",
        "Shona",
        "Sindhi",
        "Sinhala",
        "Sinhalese",
        "Slovak",
        "Slovenian",
        "Somali",
        "Spanish",
        "Sundanese",
        "Swahili",
        "Swedish",
        "Tagalog",
        "Tajik",
        "Tamil",
        "Tatar",
        "Telugu",
        "Thai",
        "Tibetan",
        "Turkish",
        "Turkmen",
        "Ukrainian",
        "Urdu",
        "Uzbek",
        "Valencian",
        "Vietnamese",
        "Welsh",
        "Yiddish",
        "Yoruba"
      ],
      "type": "string",
      "title": "language",
      "description": "Language spoken in the audio, specify 'auto' for automatic language detection",
      "default": "auto",
      "x-order": 3
    },
    "patience": {
      "type": "number",
      "title": "Patience",
      "x-order": 5,
      "description": "optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search"
    },
    "translate": {
      "type": "boolean",
      "title": "Translate",
      "default": false,
      "x-order": 2,
      "description": "Translate the text to English when set to True"
    },
    "temperature": {
      "type": "number",
      "title": "Temperature",
      "default": 0,
      "x-order": 4,
      "description": "temperature to use for sampling"
    },
    "transcription": {
      "enum": [
        "plain text",
        "srt",
        "vtt"
      ],
      "type": "string",
      "title": "transcription",
      "description": "Choose the format for the transcription",
      "default": "plain text",
      "x-order": 1
    },
    "initial_prompt": {
      "type": "string",
      "title": "Initial Prompt",
      "x-order": 7,
      "description": "optional text to provide as a prompt for the first window."
    },
    "suppress_tokens": {
      "type": "string",
      "title": "Suppress Tokens",
      "default": "-1",
      "x-order": 6,
      "description": "comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations"
    },
    "logprob_threshold": {
      "type": "number",
      "title": "Logprob Threshold",
      "default": -1,
      "x-order": 11,
      "description": "if the average log probability is lower than this value, treat the decoding as failed"
    },
    "no_speech_threshold": {
      "type": "number",
      "title": "No Speech Threshold",
      "default": 0.6,
      "x-order": 12,
      "description": "if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence"
    },
    "condition_on_previous_text": {
      "type": "boolean",
      "title": "Condition On Previous Text",
      "default": true,
      "x-order": 8,
      "description": "if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop"
    },
    "compression_ratio_threshold": {
      "type": "number",
      "title": "Compression Ratio Threshold",
      "default": 2.4,
      "x-order": 10,
      "description": "if the gzip compression ratio is higher than this value, treat the decoding as failed"
    },
    "temperature_increment_on_fallback": {
      "type": "number",
      "title": "Temperature Increment On Fallback",
      "default": 0.2,
      "x-order": 9,
      "description": "temperature to increase when falling back when the decoding fails to meet either of the thresholds below"
    }
  }
}
```

# Output Schema
```json
{
  "type": "object",
  "title": "Output",
  "required": [
    "detected_language",
    "transcription"
  ],
  "properties": {
    "segments": {
      "title": "Segments"
    },
    "srt_file": {
      "type": "string",
      "title": "Srt File",
      "format": "uri"
    },
    "txt_file": {
      "type": "string",
      "title": "Txt File",
      "format": "uri"
    },
    "translation": {
      "type": "string",
      "title": "Translation"
    },
    "transcription": {
      "type": "string",
      "title": "Transcription"
    },
    "detected_language": {
      "type": "string",
      "title": "Detected Language"
    }
  }
}
```