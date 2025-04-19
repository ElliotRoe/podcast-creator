# Podcast Audio Combiner, Transcriber, and Highlight Reel Creator (Azure)

A set of Python tools for combining podcast audio files, transcribing them using Azure Speech-to-Text, and creating an AI-generated highlight reel.

## Features

- Combine multiple M4A audio files into segments of maximum 8 hours each (using `pydub` and `ffmpeg`).
- Transcribe large audio files using Azure's batch transcription endpoint.
- Supports speaker diarization (identifying different speakers).
- Generates formatted text transcripts (`.txt`) and optional raw JSON output from Azure.
- **NEW:** Analyzes transcripts using Azure OpenAI to identify highlights (funny, sweet, nice, memorable moments).
- **NEW:** Extracts timestamps for highlights from the transcript.
- **NEW:** Creates a combined audio highlight reel (`.mp3` or other format) from the original audio.

## Requirements

- Python 3.7+
- FFmpeg (for audio processing)
- Azure Account with:
  - A Speech Service resource (Key and Region needed).
  - An Azure OpenAI resource (Key, Endpoint, and Deployment Name needed).

## Installation

1. Clone this repository or download the scripts:

   - `audio_combiner.py` - For combining audio files.
   - `audio_analyzer.py` - For transcribing audio with Azure Speech API.
   - `analyze_podcast.py` - Wrapper script to combine and transcribe in one step.
   - `highlight_creator.py` - **NEW:** Script to generate highlight reels.

2. Install required Python dependencies:

```bash
# pydub for combining/slicing, requests for Azure Speech, openai for Azure OpenAI
pip install pydub requests openai
```

3. Install FFmpeg:

   - macOS: `brew install ffmpeg`
   - Ubuntu/Debian: `sudo apt install ffmpeg`
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

4. Set up Azure Credentials:
   You need credentials for both Speech Service and Azure OpenAI.

   Set them as environment variables (recommended) in a `.env` file (copy from `.env.example`):

```dotenv
# .env file
AZURE_SPEECH_KEY="YOUR_SPEECH_KEY"
AZURE_SPEECH_REGION="YOUR_SPEECH_REGION"

AZURE_OPENAI_KEY="YOUR_AZURE_OPENAI_API_KEY"
AZURE_OPENAI_ENDPOINT="YOUR_AZURE_OPENAI_ENDPOINT"
AZURE_OPENAI_DEPLOYMENT_NAME="YOUR_GPT_DEPLOYMENT_NAME"
```

Alternatively, you can pass credentials directly via command-line arguments.

## Usage Workflow

**Step 1: Combine and Transcribe**

Use `analyze_podcast.py` to process your raw M4A files. This creates combined audio file(s) (e.g., `combined_audio.wav`) and corresponding transcript file(s) (e.g., `combined_audio_transcript.txt`) in the output directory.

```bash
# Make sure AZURE_SPEECH_KEY and AZURE_SPEECH_REGION are set
python analyze_podcast.py --input-dir /path/to/m4a/files/ --output-dir /path/to/processed_output/

# Example output files in /path/to/processed_output/:
# - combined_audio.wav
# - combined_audio_transcript.txt
# - combined_audio_azure_raw.json (if analyzer produced it)
# - (Possibly combined_audio_2.wav, combined_audio_2_transcript.txt, etc. if > 8 hours)
```

**Step 2: Create Highlight Reel**

Use `highlight_creator.py` with the outputs from Step 1.

```bash
# Make sure AZURE_OPENAI_* variables are set in .env
python highlight_creator.py \
    --transcript-file /path/to/processed_output/combined_audio_transcript.txt \
    --audio-file /path/to/processed_output/combined_audio.wav \
    --output-reel /path/to/processed_output/college_highlights.mp3

# Or provide OpenAI credentials via arguments
python highlight_creator.py \
    --transcript-file /path/to/processed_output/combined_audio_transcript.txt \
    --audio-file /path/to/processed_output/combined_audio.wav \
    --output-reel /path/to/processed_output/college_highlights.mp3 \
    --openai-key YOUR_OAI_KEY \
    --openai-endpoint YOUR_OAI_ENDPOINT \
    --openai-deployment YOUR_OAI_DEPLOYMENT
```

**(Repeat Step 2 for each combined audio/transcript pair if your source audio was longer than 8 hours).**

## Individual Script Usage

### `audio_combiner.py`

Combines M4A files interactively.

```bash
python audio_combiner.py
```

### `audio_analyzer.py`

Transcribes a single audio file using Azure.
(See previous README sections for details)

### `analyze_podcast.py`

Combines M4A files and transcribes them using Azure.
(See previous README sections for details)

### `highlight_creator.py` (NEW)

Creates a highlight reel from a single transcript/audio pair.

**Required Arguments:**

- `--transcript-file`: Path to the input `.txt` transcript (from `audio_analyzer.py`).
- `--audio-file`: Path to the corresponding input audio file (e.g., `.wav`, `.mp3`).
- `--output-reel`: Path to save the output highlight reel audio file.

**Credential Arguments (use environment variables or provide these):**

- `--openai-key`: Your Azure OpenAI API Key.
- `--openai-endpoint`: Your Azure OpenAI Endpoint.
- `--openai-deployment`: Your Azure OpenAI Model Deployment Name.

**Optional Arguments:**

- `--prompt`: Custom prompt for the AI to identify highlights.
- `--padding-ms`: Milliseconds of silence between highlight clips (default: 500).

## Notes

- Ensure your Azure OpenAI model deployment (e.g., GPT-4, GPT-3.5-turbo) is suitable for analyzing text and following instructions accurately.
- The quality of highlights depends heavily on the prompt provided to the AI and the clarity of the transcript.
- Timestamp extraction relies on the specific format `[start.ss - end.ss]` present in the transcript lines or preceding speaker lines. Ensure `audio_analyzer.py` generates this format.

## Limitations

- Highlight text matching might fail if the AI doesn't return verbatim text or if the transcript structure is unexpected.
- Audio slicing accuracy depends on `pydub` and `ffmpeg`.
- Relies on the availability and potential costs of both Azure Speech Service and Azure OpenAI.

## References

- [Azure Speech-to-Text Batch Transcription Documentation](https://docs.microsoft.com/en-us/azure/cognitive-services/speech-service/batch-transcription)
- [Azure OpenAI Documentation](https://docs.microsoft.com/en-us/azure/cognitive-services/openai/)
