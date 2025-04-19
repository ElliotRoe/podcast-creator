#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Transcribes all audio files in a specified folder using OpenAI's Whisper model.
Saves transcriptions as .txt files and optionally raw JSON responses.
"""

import os
import sys
import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Supported audio extensions
SUPPORTED_AUDIO_EXTENSIONS = ['.wav', '.mp3', '.ogg', '.flac', '.m4a', '.webm']

def transcribe_audio_whisper(
    audio_file_path: str,
    model_name: str = "small",
    language: Optional[str] = None,
    task: str = "transcribe",
    verbose: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Transcribes audio using OpenAI's Whisper model locally.
    
    Args:
        audio_file_path: Path to the audio file.
        model_name: Whisper model to use (tiny, base, small, medium, large, turbo).
        language: Language code (optional, will auto-detect if not provided).
        task: Task to perform (transcribe or translate).
        verbose: Whether to show verbose output.
        
    Returns:
        Dictionary containing the transcription result.
    """
    try:
        import whisper
    except ImportError:
        logger.error("OpenAI Whisper package not found. Please install with: pip install openai-whisper")
        sys.exit(1)
    
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
    
    start_time = time.time()
    logger.info(f"Loading Whisper model: {model_name}")
    
    # Load the model
    model = whisper.load_model(model_name)
    
    logger.info(f"Transcribing: {os.path.basename(audio_file_path)}")
    
    # Set up transcription options
    options = {}
    if language:
        options["language"] = language
    
    # Perform transcription
    result = model.transcribe(
        audio_file_path, 
        task=task,
        verbose=verbose,
        **options
    )
    
    elapsed = time.time() - start_time
    logger.info(f"Transcription completed in {elapsed:.2f} seconds")
    
    return result

def process_directory(
    input_dir: str,
    output_dir: str,
    model_name: str = "small",
    language: Optional[str] = None,
    task: str = "transcribe",
    skip_existing: bool = False,
    save_json: bool = False,
    extensions: Optional[List[str]] = None,
    verbose: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Process all audio files in a directory using Whisper.
    
    Args:
        input_dir: Directory containing audio files.
        output_dir: Directory to save transcriptions.
        model_name: Whisper model to use.
        language: Language code (optional).
        task: Task to perform (transcribe or translate).
        skip_existing: Skip files that already have transcriptions.
        save_json: Save raw JSON results.
        extensions: List of file extensions to process.
        verbose: Whether to show verbose output.
        
    Returns:
        Dictionary with results and statistics.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Use default extensions if none provided
    if extensions is None:
        extensions = SUPPORTED_AUDIO_EXTENSIONS
    
    # Find all audio files
    audio_files = []
    for ext in extensions:
        files = list(input_path.glob(f"*{ext}"))
        files.extend(list(input_path.glob(f"*{ext.upper()}")))
        audio_files.extend(files)
    
    logger.info(f"Found {len(audio_files)} audio files to process")
    
    if not audio_files:
        logger.warning(f"No audio files found with extensions: {extensions}")
        return {"files": [], "stats": {"processed": 0, "skipped": 0, "failed": 0}}
    
    # Process each file
    results = []
    stats = {"processed": 0, "skipped": 0, "failed": 0}
    
    for i, audio_path in enumerate(sorted(audio_files)):
        base_name = audio_path.stem
        txt_path = output_path / f"{base_name}.txt"
        json_path = output_path / f"{base_name}.json" if save_json else None
        
        logger.info(f"[{i+1}/{len(audio_files)}] Processing: {audio_path.name}")
        
        # Skip if output exists and flag is set
        if skip_existing and txt_path.exists():
            logger.info(f"Skipping {audio_path.name} (output already exists)")
            stats["skipped"] += 1
            continue
        
        try:
            # Transcribe with Whisper
            result = transcribe_audio_whisper(
                str(audio_path),
                model_name=model_name,
                language=language,
                task=task,
                verbose=verbose,
                **kwargs
            )
            
            # Save as text file
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(result["text"])
            
            logger.info(f"Saved transcript to {txt_path}")
            
            # Save raw result as JSON if requested
            if save_json and json_path:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved JSON to {json_path}")
            
            # Add to results
            results.append({
                "file": str(audio_path),
                "output": str(txt_path),
                "success": True
            })
            
            stats["processed"] += 1
            
        except Exception as e:
            logger.error(f"Error processing {audio_path.name}: {str(e)}")
            results.append({
                "file": str(audio_path),
                "error": str(e),
                "success": False
            })
            stats["failed"] += 1
    
    return {
        "files": results,
        "stats": stats
    }

def create_pretty_transcript_from_segments(segments: List[Dict[str, Any]]) -> str:
    """
    Create a human-readable transcript from Whisper segments.
    Includes speaker information if available.
    
    Args:
        segments: List of segment dictionaries from Whisper.
        
    Returns:
        Formatted transcript string.
    """
    lines = []
    current_speaker = None
    
    for segment in segments:
        # Check if there's a speaker field (from diarization)
        speaker = segment.get("speaker", None)
        text = segment.get("text", "").strip()
        
        if not text:
            continue
            
        # Add speaker marker if speaker changes
        if speaker is not None and speaker != current_speaker:
            if lines:  # Add blank line for readability
                lines.append("")
            lines.append(f"--- Speaker {speaker} ---")
            current_speaker = speaker
            
        # Add the text
        lines.append(text)
    
    return "\n".join(lines)

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using OpenAI Whisper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input-folder", 
        required=True,
        help="Path to folder containing audio files"
    )
    
    parser.add_argument(
        "--output-folder", 
        required=True,
        help="Path to folder for saving transcripts"
    )
    
    parser.add_argument(
        "--model", 
        default="small",
        choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large", "large-v1", "large-v2", "large-v3", "turbo"],
        help="Whisper model to use"
    )
    
    parser.add_argument(
        "--language", 
        default=None,
        help="Language code (e.g., 'en', 'es', 'fr', etc.). If not provided, will auto-detect"
    )
    
    parser.add_argument(
        "--task", 
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Whether to transcribe or translate to English"
    )
    
    parser.add_argument(
        "--save-json", 
        action="store_true",
        help="Save raw Whisper JSON output alongside transcript"
    )
    
    parser.add_argument(
        "--skip-existing", 
        action="store_true",
        help="Skip files that already have transcripts"
    )
    
    parser.add_argument(
        "--file-type", 
        default="all",
        choices=["wav", "mp3", "m4a", "all"],
        help="Audio file type to process"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Show verbose output"
    )
    
    args = parser.parse_args()
    
    # Set up file extensions based on the file-type argument
    if args.file_type == "all":
        extensions = SUPPORTED_AUDIO_EXTENSIONS
    else:
        extensions = [f".{args.file_type}"]
    
    # Process the directory
    start_time = time.time()
    results = process_directory(
        input_dir=args.input_folder,
        output_dir=args.output_folder,
        model_name=args.model,
        language=args.language,
        task=args.task,
        skip_existing=args.skip_existing,
        save_json=args.save_json,
        extensions=extensions,
        verbose=args.verbose
    )
    
    # Print summary
    stats = results["stats"]
    total_time = time.time() - start_time
    
    logger.info("\n--- Transcription Summary ---")
    logger.info(f"Successfully transcribed: {stats['processed']}")
    logger.info(f"Failed:                  {stats['failed']}")
    logger.info(f"Skipped (already exist): {stats['skipped']}")
    logger.info(f"Total files:             {stats['processed'] + stats['failed'] + stats['skipped']}")
    logger.info(f"Total time:              {total_time:.2f} seconds")
    logger.info("--------------------------")
    
    if stats["failed"] > 0:
        sys.exit(1)  # Exit with error code if any transcriptions failed
    
if __name__ == "__main__":
    main() 