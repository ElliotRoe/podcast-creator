#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyzes transcripts in a folder using Azure OpenAI to identify highlights
and extracts their timestamps.

Input: Folder containing .txt transcript files (from transcribe_folder.py).
Output: Folder containing .json files, each with a list of [start_ms, end_ms]
highlight timestamps corresponding to the transcript.
"""

import os
import sys
import argparse
import re
import logging
import json
from openai import AzureOpenAI
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv

# --- Copy/Adapt from highlight_creator.py --- 

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default prompt for highlight identification
DEFAULT_HIGHLIGHT_PROMPT = ("Review the following conversation transcript which includes speaker labels and timestamps (e.g., --- Speaker 1 [34.56s - 38.12s] --- or [12.34s - 56.78s]). "
                            "Identify segments that are particularly funny, sweet, nice, heartwarming, or capture memorable moments. "
                            "For each identified highlight, return ONLY the exact, verbatim text segment from the transcript AS IT APPEARS, including any speaker/timestamp lines immediately preceding the text. "
                            "Each complete highlight segment (including potential speaker/timestamp line and text line(s)) should be on its own line in your output. "
                            "Do not add any extra explanation, numbering, commentary, or introductory/concluding remarks. "
                            "Focus on moments of genuine connection, humor, or reflection."
                            "Example input lines:\n"
                            "---\n"
                            "--- Speaker 1 [34.56s - 38.12s] ---\n"
                            "Yeah, that class was brutal!\n"
                            "[38.50s - 40.10s] It really was.\n"
                            "---\n"
                            "Example output lines if both were highlights:\n"
                            "---\n"
                            "--- Speaker 1 [34.56s - 38.12s] ---\n"
                            "Yeah, that class was brutal!\n"
                            "[38.50s - 40.10s] It really was.\n"
                            "---"
                           )


# --- OpenAI Interaction (Copied/Adapted from highlight_creator.py) --- 

def authenticate_openai_client(api_key: str, endpoint: str) -> AzureOpenAI:
    """Authenticate with Azure OpenAI service."""
    try:
        client = AzureOpenAI(
            api_key=api_key,
            # Ensure this version aligns with your Azure deployment capabilities
            api_version="2024-02-15-preview", 
            azure_endpoint=endpoint
        )
        # Test connection - optional, but good practice
        # client.models.list() # This might incur a small cost
        logger.info("Azure OpenAI client authenticated successfully.")
        return client
    except Exception as e:
        logger.error(f"Failed to authenticate Azure OpenAI client: {e}")
        raise

def get_highlights_from_ai(client: AzureOpenAI, deployment_name: str, transcript_text: str, prompt: str) -> List[str]:
    """Sends transcript to Azure OpenAI and asks for highlights."""
    logger.info(f"Requesting highlights from Azure OpenAI model: {deployment_name}")
    try:
        messages = [
            {"role": "system", "content": prompt},
            # Provide the full transcript text as user content
            {"role": "user", "content": f"Here is the transcript:\n\n{transcript_text}"} 
        ]
        
        response = client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            temperature=0.3, # Lower temperature for more deterministic extraction
            max_tokens=1500, # Adjust based on transcript length and expected highlight volume
            # Consider adding stop sequences if the model tends to add extra text
            # stop=["--- end highlights ---"] 
        )
        
        # Check for valid response structure
        if not response.choices or not response.choices[0].message or not response.choices[0].message.content:
             logger.warning("Azure OpenAI response structure is invalid or content is missing.")
             return []
             
        raw_highlights = response.choices[0].message.content.strip()
        # Split into lines and remove empty lines
        highlight_lines = [line for line in raw_highlights.split('\n') if line.strip()]
        
        if not highlight_lines:
             logger.warning("Azure OpenAI returned an empty response or only whitespace.")
        else:
             logger.info(f"Received {len(highlight_lines)} potential highlight segment lines from AI.")
             # logger.debug(f"Raw AI response:\n{raw_highlights}")
             # logger.debug(f"Parsed highlight lines:\n{highlight_lines}")
        return highlight_lines
        
    except Exception as e:
        logger.error(f"Error getting highlights from Azure OpenAI: {e}")
        # Log more details if possible (e.g., response status code if it's an APIError)
        if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
            logger.error(f"OpenAI API Response Status Code: {e.response.status_code}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
             logger.error(f"OpenAI API Response Body: {e.response.text[:500]}...") # Log first 500 chars
        return [] # Return empty list on failure

# --- Timestamp Extraction (Copied/Adapted from highlight_creator.py) --- 

def extract_timestamps_from_transcript(transcript_lines: List[str], highlight_ai_output_lines: List[str]) -> List[Tuple[float, float]]:
    """Finds highlight text in the transcript and extracts start/end times.
    
    This version assumes the AI returns verbatim lines, including potential speaker/timestamp lines.
    It looks for these lines in the original transcript and extracts the FIRST timestamp found
    within or immediately preceding the matched block.

    Args:
        transcript_lines: List of lines from the formatted transcript file.
        highlight_ai_output_lines: List of verbatim highlight lines returned by the AI.
                                    These might be single lines or multi-line blocks.

    Returns:
        List of tuples, where each tuple is (start_time_ms, end_time_ms).
        Timestamps are sorted and deduplicated.
    """
    timestamps_ms = []
    
    # Regex to find timestamps like [12.34s - 56.78s] or --- Speaker N [12.34s - 56.78s] ---
    # Capture the start and end times (groups 1 and 2)
    timestamp_pattern = re.compile(r"\[\s*(\d+\.\d+)s\s*-\s*(\d+\.\d+)s\s*\]")
    
    if not highlight_ai_output_lines:
        logger.warning("AI returned no highlight lines to process.")
        return []
        
    # Process AI output into logical blocks. Consecutive lines from AI output form a block.
    # This simplistic approach assumes AI output is reasonably structured.
    ai_blocks = []
    current_block = []
    for line in highlight_ai_output_lines:
        if line.strip(): # Only add non-empty lines
            current_block.append(line)
        # If a line seems like a timestamp/speaker line, consider it the start of a new block?
        # For now, let's keep it simple: treat all consecutive non-empty lines as one block
        # unless separated by empty lines in AI output (which we already filter out). 
        # A better approach might group based on finding timestamps.
    if current_block: # Add the last block
        ai_blocks.append("\n".join(current_block))
        
    # We could also just treat each line from the AI as a potential start of a match
    # Let's refine: treat each non-empty line from AI as something to search for.
    # This avoids issues with how the AI groups lines. 
    search_fragments = [line.strip() for line in highlight_ai_output_lines if line.strip()] 
    if not search_fragments:
        logger.warning("AI output contained no searchable text after stripping whitespace.")
        return []

    logger.info(f"Attempting to match {len(search_fragments)} AI output lines/fragments in the transcript...")
    full_transcript_text = "\n".join(transcript_lines)
    found_indices = set() # Track indices in transcript to avoid duplicate matches from overlapping AI lines

    # Iterate through each fragment provided by the AI
    for fragment in search_fragments:
        start_index = 0
        while True:
            # Find the next occurrence of the fragment in the transcript
            try:
                match_index = full_transcript_text.index(fragment, start_index)
            except ValueError:
                # Fragment not found (or no more occurrences)
                break # Stop searching for this fragment

            # Check if this match index overlaps with an already found timestamp region
            is_already_found = False
            for found_start, found_end in found_indices:
                if max(found_start, match_index) < min(found_end, match_index + len(fragment)):
                    is_already_found = True
                    break
            
            if is_already_found:
                # Move past this match and search again
                start_index = match_index + 1
                continue
                
            # Found a new potential match. Now find the *closest preceding* timestamp.
            # Search backwards from the start of the match index in the transcript text.
            search_area = full_transcript_text[:match_index + len(fragment)] # Include the fragment itself
            best_match_pos = -1
            timestamp_match = None

            # Find the last timestamp before or within the fragment
            for match in timestamp_pattern.finditer(search_area):
                if match.start() > best_match_pos:
                    best_match_pos = match.start()
                    timestamp_match = match
            
            if timestamp_match:
                start_s = float(timestamp_match.group(1))
                end_s = float(timestamp_match.group(2))
                start_ms = start_s * 1000
                end_ms = end_s * 1000
                
                # Check for validity (e.g., end > start)
                if start_ms < end_ms:
                    timestamps_ms.append((start_ms, end_ms))
                    # Record the span of the transcript covered by this timestamp
                    # to prevent double-matching based on overlapping AI fragments
                    # Note: This is approximate. A better way would be to use line numbers.
                    found_indices.add((match_index, match_index + len(fragment)))
                    logger.debug(f"Matched fragment '{fragment[:50]}...' -> Timestamp [{start_s:.2f}s - {end_s:.2f}s]")
                    # Since we found a timestamp for this occurrence, break the inner search loop
                    # for this specific `fragment` instance and move to the next fragment.
                    # If we expect multiple non-overlapping occurrences of the same fragment, remove this break.
                    break # Found timestamp for this match index, move to next fragment
                else:
                    logger.warning(f"Invalid timestamp found (start >= end): [{start_s:.2f}s - {end_s:.2f}s] near fragment '{fragment[:50]}...'")
                    # Move past this match index to avoid re-matching the same invalid timestamp
                    start_index = match_index + 1
            else:
                 logger.warning(f"Could not find timestamp preceding/within match for fragment: '{fragment[:50]}...' at index {match_index}")
                 # Move past this match and search again
                 start_index = match_index + 1
                 
    if not timestamps_ms:
        logger.warning("Could not extract any valid timestamps for the AI-identified highlights.")
        logger.warning("Check AI prompt and output format. Ensure AI returns verbatim text including timestamps/speaker lines.")
        logger.warning("Also check transcript format for consistent timestamp patterns.")
    else:
        # Remove duplicates and sort by start time
        unique_timestamps = sorted(list(set(timestamps_ms)), key=lambda t: t[0])
        if len(unique_timestamps) < len(timestamps_ms):
             logger.info(f"Removed {len(timestamps_ms) - len(unique_timestamps)} duplicate timestamps.")
        timestamps_ms = unique_timestamps
        logger.info(f"Successfully extracted {len(timestamps_ms)} unique timestamps for highlights.")
        
    return timestamps_ms

# --- File Handling --- 

def save_timestamps_json(timestamps: List[Tuple[float, float]], output_path: str):
    """Saves the list of [start_ms, end_ms] timestamps to a JSON file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # Save as a simple JSON list of lists/tuples
            json.dump(timestamps, f, indent=4)
        logger.info(f"Highlight timestamps saved to: {output_path}")
    except Exception as e:
        logger.error(f"Error saving timestamps JSON to {output_path}: {e}")

# --- Main Execution --- 

def main():
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(description="Analyze transcripts using Azure OpenAI to find highlights and extract timestamps.")
    
    # Input/Output Folders
    parser.add_argument("--transcript-folder", required=True, help="Path to the folder containing input transcript files (.txt).")
    parser.add_argument("--output-folder", required=True, help="Path to the folder where highlight timestamp JSON files (.json) will be saved.")
    
    # Azure OpenAI Credentials
    parser.add_argument("--openai-key", required=False, help="Azure OpenAI API Key (or set AZURE_OPENAI_KEY env var).")
    parser.add_argument("--openai-endpoint", required=False, help="Azure OpenAI Endpoint (or set AZURE_OPENAI_ENDPOINT env var).")
    parser.add_argument("--openai-deployment", required=False, help="Azure OpenAI Deployment Name (or set AZURE_OPENAI_DEPLOYMENT_NAME env var).")

    # Optional Arguments
    parser.add_argument("--prompt", default=DEFAULT_HIGHLIGHT_PROMPT, help="Custom prompt for the AI to identify highlights.")
    parser.add_argument('--skip-existing', action='store_true', help='Skip analysis if the output .json file already exists.')

    args = parser.parse_args()

    # --- Get Credentials ---
    openai_key = args.openai_key or os.getenv("AZURE_OPENAI_KEY")
    openai_endpoint = args.openai_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
    openai_deployment = args.openai_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

    if not all([openai_key, openai_endpoint, openai_deployment]):
        logger.error("Missing Azure OpenAI credentials. Provide via arguments or environment variables (AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME).")
        sys.exit(1)
        
    # --- Validate Folders ---
    transcript_folder = Path(args.transcript_folder)
    output_folder = Path(args.output_folder)
    if not transcript_folder.is_dir():
        logger.error(f"Transcript input folder not found: {transcript_folder}")
        sys.exit(1)
        
    try:
        output_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Highlight timestamp JSONs will be saved to: {output_folder}")
    except OSError as e:
        logger.error(f"Could not create output folder {output_folder}: {e}")
        sys.exit(1)

    # --- Find Transcript Files ---
    transcript_files = list(transcript_folder.glob('*.txt'))
    if not transcript_files:
        logger.warning(f"No transcript files (.txt) found in {transcript_folder}")
        sys.exit(0)
        
    logger.info(f"Found {len(transcript_files)} transcript files to analyze.")

    # --- Authenticate OpenAI Client (once) ---
    try:
        openai_client = authenticate_openai_client(openai_key, openai_endpoint)
    except Exception as e:
        logger.error(f"Failed to initialize Azure OpenAI client. Exiting. Error: {e}")
        sys.exit(1)

    # --- Process Each Transcript --- 
    success_count = 0
    failure_count = 0
    skipped_count = 0

    for transcript_path in transcript_files:
        base_name = transcript_path.stem
        output_json_path = output_folder / f"{base_name}_timestamps.json" # Changed suffix

        logger.info(f"\n--- Analyzing: {transcript_path.name} ---")

        # Skip if output exists and flag is set
        if args.skip_existing and output_json_path.exists():
            logger.info(f"Skipping '{transcript_path.name}' because output timestamps file '{output_json_path.name}' already exists.")
            skipped_count += 1
            continue
            
        try:
            # 1. Read Transcript
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript_lines = f.readlines()
            transcript_text = "".join(transcript_lines)
            if not transcript_text.strip():
                logger.warning(f"Transcript file '{transcript_path.name}' is empty. Skipping.")
                failure_count += 1 # Count as failure/skip
                continue
                
            # 2. Get Highlights from AI
            highlight_ai_lines = get_highlights_from_ai(
                openai_client,
                openai_deployment,
                transcript_text,
                args.prompt
            )
            
            if not highlight_ai_lines:
                logger.warning(f"AI did not return any highlight candidates for '{transcript_path.name}'. Skipping timestamp extraction.")
                # Decide if this is a failure or just no highlights found. Let's count it as processed but no highlights.
                # To count as failure, increment failure_count here.
                # If we save an empty list, it's technically success. Let's do that.
                timestamps_ms = [] 
            else:
                # 3. Extract Timestamps
                timestamps_ms = extract_timestamps_from_transcript(transcript_lines, highlight_ai_lines)
                if not timestamps_ms:
                    logger.warning(f"Could not extract any timestamps from the highlights identified by AI for '{transcript_path.name}'. Saving empty list.")
                    # continue # Or save an empty JSON?
            
            # 4. Save Timestamps JSON (even if empty)
            save_timestamps_json(timestamps_ms, str(output_json_path))
            success_count += 1 # Count as success if we reached here, even with 0 timestamps

        except FileNotFoundError as e:
            logger.error(f"Error accessing transcript file {transcript_path.name}: {e}")
            failure_count += 1
        except Exception as e:
            logger.exception(f"An unexpected error occurred while analyzing {transcript_path.name}: {e}") # Log traceback
            failure_count += 1
            # Continue to the next file

    logger.info("\n--- Analysis Summary ---")
    logger.info(f"Successfully analyzed:     {success_count}")
    logger.info(f"Failed analysis:         {failure_count}")
    logger.info(f"Skipped (already exist): {skipped_count}")
    logger.info(f"Total processed:         {success_count + failure_count + skipped_count}")
    logger.info("-------------------------")

    if failure_count > 0:
        sys.exit(1) # Exit with error code if any analysis failed

if __name__ == "__main__":
    main() 