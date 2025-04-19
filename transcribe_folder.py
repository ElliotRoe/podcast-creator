#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Transcribes all audio files in a specified folder (or Azure container) using the 
Azure Speech-to-Text batch API.

Saves transcriptions as .txt files and optionally raw JSON responses.
"""

import os
import sys
import argparse
import json
import time
import requests
import logging
import uuid
from typing import Dict, Any, List, Optional
from pathlib import Path
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Supported audio extensions
SUPPORTED_AUDIO_EXTENSIONS = ['.wav', '.mp3', '.ogg', '.flac']

def transcribe_container(
    speech_key: str,
    service_region: str,
    container_sas_url: str,
    locale: str = "en-US",
    output_folder: str = None,
    cleanup: bool = False  # Make cleanup optional and disabled by default
) -> Dict[str, Any]:
    """
    Transcribes all audio files in an Azure Blob Storage container using the batch API.
    
    Args:
        speech_key: Azure Speech Service subscription key.
        service_region: Azure Speech Service region (e.g., 'eastus').
        container_sas_url: SAS URL with read access to the container with audio files.
        locale: Language locale for transcription.
        output_folder: Local folder to save transcription results.
        cleanup: Whether to delete the transcription job after processing (default: False).
        
    Returns:
        Dictionary mapping filenames to their transcription results.
    """
    logger.info(f"Starting batch transcription for container: {container_sas_url}")
    
    # 1. Create a batch transcription for the entire container
    transcription_id = create_container_transcription(
        speech_key=speech_key,
        service_region=service_region,
        container_url=container_sas_url,
        locale=locale,
        display_name=f"ContainerTranscription_{uuid.uuid4()}"
    )
    
    if not transcription_id:
        raise Exception("Failed to create batch transcription for container")
    
    logger.info(f"Created batch transcription with ID: {transcription_id}")
    
    # 2. Poll until the transcription is complete
    transcription_result = poll_transcription_status(
        speech_key=speech_key,
        service_region=service_region,
        transcription_id=transcription_id
    )
    
    # 3. Get the transcription files (results)
    transcription_files = get_transcription_files(
        speech_key=speech_key,
        service_region=service_region,
        transcription_id=transcription_id
    )
    
    # 4. Process and save all results
    results = {}
    
    for file_data in transcription_files:
        if file_data.get("kind") == "Transcription":
            # Get the original audio filename
            audio_filename = file_data.get("name", "unknown.wav")
            result_url = file_data.get("links", {}).get("contentUrl")
            
            if result_url:
                logger.info(f"Processing result for {audio_filename}")
                result_response = requests.get(result_url)
                if result_response.status_code == 200:
                    result_data = result_response.json()
                    
                    # Save results
                    if output_folder:
                        base_name = Path(audio_filename).stem
                        txt_path = os.path.join(output_folder, f"{base_name}.txt")
                        json_path = os.path.join(output_folder, f"{base_name}.json")
                        
                        # Create and save transcript
                        transcript = create_pretty_transcript(result_data)
                        save_text_file(transcript, txt_path)
                        
                        # Save raw JSON
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(result_data, f, indent=2)
                    
                    # Add to results dictionary
                    results[audio_filename] = result_data
    
    # 5. Clean up the transcription only if requested
    if cleanup:
        logger.info(f"Cleaning up transcription job {transcription_id}")
        delete_transcription(
            speech_key=speech_key,
            service_region=service_region,
            transcription_id=transcription_id
        )
    else:
        logger.info(f"Keeping transcription job {transcription_id} for future reference")
    
    return results

def create_container_transcription(
    speech_key: str,
    service_region: str,
    container_url: str,
    locale: str = "en-US",
    display_name: str = "Container Transcription"
) -> str:
    """
    Creates a batch transcription job for all files in a container.
    
    Args:
        speech_key: Azure Speech Service subscription key.
        service_region: Azure Speech Service region.
        container_url: SAS URL with read access to container with audio files.
        locale: Language locale (e.g., "en-US").
        display_name: Display name for the transcription.
        
    Returns:
        The transcription ID if successful, otherwise None.
    """
    endpoint = f"https://{service_region}.api.cognitive.microsoft.com/speechtotext/v3.2/transcriptions"
    
    headers = {
        "Ocp-Apim-Subscription-Key": speech_key,
        "Content-Type": "application/json"
    }
    
    body = {
        "contentContainerUrl": container_url,
        "locale": locale,
        "displayName": display_name,
        "properties": {
            "wordLevelTimestampsEnabled": True,
            "diarizationEnabled": True,
            "punctuationMode": "DictatedAndAutomatic",
            "profanityFilterMode": "Masked"
        }
    }
    
    try:
        response = requests.post(endpoint, headers=headers, json=body)
        
        if response.status_code == 201 or response.status_code == 200:
            # Extract the transcription ID from the self link
            transcription_data = response.json()
            transcription_self_url = transcription_data.get("self", "")
            transcription_id = transcription_self_url.split("/")[-1]
            return transcription_id
        else:
            logger.error(f"Failed to create transcription. Status code: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return None
    
    except Exception as e:
        logger.error(f"Error creating batch transcription: {e}")
        return None

def poll_transcription_status(
    speech_key: str,
    service_region: str,
    transcription_id: str,
    polling_interval: int = 5,
    timeout: int = 1800  # 30 minutes
) -> Dict[Any, Any]:
    """
    Polls the transcription status until it succeeds or fails.
    
    Args:
        speech_key: Azure Speech Service subscription key.
        service_region: Azure Speech Service region.
        transcription_id: The ID of the transcription to poll.
        polling_interval: Time in seconds between polling requests.
        timeout: Maximum time to wait for completion in seconds.
        
    Returns:
        The final transcription status data.
        
    Raises:
        Exception: If the transcription fails or times out.
    """
    endpoint = f"https://{service_region}.api.cognitive.microsoft.com/speechtotext/v3.2/transcriptions/{transcription_id}"
    
    headers = {
        "Ocp-Apim-Subscription-Key": speech_key
    }
    
    start_time = time.time()
    completed = False
    
    while not completed:
        # Check if we've exceeded the timeout
        if time.time() - start_time > timeout:
            raise Exception(f"Transcription timed out after {timeout} seconds")
        
        try:
            response = requests.get(endpoint, headers=headers)
            
            if response.status_code == 200:
                transcription_data = response.json()
                status = transcription_data.get("status", "")
                
                logger.info(f"Transcription status: {status}")
                
                if status == "Succeeded":
                    completed = True
                    return transcription_data
                elif status == "Failed":
                    error_message = transcription_data.get("properties", {}).get("error", {}).get("message", "Unknown error")
                    raise Exception(f"Transcription failed: {error_message}")
                else:
                    # Still running, wait and try again
                    time.sleep(polling_interval)
            else:
                logger.error(f"Failed to get transcription status. Code: {response.status_code}, Response: {response.text}")
                time.sleep(polling_interval)
        
        except Exception as e:
            if "Transcription failed" in str(e):
                raise
            logger.error(f"Error checking transcription status: {e}")
            time.sleep(polling_interval)

def get_transcription_files(
    speech_key: str,
    service_region: str,
    transcription_id: str
) -> List[Dict[Any, Any]]:
    """
    Gets the list of files (results) for a completed transcription.
    
    Args:
        speech_key: Azure Speech Service subscription key.
        service_region: Azure Speech Service region.
        transcription_id: The ID of the completed transcription.
        
    Returns:
        List of file data objects.
    """
    endpoint = f"https://{service_region}.api.cognitive.microsoft.com/speechtotext/v3.2/transcriptions/{transcription_id}/files"
    
    headers = {
        "Ocp-Apim-Subscription-Key": speech_key
    }
    
    try:
        response = requests.get(endpoint, headers=headers)
        
        if response.status_code == 200:
            files_data = response.json()
            return files_data.get("values", [])
        else:
            logger.error(f"Failed to get transcription files. Code: {response.status_code}, Response: {response.text}")
            return []
            
    except Exception as e:
        logger.error(f"Error getting transcription files: {e}")
        return []

def delete_transcription(
    speech_key: str,
    service_region: str,
    transcription_id: str
) -> bool:
    """
    Deletes a transcription from the service.
    
    Args:
        speech_key: Azure Speech Service subscription key.
        service_region: Azure Speech Service region.
        transcription_id: The ID of the transcription to delete.
        
    Returns:
        True if deletion was successful, False otherwise.
    """
    endpoint = f"https://{service_region}.api.cognitive.microsoft.com/speechtotext/v3.2/transcriptions/{transcription_id}"
    
    headers = {
        "Ocp-Apim-Subscription-Key": speech_key
    }
    
    try:
        response = requests.delete(endpoint, headers=headers)
        
        if response.status_code == 204 or response.status_code == 200:
            logger.info(f"Transcription {transcription_id} deleted successfully")
            return True
        else:
            logger.warning(f"Failed to delete transcription. Code: {response.status_code}, Response: {response.text}")
            return False
            
    except Exception as e:
        logger.warning(f"Error deleting transcription: {e}")
        return False

def create_pretty_transcript(transcript_data: Dict[Any, Any]) -> str:
    """Create a simple, human-readable transcript focusing on speakers and their text."""
    pretty_transcript = []
    
    # Check if we have recognizedPhrases in the format from the batch API
    combined_results = transcript_data.get("combinedRecognizedPhrases", [])
    if combined_results:
        # Simple approach for batch API results - just use the combined text
        for result in combined_results:
            pretty_transcript.append(result.get("display", ""))
        return "\n\n".join(pretty_transcript)
    
    # For results with recognized phrases (more detailed output)
    recognized_phrases = transcript_data.get("recognizedPhrases", [])
    if recognized_phrases:
        current_speaker = None
        
        # Sort phrases by offset time if possible
        if recognized_phrases and "offsetInTicks" in recognized_phrases[0]:
            recognized_phrases = sorted(recognized_phrases, key=lambda x: int(x.get("offsetInTicks", 0)))
        
        for phrase in recognized_phrases:
            # Get speaker information if available
            speaker_id = phrase.get("speaker", None)
            
            # Get the display text
            display_text = phrase.get("nBest", [{}])[0].get("display", "").strip()
            if not display_text:
                continue
                
            # Add speaker information if it changed
            if speaker_id is not None and speaker_id != current_speaker:
                if pretty_transcript:  # Add a blank line for readability
                    pretty_transcript.append("")
                pretty_transcript.append(f"--- Speaker {speaker_id} ---")
                current_speaker = speaker_id
            
            # Add the text
            pretty_transcript.append(display_text)
    
    # Fallback for simpler formats
    phrases = transcript_data.get('phrases', [])
    if phrases:
        # This is similar to the existing function
        if phrases and not isinstance(phrases[0].get('offset'), (int, float)):
            sorted_phrases = phrases 
        else:
            sorted_phrases = sorted(phrases, key=lambda x: x.get('offset', 0))
        
        current_speaker_label = None
        for phrase in sorted_phrases:
            speaker_id = phrase.get('speaker')
            channel = phrase.get('channel')
            text = phrase.get('text', '')
            
            if speaker_id is not None:
                speaker_label = f"Speaker {speaker_id}"
            elif channel is not None:
                speaker_label = f"Channel {channel}"
            else:
                speaker_label = "Speaker"

            if speaker_label != current_speaker_label:
                if pretty_transcript:
                    pretty_transcript.append("")
                pretty_transcript.append(f"--- {speaker_label} ---")
                current_speaker_label = speaker_label
                
            pretty_transcript.append(text.strip())
    
    if not pretty_transcript:
        return "No transcription data found in the expected format."
        
    return "\n".join(pretty_transcript)

def save_text_file(text_content: str, output_path: str) -> None:
    """Saves text content to a file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text_content)
        logger.info(f"File saved successfully to: {output_path}")
    except Exception as e:
        logger.error(f"Error saving file to {output_path}: {e}")
        
# --- Main Execution --- 

def main():
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(description="Transcribe audio files using Azure Speech-to-Text batch API.")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input-folder', help='Path to local folder containing audio files (for local processing)')
    group.add_argument('--container-sas-url', help='SAS URL for Azure Blob Storage container with audio files')
    
    parser.add_argument('--output-folder', required=True, help='Path to save transcripts and JSON files')
    parser.add_argument('--speech-key', required=False, help='Azure Speech Service Subscription Key (or set AZURE_SPEECH_KEY env var)')
    parser.add_argument('--service-region', required=False, default="eastus", help='Azure Speech Service Region (e.g., eastus). Default: eastus')
    parser.add_argument('--locale', default="en-US", help='Locale for transcription. Default: en-US')
    parser.add_argument('--save-json', action='store_true', help='Save the raw Azure JSON response alongside the transcript')
    parser.add_argument('--skip-existing', action='store_true', help='Skip transcription if the output .txt file already exists')
    parser.add_argument('--cleanup', action='store_true', help='Delete the transcription job after processing (default: False)')

    args = parser.parse_args()

    # Get credentials from args or environment variables
    speech_key = args.speech_key or os.getenv("AZURE_SPEECH_KEY") or os.getenv("AZURE_AI_KEY")
    service_region = args.service_region or os.getenv("AZURE_SPEECH_REGION", "eastus")
    
    if not speech_key:
        logger.error("Azure Speech Key not found. Provide via --speech-key or set AZURE_SPEECH_KEY/AZURE_AI_KEY env var.")
        sys.exit(1)
    
    logger.info(f"Using Azure Speech service in region: {service_region}")

    # Create output folder
    output_folder = Path(args.output_folder)
    try:
        output_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output will be saved to: {output_folder}")
    except OSError as e:
        logger.error(f"Could not create output folder {output_folder}: {e}")
        sys.exit(1)
    
    try:
        # Container-based processing (simpler)
        if args.container_sas_url:
            logger.info("Processing all files in container (batch mode)")
            results = transcribe_container(
                speech_key=speech_key,
                service_region=service_region,
                container_sas_url=args.container_sas_url,
                locale=args.locale,
                output_folder=str(output_folder),
                cleanup=args.cleanup  # Pass the cleanup flag
            )
            
            logger.info(f"Successfully processed {len(results)} files from container")
            
        # Local folder processing
        else:
            input_folder = Path(args.input_folder)
            if not input_folder.is_dir():
                logger.error(f"Input folder not found: {input_folder}")
                sys.exit(1)
                
            logger.warning("Local folder processing requires Azure Storage. Using placeholder implementation.")
            logger.info(f"To process all files at once (recommended), upload them to an Azure Storage container and use --container-sas-url")
            
            # TODO: Implement local folder processing logic using Azure Storage SDK
            # This would involve:
            # 1. Creating a container in Azure Storage
            # 2. Uploading all files to that container
            # 3. Generating a SAS URL for the container
            # 4. Calling the transcribe_container function
            
            logger.error("Local folder processing is not yet implemented")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        sys.exit(1)
        
    logger.info("Transcription process completed successfully")

if __name__ == "__main__":
    main() 