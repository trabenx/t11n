import os
import math
import argparse
import tempfile
from pydub import AudioSegment, silence
from pydub.utils import make_chunks
import openai

def transcribe_large_audio(audio_file_path, api_key, min_silence_len=500, silence_thresh=-40, buffer_ms=300):
    """
    Transcribe large audio files using Whisper API with quality-preserving segmentation
    """
    openai.api_key = api_key
    audio = AudioSegment.from_file(audio_file_path)
    
    # Calculate max chunk duration (Whisper supports ~25MB files)
    bitrate = 128  # kbps for MP3
    max_duration_ms = min(20 * 60 * 1000,  # 20 minutes maximum
                         (25 * 1024 * 1024 * 8) / (bitrate * 1000) * 1000)
    
    # Process in one piece if small enough
    if len(audio) <= max_duration_ms:
        return transcribe_chunk(audio)
    
    # Find natural silence points for splitting
    silences = silence.detect_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        seek_step=10
    )
    
    # Convert silence to split points with buffer
    split_points = [0]
    for start, end in silences:
        split_point = start + (end - start) // 2  # Middle of silence
        if split_point - split_points[-1] > max_duration_ms:
            split_points.append(split_point)
    split_points.append(len(audio))
    
    # Create chunks respecting max duration and natural silences
    chunks = []
    for i in range(len(split_points)-1):
        start = max(0, split_points[i] - buffer_ms)
        end = min(len(audio), split_points[i+1] + buffer_ms)
        
        # Ensure chunks don't exceed max duration
        if end - start > max_duration_ms:
            subchunks = make_chunks(
                audio[start:end], 
                chunk_size=max_duration_ms
            )
            chunks.extend(subchunks)
        else:
            chunks.append(audio[start:end])
    
    # Transcribe and combine results
    transcriptions = []
    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i+1}/{len(chunks)} ({len(chunk)/1000:.1f}s)")
        transcriptions.append(transcribe_chunk(chunk))
    
    return "\n".join(transcriptions)

def transcribe_chunk(audio_chunk):
    """Transcribe a single audio chunk using Whisper API"""
    temp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    temp_audio.close()  # Close it before passing to export
    # with tempfile.NamedTemporaryFile(suffix=".mp3") as temp_audio:
    # Export with optimal quality settings
    audio_chunk.export(
        temp_audio.name,
        format="mp3",
        bitrate="128k",
        parameters=["-ar", "16000", "-ac", "1"]
    )


    # Transcribe with Whisper
    with open(temp_audio.name, "rb") as audio_file:
        response = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text",
            language="he"
        )
    return response

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files using Whisper API")
    parser.add_argument("input", help="Input audio file path")
    parser.add_argument("output", help="Output text file path")
    parser.add_argument("--api_key", help="OpenAI API key", default=os.getenv("OPENAI_API_KEY"))
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("API key not provided. Set OPENAI_API_KEY environment variable or use --api_key")

    transcript = transcribe_large_audio(
        audio_file_path=args.input,
        api_key=args.api_key
    )

    with open(args.output, "w") as f:
        f.write(transcript)
    print(f"Transcription saved to {args.output}")

if __name__ == "__main__":
    main()