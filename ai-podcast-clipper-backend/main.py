import glob
import json
import pathlib
import pickle
import shutil
import subprocess
import time
import uuid
import boto3
import cv2
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import ffmpegcv
import modal
import numpy as np
from pydantic import BaseModel
import os
from google import genai

import re
import pysubs2
from tqdm import tqdm
import whisperx


class ProcessVideoRequest(BaseModel):
    s3_key: str  # Required - S3 key of uploaded video
    video_type: str = "podcast"  # "podcast", "sermon", "presentation"
    max_clips: int = 3  # User-configurable, validated by frontend against tier limits
    caption_effect: str = "karaoke"  # "none", "pop", "fade", "karaoke"
    caption_font: str = "Anton"
    caption_font_size: int = 140
    caption_color: str = "#FFFFFF"
    highlight_color: str = "#FFD700"  # Gold for karaoke highlight
    regenerate_mode: bool = False
    target_start: float | None = None
    target_end: float | None = None
    exclude_ranges: list[dict] = []  # List of {"start": float, "end": float} to exclude


image = (modal.Image.from_registry(
    "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10")
    .run_commands([
        # Fix stale package lists with clean and update
        "apt-get clean",
        "rm -rf /var/lib/apt/lists/*",
        "apt-get update --fix-missing",
        # Install minimal core dependencies (avoiding problematic ffmpeg meta-package)
        # libgl1-mesa-glx and libglib2.0-0 are required for OpenCV
        "apt-get install -y --no-install-recommends wget git curl unzip fontconfig libgl1-mesa-glx libglib2.0-0",
        # Install ffmpeg from static build to avoid dependency issues
        "wget -q https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -O /tmp/ffmpeg.tar.xz",
        "tar -xf /tmp/ffmpeg.tar.xz -C /tmp",
        "mv /tmp/ffmpeg-*-amd64-static/ffmpeg /usr/local/bin/",
        "mv /tmp/ffmpeg-*-amd64-static/ffprobe /usr/local/bin/",
        "rm -rf /tmp/ffmpeg*",
        # Install fonts
        "mkdir -p /usr/share/fonts/truetype/custom",
        "wget -O /usr/share/fonts/truetype/custom/Anton-Regular.ttf https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf",
        "fc-cache -f -v"
    ])
    .pip_install_from_requirements("requirements.txt")
    .pip_install("pysubs2")
    .add_local_dir("asd", "/asd", copy=True, ignore=[".git", "**/.git"]))

app = modal.App("ai-podcast-clipper", image=image)

volume = modal.Volume.from_name(
    "ai-podcast-clipper-model-cache", create_if_missing=True
)

mount_path = "/root/.cache/torch"

auth_scheme = HTTPBearer()


def create_vertical_video(tracks, scores, pyframes_path, pyavi_path, audio_path, output_path, framerate=25):
    target_width = 1080
    target_height = 1920

    flist = glob.glob(os.path.join(pyframes_path, "*.jpg"))
    flist.sort()

    faces = [[] for _ in range(len(flist))]

    for tidx, track in enumerate(tracks):
        score_array = scores[tidx]
        for fidx, frame in enumerate(track["track"]["frame"].tolist()):
            slice_start = max(fidx - 30, 0)
            slice_end = min(fidx + 30, len(score_array))
            score_slice = score_array[slice_start:slice_end]
            avg_score = float(np.mean(score_slice)
                              if len(score_slice) > 0 else 0)

            faces[frame].append(
                {'track': tidx, 'score': avg_score, 's': track['proc_track']["s"][fidx], 'x': track['proc_track']["x"][fidx], 'y': track['proc_track']["y"][fidx]})

    temp_video_path = os.path.join(pyavi_path, "video_only.mp4")

    vout = None
    for fidx, fname in tqdm(enumerate(flist), total=len(flist), desc="Creating vertical video"):
        img = cv2.imread(fname)
        if img is None:
            continue

        current_faces = faces[fidx]

        max_score_face = max(
            current_faces, key=lambda face: face['score']) if current_faces else None

        if max_score_face and max_score_face['score'] < 0:
            max_score_face = None

        if vout is None:
            # Use regular VideoWriter (software encoding) instead of VideoWriterNV (NVENC)
            # The static ffmpeg build doesn't have NVENC support
            vout = ffmpegcv.VideoWriter(
                file=temp_video_path,
                codec='libx264',
                fps=framerate,
                resize=(target_width, target_height)
            )

        if max_score_face:
            mode = "crop"
        else:
            mode = "resize"

        if mode == "resize":
            scale = target_width / img.shape[1]
            resized_height = int(img.shape[0] * scale)
            resized_image = cv2.resize(
                img, (target_width, resized_height), interpolation=cv2.INTER_AREA)

            scale_for_bg = max(
                target_width / img.shape[1], target_height / img.shape[0])
            bg_width = int(img.shape[1] * scale_for_bg)
            bg_heigth = int(img.shape[0] * scale_for_bg)

            blurred_background = cv2.resize(img, (bg_width, bg_heigth))
            blurred_background = cv2.GaussianBlur(
                blurred_background, (121, 121), 0)

            crop_x = (bg_width - target_width) // 2
            crop_y = (bg_heigth - target_height) // 2
            blurred_background = blurred_background[crop_y:crop_y +
                                                    target_height, crop_x:crop_x + target_width]

            center_y = (target_height - resized_height) // 2
            blurred_background[center_y:center_y +
                               resized_height, :] = resized_image

            vout.write(blurred_background)

        elif mode == "crop":
            scale = target_height / img.shape[0]
            resized_image = cv2.resize(
                img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            frame_width = resized_image.shape[1]

            center_x = int(
                max_score_face["x"] * scale if max_score_face else frame_width // 2)
            top_x = max(min(center_x - target_width // 2,
                        frame_width - target_width), 0)

            image_cropped = resized_image[0:target_height,
                                          top_x:top_x + target_width]

            vout.write(image_cropped)

    if vout:
        vout.release()

    ffmpeg_command = (f"ffmpeg -y -i {temp_video_path} -i {audio_path} "
                      f"-c:v h264 -preset fast -crf 23 -c:a aac -b:a 128k "
                      f"{output_path}")
    subprocess.run(ffmpeg_command, shell=True, check=True, text=True)


def create_subtitles_with_ffmpeg(
    transcript_segments: list, 
    clip_start: float, 
    clip_end: float, 
    clip_video_path: str, 
    output_path: str, 
    max_words: int = 5,
    effect: str = "karaoke",
    font: str = "Anton",
    font_size: int = 140,
    color: str = "#FFFFFF",
    highlight_color: str = "#FFD700"
):
    """Create subtitles with animation effects and burn them into the video.
    
    Effects:
    - none: Static text
    - pop: Words scale up when appearing
    - fade: Smooth fade-in
    - karaoke: Word-by-word highlight
    """
    temp_dir = os.path.dirname(output_path)
    subtitle_path = os.path.join(temp_dir, "temp_subtitles.ass")

    # Filter segments for this clip
    clip_segments = [segment for segment in transcript_segments
                     if segment.get("start") is not None
                     and segment.get("end") is not None
                     and segment.get("end") > clip_start
                     and segment.get("start") < clip_end
                     ]

    # Parse color (hex to BGR for ASS)
    def hex_to_ass_color(hex_color: str) -> pysubs2.Color:
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return pysubs2.Color(r, g, b)

    primary_color = hex_to_ass_color(color)
    highlight_ass_color = hex_to_ass_color(highlight_color)

    # Create SSA file
    subs = pysubs2.SSAFile()
    subs.info["WrapStyle"] = 0
    subs.info["ScaledBorderAndShadow"] = "yes"
    subs.info["PlayResX"] = 1080
    subs.info["PlayResY"] = 1920
    subs.info["ScriptType"] = "v4.00+"

    # Base style - Position between 50-75% of screen (lower-center area)
    # For 1920px height:
    # - 50% from top = 960px from top = 960px from bottom
    # - 75% from top = 1440px from top = 480px from bottom
    # - Target: ~62% from top = 38% from bottom = ~730px marginv
    # Using alignment=2 (bottom center) with large marginv to push up
    style_name = "Default"
    new_style = pysubs2.SSAStyle()
    new_style.fontname = font
    new_style.fontsize = font_size
    new_style.primarycolor = primary_color
    new_style.outline = 3.0  # Thicker outline for visibility
    new_style.shadow = 3.0
    new_style.shadowcolor = pysubs2.Color(0, 0, 0, 200)  # Dark shadow
    new_style.alignment = 2  # Bottom center, then push up with margin
    new_style.marginl = 50
    new_style.marginr = 50
    new_style.marginv = 550  # Push up from bottom to ~60-65% from top
    new_style.spacing = 0.0
    new_style.bold = True
    subs.styles[style_name] = new_style

    # Highlight style for karaoke - same positioning
    highlight_style = pysubs2.SSAStyle()
    highlight_style.fontname = font
    highlight_style.fontsize = font_size
    highlight_style.primarycolor = highlight_ass_color
    highlight_style.outline = 3.0
    highlight_style.shadow = 3.0
    highlight_style.shadowcolor = pysubs2.Color(0, 0, 0, 200)
    highlight_style.alignment = 2  # Bottom center
    highlight_style.marginl = 50
    highlight_style.marginr = 50
    highlight_style.marginv = 550  # Push up from bottom
    highlight_style.bold = True
    subs.styles["Highlight"] = highlight_style
    
    print(f"Creating subtitles with effect: {effect}, segments: {len(clip_segments)}")

    if effect == "karaoke":
        # Word-by-word with highlight color change
        word_count = 0
        for segment in clip_segments:
            word = segment.get("word", "").strip()
            seg_start = segment.get("start")
            seg_end = segment.get("end")
            
            if not word or seg_start is None or seg_end is None:
                continue
            
            start_rel = max(0.0, seg_start - clip_start)
            end_rel = max(0.0, seg_end - clip_start)
            
            if end_rel <= 0:
                continue
            
            # Create event with karaoke effect
            start_time = pysubs2.make_time(s=start_rel)
            end_time = pysubs2.make_time(s=end_rel + 0.3)  # Slight extension
            
            # Use ASS karaoke tag for color change
            # {\c&HBBGGRR&} for highlight color
            highlight_hex = f"&H{highlight_color[5:7]}{highlight_color[3:5]}{highlight_color[1:3]}&"
            text = f"{{\\c{highlight_hex}}}{word}"
            
            line = pysubs2.SSAEvent(
                start=start_time, end=end_time, text=text, style=style_name
            )
            subs.events.append(line)
            word_count += 1
        
        print(f"Created {word_count} karaoke word events")
    
    else:
        # Group words into lines for other effects
        subtitles = []
        current_words = []
        current_start = None
        current_end = None

        for segment in clip_segments:
            word = segment.get("word", "").strip()
            seg_start = segment.get("start")
            seg_end = segment.get("end")

            if not word or seg_start is None or seg_end is None:
                continue

            start_rel = max(0.0, seg_start - clip_start)
            end_rel = max(0.0, seg_end - clip_start)

            if end_rel <= 0:
                continue

            if not current_words:
                current_start = start_rel
                current_end = end_rel
                current_words = [word]
            elif len(current_words) >= max_words:
                subtitles.append((current_start, current_end, ' '.join(current_words)))
                current_words = [word]
                current_start = start_rel
                current_end = end_rel
            else:
                current_words.append(word)
                current_end = end_rel

        if current_words:
            subtitles.append((current_start, current_end, ' '.join(current_words)))

        print(f"Grouped {len(subtitles)} subtitle lines for effect: {effect}")

        for i, (start, end, text) in enumerate(subtitles):
            start_time = pysubs2.make_time(s=start)
            end_time = pysubs2.make_time(s=end)
            
            # Apply animation effect via ASS tags
            if effect == "pop":
                # Pop effect: Scale from 50% to 110% then settle at 100%
                # Using simpler scale animation that works better with ASS
                text = f"{{\\fscx50\\fscy50\\t(0,150,\\fscx110\\fscy110)\\t(150,250,\\fscx100\\fscy100)}}{text}"
            elif effect == "fade":
                # Fade in over 300ms, fade out over 100ms
                text = f"{{\\fad(300,100)}}{text}"
            # else "none" - no modification, just show static text
            
            line = pysubs2.SSAEvent(
                start=start_time, end=end_time, text=text, style=style_name
            )
            subs.events.append(line)

    print(f"Total subtitle events created: {len(subs.events)}")
    subs.save(subtitle_path)
    print(f"Saved subtitle file: {subtitle_path}")

    # Run ffmpeg to burn subtitles
    ffmpeg_cmd = (f"ffmpeg -y -i {clip_video_path} -vf \"ass={subtitle_path}\" "
                  f"-c:v h264 -preset fast -crf 23 -c:a aac {output_path}")

    print(f"Running ffmpeg command to burn subtitles...")
    result = subprocess.run(ffmpeg_cmd, shell=True, check=True, capture_output=True, text=True)
    print(f"Subtitles burned successfully into video: {output_path}")


def process_clip(
    base_dir: str, 
    original_video_path: str, 
    s3_key: str, 
    start_time: float, 
    end_time: float, 
    clip_index: int, 
    transcript_segments: list,
    caption_effect: str = "karaoke",
    caption_font: str = "Anton",
    caption_font_size: int = 140,
    caption_color: str = "#FFFFFF",
    highlight_color: str = "#FFD700"
):
    clip_name = f"clip_{clip_index}"
    s3_key_dir = os.path.dirname(s3_key)
    output_s3_key = f"{s3_key_dir}/{clip_name}.mp4"
    print(f"Output S3 key: {output_s3_key}")

    clip_dir = base_dir / clip_name
    clip_dir.mkdir(parents=True, exist_ok=True)

    clip_segment_path = clip_dir / f"{clip_name}_segment.mp4"
    vertical_mp4_path = clip_dir / "pyavi" / "video_out_vertical.mp4"
    subtitle_output_path = clip_dir / "pyavi" / "video_with_subtitles.mp4"

    (clip_dir / "pywork").mkdir(exist_ok=True)
    pyframes_path = clip_dir / "pyframes"
    pyavi_path = clip_dir / "pyavi"
    audio_path = clip_dir / "pyavi" / "audio.wav"

    pyframes_path.mkdir(exist_ok=True)
    pyavi_path.mkdir(exist_ok=True)

    duration = end_time - start_time
    cut_command = (f"ffmpeg -i {original_video_path} -ss {start_time} -t {duration} "
                   f"{clip_segment_path}")
    subprocess.run(cut_command, shell=True, check=True,
                   capture_output=True, text=True)

    extract_cmd = f"ffmpeg -i {clip_segment_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
    subprocess.run(extract_cmd, shell=True,
                   check=True, capture_output=True)

    shutil.copy(clip_segment_path, base_dir / f"{clip_name}.mp4")

    columbia_command = (f"python Columbia_test.py --videoName {clip_name} "
                        f"--videoFolder {str(base_dir)} "
                        f"--pretrainModel weight/finetuning_TalkSet.model")

    columbia_start_time = time.time()
    subprocess.run(columbia_command, cwd="/asd", shell=True)
    columbia_end_time = time.time()
    print(
        f"Columbia script completed in {columbia_end_time - columbia_start_time:.2f} seconds")

    tracks_path = clip_dir / "pywork" / "tracks.pckl"
    scores_path = clip_dir / "pywork" / "scores.pckl"
    if not tracks_path.exists() or not scores_path.exists():
        raise FileNotFoundError("Tracks or scores not found for clip")

    with open(tracks_path, "rb") as f:
        tracks = pickle.load(f)

    with open(scores_path, "rb") as f:
        scores = pickle.load(f)

    cvv_start_time = time.time()
    create_vertical_video(
        tracks, scores, pyframes_path, pyavi_path, audio_path, vertical_mp4_path
    )
    cvv_end_time = time.time()
    print(
        f"Clip {clip_index} vertical video creation time: {cvv_end_time - cvv_start_time:.2f} seconds")

    create_subtitles_with_ffmpeg(
        transcript_segments, 
        start_time,
        end_time, 
        vertical_mp4_path, 
        subtitle_output_path, 
        max_words=5,
        effect=caption_effect,
        font=caption_font,
        font_size=caption_font_size,
        color=caption_color,
        highlight_color=highlight_color
    )

    s3_client = boto3.client("s3")
    s3_client.upload_file(
        subtitle_output_path, "josh-video-clipper", output_s3_key)



def parse_youtube_subtitles(srt_content: str):
    """
    Parses SRT content strings into word-level segments with interpolated timing.
    Returns: [{"start": float, "end": float, "word": str}, ...]
    """
    subs = pysubs2.SSAFile.from_string(srt_content)
    segments = []

    for event in subs:
        # Pysubs2 times are in milliseconds
        start_sec = event.start / 1000.0
        end_sec = event.end / 1000.0
        duration = end_sec - start_sec
        
        # Clean text
        text = event.text.replace("\\N", " ").strip()
        # Remove HTML-like tags if any
        text = re.sub(r'<[^>]+>', '', text)
        
        words = text.split()
        if not words:
            continue
            
        word_duration = duration / len(words)
        
        for i, word in enumerate(words):
            word_start = start_sec + (i * word_duration)
            word_end = start_sec + ((i + 1) * word_duration)
            segments.append({
                "start": float(f"{word_start:.3f}"),
                "end": float(f"{word_end:.3f}"),
                "word": word
            })
            
    return segments


@app.cls(gpu="L40S", timeout=900, retries=0, scaledown_window=20, secrets=[modal.Secret.from_name("ai-podcast-clipper-secret")], volumes={mount_path: volume})
class AiPodcastClipper:
    @modal.enter()
    def load_model(self):
        self.whisperx_model = whisperx.load_model(
            "large-v2", device="cuda", compute_type="float16")

        self.alignment_model, self.metadata = whisperx.load_align_model(
            language_code="en",
            device="cuda"
        )

        print("Transcription models loaded...")

        print("Creating gemini client...")
        self.gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        print("Created gemini client...")


    def transcribe_video(self, base_dir: str, video_path: str) -> str:
        audio_path = base_dir / "audio.wav"
        extract_cmd = f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
        subprocess.run(extract_cmd, shell=True,
                       check=True, capture_output=True)

        print("Starting transcription with WhisperX (English only)...")
        start_time = time.time()

        audio = whisperx.load_audio(str(audio_path))
        
        # Force English language to prevent hallucinations in other languages
        result = self.whisperx_model.transcribe(
            audio, 
            batch_size=16,
            language="en"  # Force English only
        )

        result = whisperx.align(
            result["segments"],
            self.alignment_model,
            self.metadata,
            audio,
            device="cuda",
            return_char_alignments=False
        )

        duration = time.time() - start_time
        print("Transcription and alignment took " + str(duration) + " seconds")
        
        # Filter out hallucinated/repeated words
        segments = []
        last_word = ""
        repeat_count = 0
        max_repeats = 3  # Allow max 3 consecutive same words

        if "word_segments" in result:
            for word_segment in result["word_segments"]:
                if "start" not in word_segment or "end" not in word_segment:
                    continue
                    
                word = word_segment["word"].strip().lower()
                
                # Check for repeated words (hallucination sign)
                if word == last_word:
                    repeat_count += 1
                    if repeat_count > max_repeats:
                        print(f"Skipping hallucinated repeat: {word}")
                        continue
                else:
                    repeat_count = 0
                    last_word = word
                
                segments.append({
                    "start": word_segment["start"],
                    "end": word_segment["end"],
                    "word": word_segment["word"],
                })
        
        print(f"Transcription complete: {len(segments)} words extracted")
        return json.dumps(segments)

    def identify_moments(self, transcript: dict, mode: str = "podcast"):
        base_prompt = ""
        
        if mode == "sermon":
            base_prompt = """
# Viral Sermon Clip Extraction

You are an expert at identifying viral-worthy sermon moments. Your input is a sermon transcript with word-level start/end timestamps. 

## Your Mission
Select HIGH-IMPACT clips (60–90 seconds each) that **stop the scroll and deliver a COMPLETE message**. These clips should work as standalone content on social media platforms like TikTok, Instagram Reels, and YouTube Shorts.

## What Makes a Viral Sermon Clip

### 1. The Hook (CRITICAL - First 3 seconds)
The opening MUST immediately grab attention with one of these:
- A bold, provocative claim that challenges conventional thinking
- A thought-provoking question that creates curiosity
- An unexpected statement that surprises the viewer
- A relatable scenario that makes viewers think "that's me!"
- A counterintuitive insight that defies expectations

### 2. The Body (Core Content)
Look for these types of content:
- **"Aha" Moments**: Profound, standalone theological insights or "mic drop" quotes that resonate universally
- **Powerful Illustrations**: Brief stories, metaphors, or analogies that illuminate a spiritual truth with vivid imagery
- **Relatable Struggles**: Segments addressing common human emotions (anxiety, fear, doubt, loneliness, failure, regret) with faith-based perspective
- **Practical Wisdom**: Actionable advice, life principles, or transformative mindset shifts
- **Inspirational Exhortations**: Encouraging words that uplift, challenge, or motivate
- **Counter-Cultural Truths**: Messages that challenge worldly perspectives with spiritual wisdom
- **Redemption Narratives**: Stories of hope, healing, second chances, or transformation

### 3. The Payoff (CRITICAL - Last sentence)
The clip MUST end with:
- A clear resolution or insight that fulfills the hook's promise
- A memorable conclusion, lesson, or call-to-action
- A satisfying "landing" that leaves viewers thinking

## Strict Rules

### Length Requirements
- **Target: 60-90 seconds** per clip
- Minimum: 60 seconds (complete thoughts only)
- Maximum: 90 seconds (hard limit - do not exceed)
- Shorter clips are acceptable ONLY if the thought is genuinely complete and impactful

### Timestamp Rules  
- **Use EXACT timestamps provided** - do not modify or estimate
- Start each clip at the FIRST WORD of a sentence
- End each clip at the LAST WORD of a sentence
- This ensures clean sentence boundaries with no cut-off thoughts

### Ordering and Overlap
- Clips must be chronologically ordered
- Clips must NOT overlap - each timestamp belongs to at most one clip
- Leave buffer between clips for clean extraction

### Content to EXCLUDE
- Opening greetings, welcomes, or announcements
- Housekeeping (sit down, stand up, turn to your neighbor, etc.)
- Pure scripture reading without commentary or application
- Prayers without teaching content
- Instructions specific to the live setting ("over there," "back here")
- References to specific people present in the room by name
- Incomplete thoughts that require outside context
- Technical difficulties or interruptions
- Transitions between topics without substance

## Output Format

Return ONLY a JSON array. Each object must have 'start' and 'end' keys with values in seconds:
[{"start": seconds, "end": seconds}, {"start": seconds, "end": seconds}, ...]

The output MUST be valid JSON that can be parsed by Python's json.loads() function.
"""
        else:
            # Default Podcast Prompt (Preserved from existing main.py)
            base_prompt = """
    This is a podcast video transcript consisting of word, along with each words's start and end time. I am looking to create clips between a minimum of 30 and maximum of 60 seconds long, should not be less than 30 seconds. The clip should never exceed 60 seconds.

    Your task is to find and extract stories, or question and their corresponding answers from the transcript.
    Each clip should begin with the question and conclude with the answer.
    It is acceptable for the clip to include a few additional sentences before a question if it aids in contextualizing the question.

    Please adhere to the following rules:
    - Ensure that clips do not overlap with one another.
    - Start and end timestamps of the clips should align perfectly with the sentence boundaries in the transcript.
    - Only use the start and end timestamps provided in the input. modifying timestamps is not allowed.
    - Format the output as a list of JSON objects, each representing a clip with 'start' and 'end' timestamps: [{"start": seconds, "end": seconds}, ...clip2, clip3]. The output should always be readable by the python json.loads function.
    - Aim to generate longer clips between 40-60 seconds, and ensure to include as much content from the context as viable.

    Avoid including:
    - Moments of greeting, thanking, or saying goodbye.
    - Non-question and answer interactions.
    """

        full_content = base_prompt + """

## Final Instructions

CRITICAL REQUIREMENTS:
1. Return the MOST VIRAL-WORTHY clips - prioritize QUALITY over quantity
2. Maximum 10 clips, but only include clips that are truly exceptional
3. Clips MUST be in chronological order (sorted by start time)
4. Each clip MUST be 60-90 seconds (shorter only if thought is genuinely complete)
5. Every clip must have a clear HOOK at the start and PAYOFF at the end
6. If no clips meet the quality bar, return an empty array []

OUTPUT FORMAT:
- Return ONLY a valid JSON array
- No markdown, no explanation, no extra text
- Must be parseable by Python's json.loads()
- Example: [{"start": 120.5, "end": 185.2}, {"start": 300.0, "end": 375.8}]

The transcript is as follows:

""" + str(transcript)

        response = self.gemini_client.models.generate_content(
            model="gemini-2.5-flash", contents=full_content)
        print(f"Identified moments response for mode {mode}: " + f"${response.text}")
        return response.text

    @modal.fastapi_endpoint(method="POST")
    def process_video(self, request: ProcessVideoRequest, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
        print("Processing video...")

        if token.credentials != os.environ["AUTH_TOKEN"]:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail="Incorrect bearer token", headers={"WWW-Authenticate": "Bearer"})

        run_id = str(uuid.uuid4())
        base_dir = pathlib.Path("/tmp") / run_id
        base_dir.mkdir(parents=True, exist_ok=True)

        s3_key = request.s3_key
        
        # Download video file from S3
        video_path = base_dir / "input.mp4"
        s3_client = boto3.client("s3")
        s3_client.download_file("josh-video-clipper", s3_key, str(video_path))
        
        # Transcribe with WhisperX
        print("Running WhisperX transcription...")
        transcript_segments_json = self.transcribe_video(base_dir, video_path)
        transcript_segments = json.loads(transcript_segments_json)

        # Identify moments for clips
        print(f"Identifying clip moments (Mode: {request.video_type})")
        identified_moments_raw = self.identify_moments(transcript_segments, mode=request.video_type)

        cleaned_json_string = identified_moments_raw.strip()
        
        # Locate the start of the JSON list
        json_start_index = cleaned_json_string.find("[")
        if json_start_index != -1:
             cleaned_json_string = cleaned_json_string[json_start_index:]
             
        # Locate the end of the JSON list
        json_end_index = cleaned_json_string.rfind("]")
        if json_end_index != -1:
            cleaned_json_string = cleaned_json_string[:json_end_index+1]
            
        if cleaned_json_string.startswith("```json"):
            cleaned_json_string = cleaned_json_string[len("```json"):].strip()
        if cleaned_json_string.endswith("```"):
            cleaned_json_string = cleaned_json_string[:-len("```")].strip()

        # Try to parse JSON, handle truncated responses gracefully
        clip_moments = []
        try:
            clip_moments = json.loads(cleaned_json_string)
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            # Try to salvage what we can - find last complete object
            # Look for the last complete "}" followed by comma or bracket
            last_complete = cleaned_json_string.rfind("},")
            if last_complete > 0:
                try:
                    salvaged = cleaned_json_string[:last_complete+1] + "]"
                    clip_moments = json.loads(salvaged)
                    print(f"Salvaged {len(clip_moments)} moments from truncated response")
                except:
                    print("Could not salvage truncated JSON")
                    clip_moments = []
            else:
                clip_moments = []
        
        if not clip_moments or not isinstance(clip_moments, list):
            print("Warning: No valid clip moments identified, using empty list")
            clip_moments = []

        print(f"Identified {len(clip_moments)} clip moments")

        # Filter out excluded ranges (previously generated clips)
        if request.exclude_ranges and len(request.exclude_ranges) > 0:
            filtered_moments = []
            print(f"Filtering against {len(request.exclude_ranges)} existing clips")
            for moment in clip_moments:
                start = moment.get("start")
                end = moment.get("end")
                if start is None or end is None:
                    continue
                    
                is_excluded = False
                for existing in request.exclude_ranges:
                    ex_start = existing.get("start")
                    ex_end = existing.get("end")
                    if ex_start is None or ex_end is None:
                        continue
                        
                    # Check overlap
                    overlap_start = max(start, ex_start)
                    overlap_end = min(end, ex_end)
                    overlap = max(0, overlap_end - overlap_start)
                    
                    # If overlap is > 30% of the clip duration, exclude it
                    duration = end - start
                    if duration > 0 and (overlap / duration) > 0.3:
                        is_excluded = True
                        print(f"Excluding moment {start}-{end} due to overlap with {ex_start}-{ex_end}")
                        break
                
                if not is_excluded:
                    filtered_moments.append(moment)
            
            clip_moments = filtered_moments
            print(f"Remaining moments after filtering: {len(clip_moments)}")

        # Process video clips
        s3_key_dir = os.path.dirname(s3_key)
        clips_to_process = clip_moments[:request.max_clips]
        processed_clips = []
        
        for index, moment in enumerate(clips_to_process):
            if "start" in moment and "end" in moment:
                print("Processing clip" + str(index) + " from " +
                      str(moment["start"]) + " to " + str(moment["end"]))
                process_clip(
                    base_dir, video_path, s3_key,
                    moment["start"], moment["end"], index, transcript_segments,
                    caption_effect=request.caption_effect,
                    caption_font=request.caption_font,
                    caption_font_size=request.caption_font_size,
                    caption_color=request.caption_color,
                    highlight_color=request.highlight_color
                )
                processed_clips.append({
                    "start": moment["start"],
                    "end": moment["end"],
                    "s3_key": f"{s3_key_dir}/clip_{index}.mp4"
                })

        if base_dir.exists():
            print(f"Cleaning up temp dir after {base_dir}")
            shutil.rmtree(base_dir, ignore_errors=True)
        
        # Return transcript and clip data for Convex storage
        # Include ALL identified moments (for future regeneration) and processed clips
        return {
            "status": "success", 
            "s3_key": s3_key, 
            "clips_created": len(processed_clips),
            "transcript_segments": transcript_segments,
            "clip_moments": processed_clips,  # Only the clips that were actually created
            "all_identified_moments": clip_moments,  # ALL moments identified by AI (for regeneration)
            "processed_indices": list(range(len(processed_clips)))  # Which indices were processed
        }


@app.local_entrypoint()
def main():
    import requests

    ai_podcast_clipper = AiPodcastClipper()

    url = ai_podcast_clipper.process_video.web_url

    payload = {
        "s3_key": "test2/journeysermon.mp4",
        "video_type": "sermon"
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer 123123"
    }

    response = requests.post(url, json=payload,
                             headers=headers)
    response.raise_for_status()
    result = response.json()
    print(result)
