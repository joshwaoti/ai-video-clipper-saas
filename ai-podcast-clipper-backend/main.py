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
from ultralytics import YOLO
import parselmouth
from parselmouth.praat import call


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
    .pip_install("ultralytics", "DeepFilterNet", "praat-parselmouth")
    .add_local_dir("asd", "/asd", copy=True, ignore=[".git", "**/.git"]))

app = modal.App("ai-podcast-clipper", image=image)

volume = modal.Volume.from_name(
    "ai-podcast-clipper-model-cache", create_if_missing=True
)

mount_path = "/root/.cache/torch"

auth_scheme = HTTPBearer()


class CinematicCamera:
    """
    Implements a virtual camera with stabilization, dead zones, and damping
    to mimic a professional camera operator.
    """
    def __init__(self, target_width=1080, target_height=1920, dead_zone_ratio=0.15, alpha=0.08):
        self.target_width = target_width
        self.target_height = target_height
        self.dead_zone_ratio = dead_zone_ratio
        self.alpha = alpha  # Smoothing factor (lower = smoother/slower)
        self.current_center_x = None

    def update(self, frame_width, frame_height, subject_box):
        """
        Update camera position based on subject detection.
        subject_box: [x1, y1, x2, y2]
        """
        if subject_box is None:
            # If subject lost, stay put or drift to center? Stay put is safer to avoid jumps.
            target_x = self.current_center_x if self.current_center_x is not None else frame_width / 2
        else:
            x1, y1, x2, y2 = subject_box
            target_x = (x1 + x2) / 2

        # Initialize
        if self.current_center_x is None:
            self.current_center_x = target_x

        # Dead Zone Logic
        # If target moves only slightly (within dead zone relative to current camera), ignore.
        # Calculate distance from current camera center
        diff = target_x - self.current_center_x
        dead_zone_width = frame_width * self.dead_zone_ratio
        
        if abs(diff) < (dead_zone_width / 2):
            # Inside dead zone - hold current position (target is effectively "current")
            effective_target = self.current_center_x
        else:
            # Outside - move towards the subject, but respect the dampening
            effective_target = target_x

        # Apply PID/Exponential Damping
        self.current_center_x = self.current_center_x + self.alpha * (effective_target - self.current_center_x)

        # Boundary checks
        half_w = self.target_width / 2
        min_x = half_w
        max_x = frame_width - half_w
        final_center_x = max(min_x, min(self.current_center_x, max_x))
        self.current_center_x = final_center_x

        # Calculate crop top-left
        crop_x = int(final_center_x - half_w)
        
        # Vertical centering (assuming subject is roughly vertically centered or we crop center)
        # Ideally we track vertical too, but horizontal is most critical for landscape->vertical
        crop_y = (frame_height - self.target_height) // 2
        
        return crop_x, crop_y


def enhance_audio(input_path: str, output_path: str):
    """
    Enhance audio using DeepFilterNet to remove reverb and noise.
    """
    print(f"Enhancing audio with DeepFilterNet: {input_path}")
    # We use subprocess to call deepFilter. 
    # Assumes 'deepFilter' is in PATH (installed via pip).
    # Output dir logic: deepFilter outputs to a dir.
    
    output_dir = os.path.dirname(output_path)
    # deepFilter generates file with suffix, so we might need to rename.
    
    cmd = [
        "deepFilter",
        input_path,
        "-o", output_dir
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Find the generated file. DeepFilterNet usually appends _DeepFilterNet3.wav
        # We need to find it and rename to output_path
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        # glob for candidate
        candidates = glob.glob(os.path.join(output_dir, f"{base_name}*DeepFilterNet*.wav"))
        if candidates:
            # Success
            enhanced_file = candidates[0]
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(enhanced_file, output_path)
            print(f"Audio enhancement complete: {output_path}")
        else:
            print("DeepFilterNet output not found, using original.")
            shutil.copy(input_path, output_path)
            
    except Exception as e:
        print(f"DeepFilterNet failed: {e}. Using original audio.")
        shutil.copy(input_path, output_path)


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


def analyze_video_yolo(video_path: str, model_name="yolov8n.pt"):
    """
    Run YOLOv8 tracking on the video.
    Returns list of frame detections: [ [x1,y1,x2,y2] or None, ... ]
    """
    print(f"Running YOLOv8 tracking on {video_path}...")
    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"Failed to load YOLO model: {e}. Downloading default.")
        model = YOLO("yolov8n.pt")

    # Run tracking
    results = model.track(source=video_path, stream=True, persist=True, verbose=False, classes=[0]) # class 0 is person
    
    detections = []
    for r in results:
        # Get best person detection (highest confidence)
        best_box = None
        max_conf = -1.0
        
        if r.boxes:
            for box in r.boxes:
                # We already filtered classes=[0] but check to be safe
                if int(box.cls[0]) == 0:
                    conf = float(box.conf[0])
                    if conf > max_conf:
                        max_conf = conf
                        best_box = box.xyxy[0].tolist() # [x1, y1, x2, y2]
        
        detections.append(best_box)
        
    return detections


def create_vertical_video_sermon(video_path, detections, output_path, audio_path):
    """
    Create vertical video using CinematicCamera logic and YOLO detections.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize Camera
    cam = CinematicCamera(target_width=1080, target_height=1920)
    
    # Validate video properties
    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError(f"Invalid video properties: width={width}, height={height}, fps={fps}")
    
    print(f"Source video: {width}x{height} @ {fps:.2f} fps")
    
    # Setup Writer using temp file, ensure directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    temp_video = os.path.join(output_dir, "temp_sermon_raw.mp4")
    temp_encoded = os.path.join(output_dir, "temp_sermon_encoded.mp4")
    
    # Use cv2.VideoWriter instead of ffmpegcv (more reliable)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(temp_video, fourcc, fps, (1080, 1920))
    
    if not writer.isOpened():
        raise ValueError(f"Failed to create video writer for {temp_video}")
    
    idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Rendering sermon video: {total_frames} frames")
    
    # We must iterate matches detections
    # Note: detections might be shorter/longer than cv2 reads depending on sync
    
    for _ in tqdm(range(total_frames), desc="Rendering Sermon"):
        ret, frame = cap.read()
        if not ret:
            break
        
        if idx < len(detections):
            box = detections[idx]
        else:
            box = None
            
        # Determine Scale to fit Height 1920
        # If source height < 1920, scale up. If > 1920, scale down.
        # We enforce filling vertical space.
        target_h = 1920
        scale = target_h / height
        
        # Scale the frame
        # cv2.resize might be slow for 4K. 
        # Optimization: Crop horizontally first? No, we need center X first.
        
        # Scale box to new dimensions
        # Original: box matches 'frame' (width x height)
        # Scaled: box matches (width*scale x 1920)
        
        if box:
            box_scaled = [c * scale for c in box]
        else:
            box_scaled = None
            
        w_scaled = int(width * scale)
        
        # Update Camera
        crop_x, crop_y = cam.update(w_scaled, target_h, box_scaled)
        
        # Optimization: Don't resize full frame if we only need a crop.
        # We need [crop_y : crop_y+1920, crop_x : crop_x+1080] from the SCALED image.
        # In ORIGINAL matching, this is:
        # [crop_y/scale : ..., crop_x/scale : ...]
        
        orig_crop_x = int(crop_x / scale)
        orig_crop_y = int(crop_y / scale)
        orig_crop_w = max(1, int(1080 / scale))  # Ensure minimum of 1
        orig_crop_h = max(1, int(1920 / scale))  # Ensure minimum of 1
        
        # Clamp original crop - ensure valid bounds
        orig_crop_x = max(0, min(orig_crop_x, width - orig_crop_w))
        orig_crop_y = max(0, min(orig_crop_y, height - orig_crop_h))
        
        # Ensure we have valid crop dimensions
        if orig_crop_w <= 0 or orig_crop_h <= 0 or orig_crop_x < 0 or orig_crop_y < 0:
            # Fallback: resize entire frame to target
            final_frame = cv2.resize(frame, (1080, 1920), interpolation=cv2.INTER_LINEAR)
        else:
            # Crop from original
            crop_orig = frame[orig_crop_y : orig_crop_y + orig_crop_h, 
                              orig_crop_x : orig_crop_x + orig_crop_w]
            
            # Check crop is valid before resizing
            if crop_orig.size == 0 or crop_orig.shape[0] == 0 or crop_orig.shape[1] == 0:
                final_frame = cv2.resize(frame, (1080, 1920), interpolation=cv2.INTER_LINEAR)
            else:
                # Now resize ONLY the crop to 1080x1920
                final_frame = cv2.resize(crop_orig, (1080, 1920), interpolation=cv2.INTER_LINEAR)
        
        # Ensure frame is in correct format (BGR, uint8, correct dimensions)
        if final_frame.shape != (1920, 1080, 3):
            final_frame = cv2.resize(final_frame, (1080, 1920), interpolation=cv2.INTER_LINEAR)
        
        writer.write(final_frame)
        idx += 1
        
    writer.release()
    cap.release()
    
    print(f"Frames written: {idx}")
    
    # Re-encode with ffmpeg for better compatibility (cv2 mp4v codec may not be widely compatible)
    print("Re-encoding video with libx264...")
    subprocess.run(
        f"ffmpeg -y -i {temp_video} -c:v libx264 -preset fast -crf 23 {temp_encoded}",
        shell=True, check=True, capture_output=True
    )
    
    # Mux Audio
    print("Muxing audio...")
    subprocess.run(f"ffmpeg -y -i {temp_encoded} -i {audio_path} -c:v copy -c:a aac {output_path}", 
                   shell=True, check=True)
    
    # Cleanup
    if os.path.exists(temp_video):
        os.remove(temp_video)
    if os.path.exists(temp_encoded):
        os.remove(temp_encoded)


def analyze_prosody(audio_path: str):
    """
    Analyze audio prosody to identify rhetorical intensity peaks.
    Returns a list of timestamps (seconds) where intensity/pitch is high.
    """
    print(f"Analyzing prosody for {audio_path}...")
    try:
        sound = parselmouth.Sound(audio_path)
        # Extract intensity
        intensity = sound.to_intensity()
        values = intensity.values[0] # The array
        times = intensity.xs() # The times
        
        if len(values) == 0:
            return []
            
        mean_intensity = np.mean(values)
        std_intensity = np.std(values)
        threshold = mean_intensity + 1.5 * std_intensity
        
        peaks = []
        # Finding continuous segments > threshold
        # We cluster them to avoid returning every millisecond
        
        current_peak_start = None
        min_duration = 2.0 # Minimum high-intensity duration
        
        for t, val in zip(times, values):
            if val > threshold:
                if current_peak_start is None:
                    current_peak_start = t
            else:
                if current_peak_start is not None:
                    duration = t - current_peak_start
                    if duration > min_duration:
                        # Append the range
                        peaks.append(f"{current_peak_start:.1f}s - {t:.1f}s")
                    current_peak_start = None
                    
        print(f"Detected {len(peaks)} rhetorical peaks")
        return peaks
    except Exception as e:
        print(f"Prosody analysis failed: {e}")
        return []

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
    video_type: str = "podcast",
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
    cut_command = (f"ffmpeg -i '{original_video_path}' -ss {start_time} -t {duration} "
                   f"{clip_segment_path}")
    subprocess.run(cut_command, shell=True, check=True,
                   capture_output=True, text=True)

    extract_cmd = f"ffmpeg -i {clip_segment_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
    subprocess.run(extract_cmd, shell=True,
                   check=True, capture_output=True)

    # Audio Enhancement (Sermon Only)
    if video_type == "sermon":
        try:
            enhanced_audio_path = clip_dir / "pyavi" / "audio_enhanced.wav"
            enhance_audio(str(audio_path), str(enhanced_audio_path))
            # Replace original with enhanced
            shutil.move(str(enhanced_audio_path), str(audio_path))
            print("Using enhanced audio for processing")
        except Exception as e:
            print(f"Audio enhancement skipped due to error: {e}")

    # Video Processing
    cvv_start_time = time.time()
    if video_type == "sermon":
        print("Using Sermon Pipeline (YOLO + Cinematic Camera)")
        # Run YOLO Detection
        detections = analyze_video_yolo(str(clip_segment_path))
        
        # Create Vertical Video
        create_vertical_video_sermon(
            str(clip_segment_path), 
            detections, 
            str(vertical_mp4_path),
            str(audio_path)
        )
    else:
        print("Using Podcast Pipeline (Columbia TalkNet)")
        # Legacy Pipeline
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
        
        # Extract audio from video (video_path is always a local file path now)
        extract_cmd = f"ffmpeg -i '{video_path}' -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
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

    def identify_moments(self, transcript: dict, mode: str = "podcast", rhetorical_peaks: list = None):
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
            if rhetorical_peaks:
                base_prompt += "\n\n## Audio Cues (Rhetorical Intensity)\nThe following timestamps marked high vocal intensity. Consider them as potential high-impact segments:\n"
                base_prompt += "\n".join([f"- {p}" for p in rhetorical_peaks[:20]]) + "\n"
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
        
        # Always download video locally to avoid ffmpeg issues with presigned URLs
        # FFmpeg can crash (exit 139 - segfault) when processing URLs with special characters
        s3_client = boto3.client("s3")
        print(f"Downloading video from S3: {s3_key}")
        video_path_local = base_dir / "input_video.mp4"
        s3_client.download_file("josh-video-clipper", s3_key, str(video_path_local))
        video_path = str(video_path_local)
        print(f"Video downloaded to: {video_path}")
        
        # Transcribe with WhisperX
        print("Running WhisperX transcription...")
        transcript = self.transcribe_video(base_dir, video_path)
        transcript_segments = json.loads(transcript)
        
        # Analyze Prosody (Sermon Only)
        rhetorical_peaks = None
        if request.video_type == "sermon":
             audio_path_local = base_dir / "audio.wav"
             # analyze_prosody expects file path
             rhetorical_peaks = analyze_prosody(str(audio_path_local))

        print("Identifying viral moments with Gemini...")
        identified_moments_raw = self.identify_moments(
            transcript_segments, 
            mode=request.video_type,
            rhetorical_peaks=rhetorical_peaks
        )

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
                    video_type=request.video_type,
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
