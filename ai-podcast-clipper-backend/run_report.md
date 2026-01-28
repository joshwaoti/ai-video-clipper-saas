# Run Report: Sermon Clipper Test Run

**Date:** 2025-12-17
**Mode:** Sermon
**Model:** Gemini 1.5 Flash
**Input:** `test2/journeysermon.mp4`

## Executive Summary
The application successfully executed the full pipeline:
1.  **Ingestion:** Downloaded video from S3.
2.  **Transcription:** Performed by WhisperX (since no YouTube URL was provided).
3.  **Analysis:** Gemini 1.5 Flash identified **34** potential viral moments based on the "Sermon" prompt.
4.  **Processing:** The first **3 clips** were generated producing vertical crops with subtitles.

## Clip Analysis

The following 3 clips were processed and uploaded to S3. To evaluate the quality, please watch these specific segments in your original video.

### Clip 1
*   **Timestamp:** `06:26` - `07:05` (38 seconds)
*   **Exact Seconds:** 386.981 - 425.088
*   **Why AI picked it:**
    *   **Duration:** It fits comfortably in the "sweet spot" (30-60s). short form content.
    *   **Structure:** Gemini identified a clear "Hook" at 6:26 and a "Payoff" at 7:05. This likely corresponds to a single completed thought or illustration.

### Clip 2
*   **Timestamp:** `08:08` - `09:06` (58 seconds)
*   **Exact Seconds:** 488.095 - 546.021
*   **Why AI picked it:**
    *   **Maximized Length:** At ~58 seconds, this clip pushes the 60s limit, suggesting a more complex story or "Aha" moment that required full context.
    *   **High Engagement:** This section likely contains a sustained "Relatable Struggle" or deeper theological insight that couldn't be compressed further.

### Clip 3
*   **Timestamp:** `10:29` - `11:25` (56 seconds)
*   **Exact Seconds:** 629.348 - 685.297
*   **Why AI picked it:**
    *   **Narrative Arc:** Similar to Clip 2, this is a long format short. It likely contains a full illustration or story.
    *   **Sermon Mode Logic:** The prompt specifically requested "Illustrations" and "Relatable Struggles"—longer clips often capture these better than quick one-liners.

## Technical Observations
*   **Lip Sync:** You noted a slight issue. This is likely due to the `Wav2Lip` or `Columbia` AI crop logic (the `create_vertical_video` function) or potential frame rate mismatches (25fps was used).
*   **Resource Usage:** The run took ~486 seconds (8 minutes) total.
*   **Gemini 1.5 Flash:** Provided stable output without the "Resource Exhausted" errors of the 2.0 experimental model.

## Recommendations for Prompt Refinement
If you find these clips lack "virality" upon review:
1.  **Tighten the Hook:** Modify prompt to strictly require the first 5 seconds to be "attention-grabbing".
2.  **Emotion Focus:** Add instructions to prioritize segments with high emotional inflection or volume changes (if we had audio features, but for now we can ask the LLM to infer intensity from text).
3.  **Length Variation:** If 58s is too long, reduce the max length to 45s to force snappier clips.

