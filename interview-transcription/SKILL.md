---
name: interview-transcription
description: Interview management, transcription workflows, and source note-taking for journalists. Use when preparing for interviews, managing recordings, transcribing audio/video, organizing source notes, creating timestamped references, or building interview databases. Essential for reporters conducting interviews and managing source relationships.
---

# Interview transcription and management

Practical workflows for journalists managing interviews from preparation through publication.

## When to activate

- Preparing questions for an interview
- Processing audio/video recordings
- Creating or managing transcripts
- Organizing notes from multiple sources
- Building a source relationship database
- Generating timestamped quotes for fact-checking
- Converting recordings to publishable quotes

## Pre-interview preparation

### Research checklist

Before recording starts, you should already know:

```markdown
## Source prep for: [Name]

### Background
- Role/title:
- Organization:
- Why they're relevant to this story:
- Previous media appearances (note inconsistencies):

### Key questions (prioritized)
1. [Must-ask question]
2. [Must-ask question]
3. [If time permits]

### Documents to reference
- [ ] Bring/share [specific document]
- [ ] Ask about [specific claim/data point]

### Red lines
- Topics they'll likely avoid:
- Sensitive areas to approach carefully:
```

### Recording setup

```python
# Standard recording configuration
RECORDING_SETTINGS = {
    'format': 'wav',           # Lossless for transcription
    'sample_rate': 44100,      # Standard quality
    'channels': 1,             # Mono is fine for speech
    'backup': True,            # Always run backup recorder
}

# File naming convention
# YYYY-MM-DD_source-lastname_topic.wav
# Example: 2024-03-15_smith_budget-hearing.wav
```

**Two-device rule**: Always record on two devices. Phone as backup minimum.

## Transcription workflows

### Automated transcription pipeline

```python
from pathlib import Path
import subprocess
import json

def transcribe_interview(audio_path: str, output_dir: str = "./transcripts") -> dict:
    """
    Transcribe using Whisper with speaker diarization.
    Returns transcript with timestamps.
    """
    Path(output_dir).mkdir(exist_ok=True)

    # Use whisper.cpp or OpenAI Whisper
    result = subprocess.run([
        'whisper',
        audio_path,
        '--model', 'medium',
        '--output_format', 'json',
        '--output_dir', output_dir,
        '--language', 'en',
        '--word_timestamps', 'True'
    ], capture_output=True)

    # Load and return structured transcript
    json_path = Path(output_dir) / f"{Path(audio_path).stem}.json"
    with open(json_path) as f:
        return json.load(f)

def format_for_editing(transcript: dict) -> str:
    """Convert to journalist-friendly format with timestamps."""
    lines = []
    for segment in transcript.get('segments', []):
        timestamp = format_timestamp(segment['start'])
        text = segment['text'].strip()
        lines.append(f"[{timestamp}] {text}")
    return '\n\n'.join(lines)

def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
```

### Speaker diarization

**Prerequisite:** this pipeline needs a one-time HuggingFace license acceptance and an `HF_TOKEN` environment variable. See the token note below.

Whisper alone does not label who is talking. On a two-source interview, a panel discussion, or any recording with more than one voice, the raw transcript comes back as one undifferentiated block. Attribution has to be done by ear or by hand. The tools table lists Otter.ai for speaker ID, which works but ships the audio to a third-party cloud. Local `whisperx` plus `pyannote.audio` labels speakers on your own machine, which matters when the source is confidential.

```python
import os
from typing import List, Dict, Optional

import whisperx

def transcribe_with_speakers(audio_path: str,
                             model_size: str = "large-v3",
                             min_speakers: Optional[int] = None,
                             max_speakers: Optional[int] = None) -> List[Dict]:
    """Return speaker-labeled segments: [{speaker, start, end, text}, ...]."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN not set. Accept the pyannote license at "
            "https://huggingface.co/pyannote/speaker-diarization-3.1 "
            "and export HF_TOKEN before running."
        )

    # pyannote on Apple Silicon MPS is unreliable at the time of writing.
    # CPU is the safe default. Set device = "cuda" on a GPU box if available.
    device = "cpu"
    compute_type = "int8"

    audio = whisperx.load_audio(audio_path)

    model = whisperx.load_model(model_size, device=device, compute_type=compute_type)
    result = model.transcribe(audio, batch_size=8)

    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    result = whisperx.align(
        result["segments"], align_model, metadata, audio, device,
        return_char_alignments=False,
    )

    diarize_pipeline = whisperx.DiarizationPipeline(
        use_auth_token=token, device=device
    )
    diarize_segments = diarize_pipeline(
        audio, min_speakers=min_speakers, max_speakers=max_speakers
    )
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # pyannote leaves segments unlabeled on crosstalk and overlap. Use .get() so
    # unlabeled segments become "UNKNOWN" instead of a KeyError on real audio.
    return [
        {
            "speaker": seg.get("speaker", "UNKNOWN"),
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
        }
        for seg in result["segments"]
    ]
```

`pyannote.audio` speaker-diarization-3.1 is gated on a license acceptance. Visit `https://huggingface.co/pyannote/speaker-diarization-3.1`, click Agree, then create a read token at `https://huggingface.co/settings/tokens` and export it as `HF_TOKEN`. Without it the pipeline raises the message above instead of a 40-line pyannote traceback.

```python
from dataclasses import dataclass

@dataclass
class DiarizedSegment:
    speaker: str
    start: float
    end: float
    text: str
```

```python
def format_diarized_transcript(segments: List[Dict],
                               speaker_map: Optional[Dict[str, str]] = None) -> str:
    """Render segments as [HH:MM:SS] **Speaker**: text."""
    speaker_map = speaker_map or {}
    lines = []
    for seg in segments:
        raw = seg.get("speaker", "UNKNOWN")
        # Unmapped labels fall through to the raw pyannote name, not KeyError.
        name = speaker_map.get(raw, raw)
        ts = format_timestamp(seg["start"])
        text = seg["text"].strip()
        lines.append(f"[{ts}] **{name}**: {text}")
    return "\n\n".join(lines)
```

### Assigning real speaker names

Raw pyannote labels look like `SPEAKER_00`, `SPEAKER_01`. Listen to the first thirty seconds, identify each voice once, and remap for the whole transcript:

```python
speaker_map = {
    "SPEAKER_00": "Reporter",
    "SPEAKER_01": "Mayor Chen",
    "SPEAKER_02": "Chief of Staff",
}
transcript = format_diarized_transcript(segments, speaker_map=speaker_map)
```

### Manual transcription template

For sensitive interviews or when AI transcription fails:

```markdown
## Transcript: [Source] - [Date]

**Recording file**: [filename]
**Duration**: [XX:XX]
**Transcribed by**: [name]
**Verified against recording**: [ ] Yes / [ ] No

---

[00:00:15] **Q**: [Your question]

[00:00:45] **A**: [Source response - verbatim, including ums, pauses noted as (...)]

[00:01:30] **Q**: [Follow-up]

[00:01:42] **A**: [Response]

---

## Notes
- [Anything not captured in audio: gestures, documents shown, etc.]

## Potential quotes
- [00:01:42] "Quote that stands out" - context: [why it matters]
```

## Quote extraction and verification

### Pull quotes workflow

```python
from dataclasses import dataclass
from typing import Optional
import re

@dataclass
class Quote:
    text: str
    timestamp: str
    speaker: str
    context: str
    verified: bool = False
    used_in: Optional[str] = None

class QuoteBank:
    """Manage quotes from interview transcripts."""

    def __init__(self):
        self.quotes = []

    def extract_quote(self, transcript: str, start_time: str,
                      end_time: str, speaker: str, context: str) -> Quote:
        """Extract and store a quote with metadata."""
        # Pull text between timestamps
        pattern = rf'\[{re.escape(start_time)}\](.+?)(?=\[\d|$)'
        match = re.search(pattern, transcript, re.DOTALL)

        if match:
            text = match.group(1).strip()
            quote = Quote(
                text=text,
                timestamp=start_time,
                speaker=speaker,
                context=context
            )
            self.quotes.append(quote)
            return quote
        return None

    def verify_quote(self, quote: Quote, audio_path: str) -> bool:
        """Mark quote as verified against original recording."""
        # In practice: listen to audio at timestamp, confirm accuracy
        quote.verified = True
        return True

    def export_for_story(self) -> str:
        """Export verified quotes ready for publication."""
        output = []
        for q in self.quotes:
            if q.verified:
                output.append(f'"{q.text}"\n— {q.speaker}\n[Timestamp: {q.timestamp}]')
        return '\n\n'.join(output)
```

### Quote accuracy checklist

Before publishing any quote:

```markdown
- [ ] Listened to original recording at timestamp
- [ ] Quote is verbatim (or clearly marked as paraphrased)
- [ ] Context preserved (not cherry-picked to change meaning)
- [ ] Speaker identified correctly
- [ ] Timestamp documented for fact-checker
- [ ] Source approved quote (if agreement made)
```

## Source management database

### Interview tracking schema

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from enum import Enum

class SourceStatus(Enum):
    ACTIVE = "active"           # Currently engaged
    DORMANT = "dormant"         # Not recently contacted
    DECLINED = "declined"       # Refused to participate
    OFF_RECORD = "off_record"   # Background only

class InterviewType(Enum):
    ON_RECORD = "on_record"
    BACKGROUND = "background"
    DEEP_BACKGROUND = "deep_background"
    OFF_RECORD = "off_record"

@dataclass
class Source:
    name: str
    organization: str
    contact_info: dict  # email, phone, signal, etc.
    beat: str
    status: SourceStatus = SourceStatus.ACTIVE
    interviews: List['Interview'] = field(default_factory=list)
    notes: str = ""

    # Relationship tracking
    first_contact: Optional[datetime] = None
    trust_level: int = 1  # 1-5 scale

@dataclass
class Interview:
    source: str
    date: datetime
    interview_type: InterviewType
    recording_path: Optional[str] = None
    transcript_path: Optional[str] = None
    story_slug: Optional[str] = None
    key_quotes: List[str] = field(default_factory=list)
    follow_up_needed: bool = False
    notes: str = ""
```

### Quick source lookup

```python
def find_sources_for_story(sources: List[Source], topic: str,
                           beat: str = None) -> List[Source]:
    """Find relevant sources for a new story."""
    matches = []
    for source in sources:
        # Filter by beat if specified
        if beat and source.beat != beat:
            continue
        # Only suggest active sources
        if source.status != SourceStatus.ACTIVE:
            continue
        # Check if they've spoken on similar topics
        for interview in source.interviews:
            if topic.lower() in interview.notes.lower():
                matches.append(source)
                break

    # Sort by trust level
    return sorted(matches, key=lambda s: s.trust_level, reverse=True)
```

## Audio/video processing

### Batch processing multiple recordings

```python
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import json

def batch_transcribe(recordings_dir: str, output_dir: str) -> dict:
    """Process all recordings in a directory."""
    recordings = list(Path(recordings_dir).glob('*.wav')) + \
                 list(Path(recordings_dir).glob('*.mp3')) + \
                 list(Path(recordings_dir).glob('*.m4a'))

    results = {}

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(transcribe_interview, str(rec), output_dir): rec
            for rec in recordings
        }

        for future in futures:
            rec = futures[future]
            try:
                transcript = future.result()
                results[rec.name] = {
                    'status': 'success',
                    'transcript': transcript
                }
            except Exception as e:
                results[rec.name] = {
                    'status': 'error',
                    'error': str(e)
                }

    return results
```

### Video interview extraction

```python
import subprocess

def extract_audio_from_video(video_path: str, output_path: str = None) -> str:
    """Extract audio track from video for transcription."""
    if output_path is None:
        output_path = video_path.rsplit('.', 1)[0] + '.wav'

    subprocess.run([
        'ffmpeg', '-i', video_path,
        '-vn',  # No video
        '-acodec', 'pcm_s16le',  # WAV format
        '-ar', '44100',  # Sample rate
        '-ac', '1',  # Mono
        output_path
    ], check=True)

    return output_path
```

## Hardware capture recommendations

Dedicated capture devices produce cleaner audio than a phone microphone and free your hands during in-person interviews. For remote and video interviews, a desktop app that joins the call captures both sides cleanly.

### Wearable AI voice recorders (in-person interviews)

| Device | Notes |
|--------|-------|
| Plaud (plaud.ai) | Clip-on with magnetic back, cloud sync, auto-transcript |
| Pocket (heypocket.com) | Wearable pendant, cloud sync, auto-transcript |

### Desktop AI meeting apps (remote and video interviews)

| App | Notes |
|-----|-------|
| Granola (granola.ai) | Mac desktop app that joins video calls, records both sides, generates notes and searchable transcript |

### Selection notes

- Run your phone as a backup recorder even with a wearable device. Cloud-sync failures are rare but ruinous when they happen.
- Two-party consent states apply to wearables the same as they apply to phones. Disclose the recorder.
- Cloud-sync devices upload audio and transcripts to vendor servers. Check the retention policy before recording anything sensitive.
- For high-sensitivity sources (whistleblowers, criminal referrals, protected identities), prefer a local-only handheld recorder over any cloud-sync device.
- Verify battery life against the full expected interview window before you sit down. Bring a spare or a wired backup for anything longer than an hour.

## Legal and ethical considerations

### Consent documentation

```markdown
## Recording consent record

**Date**:
**Source name**:
**Recording type**: [ ] Audio [ ] Video
**Interview type**: [ ] On record [ ] Background [ ] Off record

### Consent obtained:
- [ ] Verbal consent recorded at start of interview
- [ ] Written consent form signed
- [ ] Email confirmation of consent

### Jurisdiction notes:
- Interview location state/country:
- One-party or two-party consent jurisdiction:
- Any specific restrictions agreed:

### Agreed terms:
- [ ] Full attribution allowed
- [ ] Organization attribution only
- [ ] Anonymous source
- [ ] Review quotes before publication
- [ ] Embargo until [date]:
```

### Two-party consent states (US)

California, Connecticut, Florida, Illinois, Maryland, Massachusetts, Michigan, Montana, Nevada, New Hampshire, Pennsylvania, Washington require all-party consent.

**Always get explicit consent on recording** regardless of jurisdiction.

## Tools and resources

| Tool | Purpose | Notes |
|------|---------|-------|
| Whisper | Local transcription | Free, accurate, private |
| whisperx | Local transcription with diarization | Whisper + pyannote, speaker labels on your machine |
| Otter.ai | Cloud transcription | Real-time, speaker ID |
| Descript | Edit audio like text | Good for pulling clips |
| Rev | Human transcription | For sensitive/legal |
| Trint | Journalist-focused | Collaboration features |
| oTranscribe | Free web player | Manual transcription aid |
| Plaud | Wearable hardware recorder | In-person capture, cloud sync |
| Pocket | Wearable hardware recorder | In-person capture, cloud sync |
| Granola | Desktop AI meeting app | Remote and video interviews, joins the call |

## Related skills

- **source-verification** - Verify source credentials before interview
- **foia-requests** - Get documents to inform interview questions
- **data-journalism** - Analyze data sources mention in interviews

---

## Skill metadata

| Field | Value |
|-------|-------|
| Version | 1.1.0 |
| Created | 2025-12-26 |
| Author | Claude Skills for Journalism |
| Domain | Journalism, Research |
| Complexity | Intermediate |
