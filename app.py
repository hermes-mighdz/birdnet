"""
BirdNET Audio Species Classifier Plugin for Sage/Waggle

Records audio from the node microphone, a network camera, or reads
audio files, then runs BirdNET V2.4 inference (6,522 species — birds,
frogs, insects) and publishes per-species detections with confidence.

Audio sources (in priority order):
  --input FILE     Read from a local audio file
  --camera URL     Capture from a network camera via ffmpeg
                   e.g. 'http://user:pass@IP/control/faststream.jpg?stream=MxPEG&needlength'
  (default)        Record from the node's USB microphone via pywaggle

Uses eBird geo-filtering when --lat/--lon are provided to restrict
predictions to species expected at the node's location and time.

Model: BirdNET V2.4 (EfficientNetB0-like, 77 MB TFLite FP32, 0.826 GFLOPs)
Audio: 3-second chunks at 48 kHz, dual mel-spectrograms

Measurement topics (routed by soundscape-ecology category):
  env.detection.biophony.<scientific_name>    — living organisms (birds/frogs/insects)
  env.detection.anthrophony.<name>            — human-made (engine, siren, dog…)
  env.detection.geophony.<name>               — abiotic ambient (noise, environmental)
  env.detection.audio.summary                 — JSON summary/heartbeat of all detections
"""
import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from redaction import apply as redaction_apply
from redaction.yamnet_speech import RedactionFailure as YAMNetRedactionFailure
from redaction.redaction_gate import RedactionGateFailure

from save_match import parse_save_match, should_save, SaveMatchError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("birdnet-species")


# ── classifier ──────────────────────────────────────────────────────
class BirdNETClassifier:
    """Wraps the birdnet library for Sage plugin use."""

    def __init__(
        self,
        min_confidence: float = 0.25,
        sensitivity: float = 1.0,
        overlap: float = 0.0,
        top_k: int = 5,
        lat: float = -1.0,
        lon: float = -1.0,
        week: int = -1,
        sf_thresh: float = 0.03,
        bandpass_fmin: int = 0,
        bandpass_fmax: int = 15000,
        batch_size: int = 1,
    ):
        self.min_confidence = min_confidence
        self.sensitivity = sensitivity
        self.overlap = overlap
        self.top_k = top_k
        self.lat = lat
        self.lon = lon
        self.week = week
        self.sf_thresh = sf_thresh
        self.bandpass_fmin = bandpass_fmin
        self.bandpass_fmax = bandpass_fmax
        self.batch_size = batch_size
        self.model = None
        self.species_filter = None
        # Model + geo-filter construction is deferred to load() so it can be
        # timed inside the Plugin context (plugin.duration.loadmodel).

    def load(self):
        """Load the acoustic model and (if coordinates are set) build the geo
        species filter. Separated from __init__ so callers can wrap it in
        plugin.timeit('plugin.duration.loadmodel')."""
        import birdnet

        lat, lon, week = self.lat, self.lon, self.week
        # Load acoustic model (auto-downloads on first use)
        logger.info("Loading BirdNET V2.4 acoustic model...")
        self.model = birdnet.load("acoustic", "2.4", "tf")
        logger.info(
            "Acoustic model loaded (sample rate: %d Hz)",
            self.model.get_sample_rate(),
        )

        # Build species filter from geo model if coordinates are set.
        # NOTE: -1/-1 is the "unset" sentinel. We must NOT use `lon > -1` as the
        # "is set" test — real Western-Hemisphere longitudes are negative (e.g.
        # H00F is -87.98), so `lon > -1` is False for all of the Americas and the
        # geo filter would silently never build, letting the full global species
        # list through. Test against the sentinel + valid geographic ranges.
        coords_set = not (lat == -1 and lon == -1)
        coords_valid = -90 <= lat <= 90 and -180 <= lon <= 180
        if coords_set and coords_valid:
            logger.info(
                "Loading geo model for species filtering (%.4f, %.4f, week=%s)...",
                lat, lon, week if week > 0 else "all",
            )
            geo = birdnet.load("geo", "2.4", "tf")
            geo_week = week if 1 <= week <= 48 else None
            species_result = geo.predict(
                lat, lon, week=geo_week, min_confidence=self.sf_thresh,
            )
            self.species_filter = species_result.to_set()
            logger.info(
                "Geo filter: %d species expected at this location/time",
                len(self.species_filter),
            )

    def classify_file(self, audio_path: str) -> list[dict]:
        """Classify an audio file. Returns list of detection dicts."""
        predictions = self.model.predict(
            audio_path,
            top_k=self.top_k,
            overlap_duration_s=self.overlap,
            apply_sigmoid=True,
            sigmoid_sensitivity=self.sensitivity,
            default_confidence_threshold=self.min_confidence,
            custom_species_list=self.species_filter,
            bandpass_fmin=self.bandpass_fmin,
            bandpass_fmax=self.bandpass_fmax,
            batch_size=self.batch_size,
        )

        df = predictions.to_dataframe()
        if df.empty:
            return []

        detections = []
        for _, row in df.iterrows():
            species_name = row["species_name"]
            parts = species_name.split("_", 1)
            scientific = parts[0] if len(parts) > 0 else species_name
            common = parts[1] if len(parts) > 1 else ""

            detections.append({
                "scientific_name": scientific,
                "common_name": common,
                "confidence": float(row["confidence"]),
                "start_time": float(row["start_time"]),
                "end_time": float(row["end_time"]),
            })

        return detections


# ── audio sources ───────────────────────────────────────────────────
def record_from_microphone(duration_s: float, sample_rate: int = 48000) -> str:
    """Record audio from the node's USB microphone via pywaggle.

    Saved as FLAC (lossless) for the same reasons as the camera path: smaller
    than WAV, no quality loss for BirdNET, and the Sage portal inlines .flac.
    """
    from waggle.data.audio import Microphone

    mic = Microphone(samplerate=sample_rate)
    logger.info("Recording %g seconds from USB microphone at %d Hz...", duration_s, sample_rate)
    sample = mic.record(duration_s)

    # ── privacy gate: zero speech windows in place BEFORE persistence ──
    # sample.data is the raw 1-D float32 PCM array; redact_speech mutates it
    # IN PLACE and returns the same array object. The raw array must never
    # reach sample.save() below. On any redaction failure we zero the buffer
    # ourselves and STILL proceed to save (silence-only FLAC); we never fall
    # through to sample.save() with unredacted audio.
    try:
        _redacted, _events, _reason = redaction_apply.redact_speech(
            sample.data, int(sample.samplerate)
        )
        # sample.data was mutated in place; sample.save(flac_path) will now
        # write the already-zeroed array. No rebinding needed.
        if _reason is not None:
            logger.warning(
                "Speech redaction failed closed (%s) for %.2fs mic capture; "
                "entire buffer zeroed before save.", _reason, duration_s
            )
        else:
            logger.info(
                "Speech redaction applied: %d window(s) zeroed over %.2fs capture.",
                len(_events), duration_s
            )
    except (YAMNetRedactionFailure, RedactionGateFailure) as e:
        # Defensive: redact_speech is documented to swallow these and return a
        # zeroed buffer, but if its own try/except ever narrows, fail closed
        # here rather than persist raw audio.
        sample.data.fill(0.0)
        logger.error(
            "Redaction exception escaped redact_speech (%s); buffer force-zeroed "
            "before save. Raw audio was NOT persisted.", e
        )
    except Exception:
        # Unknown failure (MemoryError, LiteRT RuntimeError, ...). Zero the
        # buffer so the FLAC we are about to write is silence, not speech.
        sample.data.fill(0.0)
        logger.exception(
            "Unexpected redaction failure; buffer force-zeroed before save. "
            "Raw audio was NOT persisted."
        )

    tmpdir = tempfile.mkdtemp(prefix="birdnet_")
    flac_path = os.path.join(tmpdir, "recording.flac")
    # pywaggle's AudioSample.save() selects the container/codec from the file
    # extension via soundfile, which supports FLAC natively.
    sample.save(flac_path)
    logger.info("Audio saved to %s (FLAC)", flac_path)
    return flac_path


def record_from_camera(url: str, duration_s: float, sample_rate: int = 48000) -> str:
    """Capture audio from a network camera via ffmpeg.

    Captures to FLAC (lossless): same audio fidelity as PCM/WAV for BirdNET
    (which reads via librosa/soundfile — FLAC-capable), but ~50-70% smaller on
    disk and in Beehive, AND the Sage portal query-browser inlines an <audio>
    player only for .flac uploads (it does not recognize .wav/.mp3). FLAC is
    also a recognized archival audio format.

    Supports any ffmpeg-compatible source URL:
      - Mobotix MxPEG:  http://user:pass@IP/control/faststream.jpg?stream=MxPEG&needlength
      - RTSP:           rtsp://user:pass@IP/profile1/media.smp
      - HTTP streams:   http://IP/audio.cgi
    """
    tmpdir = tempfile.mkdtemp(prefix="birdnet_")
    flac_path = os.path.join(tmpdir, "camera_audio.flac")

    # Detect Mobotix MxPEG streams — need -f mxg input format
    input_args = []
    if "faststream" in url and "MxPEG" in url:
        input_args = ["-f", "mxg"]
    elif url.startswith("rtsp://"):
        input_args = ["-rtsp_transport", "tcp"]

    cmd = (
        ["ffmpeg", "-y"]
        + input_args
        + ["-i", url,
           "-vn",                     # no video
           "-acodec", "flac",         # lossless FLAC (portal inlines .flac)
           "-ar", str(sample_rate),   # resample to target rate
           "-ac", "1",               # mono
           "-t", str(duration_s),
           flac_path]
    )

    # Log the command without credentials
    safe_url = url.split("@")[-1] if "@" in url else url
    logger.info("Capturing %g seconds from camera %s...", duration_s, safe_url)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=int(duration_s) + 30,
    )

    if result.returncode != 0:
        logger.error("ffmpeg failed (exit %d): %s", result.returncode, result.stderr[-300:])
        raise RuntimeError(f"ffmpeg failed to capture audio from camera: {result.stderr[-200:]}")

    if not os.path.exists(flac_path) or os.path.getsize(flac_path) < 1000:
        raise RuntimeError("ffmpeg produced no audio output — check camera URL and credentials")

    size = os.path.getsize(flac_path)
    logger.info("Camera audio saved to %s (%d bytes, FLAC)", flac_path, size)
    return flac_path


# ── publishing ──────────────────────────────────────────────────────
# BirdNET V2.4's label set is NOT birds-only: alongside real taxa (birds, frogs,
# insects, a few mammals) it carries a small, fixed set of human-made and abiotic
# "distractor" classes (Engine, Siren, Noise, …). We route detections to three
# standard soundscape-ecology topics so consumers can separate biological signal
# from anthropogenic/geophysical sound:
#   biophony    — living organisms (the science signal): birds, frogs, insects…
#   anthrophony — human-made: engines, sirens, guns, power tools, fireworks, dogs
#   geophony    — non-biological ambient: undifferentiated noise, environmental
# The non-biophony classes are enumerated below (in the label file their
# scientific == common name). Anything NOT listed is treated as biophony, so new
# species in future model releases default to the biological bucket automatically.
ANTHROPHONY_CLASSES = {
    "Engine", "Siren", "Gun", "Fireworks", "Power tools", "Dog",
    "Human vocal", "Human non-vocal", "Human whistle",
}
GEOPHONY_CLASSES = {"Noise", "Environmental"}


def sound_category(scientific_name: str) -> str:
    """Map a BirdNET class to its soundscape-ecology category.

    Returns 'anthrophony' (human-made), 'geophony' (abiotic ambient), or
    'biophony' (any living organism — the default for real taxa).
    """
    if scientific_name in ANTHROPHONY_CLASSES:
        return "anthrophony"
    if scientific_name in GEOPHONY_CLASSES:
        return "geophony"
    return "biophony"


def publish_detections(plugin, detections: list[dict], timestamp: int):
    """Publish detections to Waggle, routed by soundscape-ecology category.

    Each detection publishes to env.detection.<category>.<name> where category is
    biophony / anthrophony / geophony (see sound_category). Per-detection topics
    appear only when a detection is present, but the summary topic is ALWAYS
    published — even with zero detections — so the data API carries a per-cycle
    heartbeat that proves the job ran. This lets us distinguish "running fine, no
    birds" from "job is dead" via the data API.
    """
    for det in detections:
        category = sound_category(det["scientific_name"])
        topic_name = det["scientific_name"].lower().replace(" ", "_")
        plugin.publish(
            f"env.detection.{category}.{topic_name}",
            det["confidence"],
            timestamp=timestamp,
            # pywaggle requires meta values to be strings — stringify the floats.
            meta={
                "common_name": str(det["common_name"]),
                "category": category,
                "start_time_s": str(det["start_time"]),
                "end_time_s": str(det["end_time"]),
            },
        )

    # Always publish a summary (heartbeat). The per-detection loop above already
    # does nothing when `detections` is empty, so total_detections == 0 is the
    # quiet-cycle signal. The summary carries a per-category breakdown so a
    # consumer can read biophony vs anthrophony vs geophony counts at a glance.
    species_best = {}
    cat_counts = {"biophony": 0, "anthrophony": 0, "geophony": 0}
    for det in detections:
        cat_counts[sound_category(det["scientific_name"])] += 1
        key = det["scientific_name"]
        if key not in species_best or det["confidence"] > species_best[key]["confidence"]:
            species_best[key] = det

    summary = {
        "total_detections": len(detections),
        "unique_species": len(species_best),
        "biophony": cat_counts["biophony"],
        "anthrophony": cat_counts["anthrophony"],
        "geophony": cat_counts["geophony"],
        "species": [
            {
                "scientific_name": d["scientific_name"],
                "common_name": d["common_name"],
                "confidence": round(d["confidence"], 4),
                "category": sound_category(d["scientific_name"]),
            }
            for d in sorted(species_best.values(), key=lambda x: x["confidence"], reverse=True)
        ],
    }
    plugin.publish(
        "env.detection.audio.summary",
        json.dumps(summary),
        timestamp=timestamp,
    )


# ── utilities ───────────────────────────────────────────────────────
def current_birdnet_week() -> int:
    """Calculate the current BirdNET week (1–48, 4 weeks per month)."""
    now = datetime.now()
    # BirdNET uses 4 weeks per month: week = (month - 1) * 4 + ceil(day / 7.5)
    # This maps Jan 1 = week 1, Dec 31 = week 48
    import math
    week = (now.month - 1) * 4 + min(4, math.ceil(now.day / 7.5))
    return max(1, min(48, week))


# Node manifest locations to probe, in order. The platform maintains the
# manifest with the node's current GPS; SES may mount it at the canonical
# host path, or a job may mount it elsewhere. We try several known spots so
# geo-filtering "just works" without hardcoding coordinates per node.
MANIFEST_PATHS = [
    os.environ.get("WAGGLE_NODE_MANIFEST", ""),   # explicit override (mount target)
    "/etc/waggle/node-manifest-v2.json",          # canonical host path
    "/run/waggle/node-manifest-v2.json",          # alt runtime path
    "/host/etc/waggle/node-manifest-v2.json",     # host rootfs mount convention
]


def _coords_from_manifest() -> tuple[float, float] | None:
    """Try each known manifest path; return (lat, lon) from the first valid one."""
    for path in MANIFEST_PATHS:
        if not path:
            continue
        try:
            with open(path) as f:
                manifest = json.load(f)
            lat = manifest.get("gps_lat")
            lon = manifest.get("gps_lon")
            if lat is not None and lon is not None:
                logger.info("Node location from manifest %s", path)
                return float(lat), float(lon)
        except (FileNotFoundError, json.JSONDecodeError, ValueError, KeyError):
            continue
    return None


def _coords_from_env() -> tuple[float, float] | None:
    """Try Waggle-injected GPS env vars (some SES deployments set these)."""
    lat = os.environ.get("WAGGLE_NODE_GPS_LAT") or os.environ.get("WAGGLE_GPS_LAT")
    lon = os.environ.get("WAGGLE_NODE_GPS_LON") or os.environ.get("WAGGLE_GPS_LON")
    if lat and lon:
        try:
            logger.info("Node location from Waggle env vars")
            return float(lat), float(lon)
        except ValueError:
            return None
    return None


def _coords_from_live_gps(timeout_s: float = 3.0) -> tuple[float, float] | None:
    """Try a live GPS fix by subscribing to the node's ``sys.gps.*`` stream.

    NOTE: pywaggle has no dedicated GPS/location accessor (no ``waggle.data.gps``
    and no ``Plugin.get_location()`` as of pywaggle 0.56). The only live-GPS
    mechanism today is the data plane: GPS-equipped nodes run a device plugin
    that publishes ``sys.gps.lat`` / ``sys.gps.lon`` measurements, which other
    plugins can ``subscribe`` to. Fixed nodes (e.g. H00F) have no such publisher,
    so this returns None quickly and we fall back to the manifest.

    Best-effort: any failure (no Plugin scope, no publisher, timeout) returns
    None. This should ideally become a first-class pywaggle feature — see the
    module docstring / project notes.
    """
    try:
        from waggle.plugin import Plugin
    except Exception:
        return None

    lat = lon = None
    try:
        with Plugin() as plugin:
            plugin.subscribe("sys.gps.lat", "sys.gps.lon")
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline and (lat is None or lon is None):
                remaining = deadline - time.monotonic()
                try:
                    msg = plugin.get(timeout=max(0.1, remaining))
                except Exception:
                    break  # no message within the window
                if msg.name == "sys.gps.lat":
                    lat = float(msg.value)
                elif msg.name == "sys.gps.lon":
                    lon = float(msg.value)
        if lat is not None and lon is not None:
            logger.info("Node location from live sys.gps.* stream")
            return lat, lon
    except Exception as e:
        logger.debug("Live GPS subscribe unavailable: %s", e)
    return None


def read_node_location(try_live_gps: bool = False) -> tuple[float, float] | None:
    """Resolve the node's GPS location dynamically, trying sources in order:

      1. Live ``sys.gps.*`` stream  (only if try_live_gps=True; GPS-equipped/
                                     mobile nodes; absent on fixed nodes)
      2. Node manifest file          (platform-maintained; canonical on Sage)
      3. Waggle-injected env vars    (if a deployment sets them)

    Returns (lat, lon) or None if no source is available. Explicit --lat/--lon
    on the CLI override this entirely (handled by the caller).

    pywaggle currently exposes none of these as a tidy "get my location" call;
    this resolver stitches together the mechanisms that DO exist. The proper
    fix is an upstream pywaggle location accessor + WES injecting node GPS into
    the plugin environment. Live-GPS is opt-in (try_live_gps) because on a fixed
    node with no GPS publisher the subscribe just wastes a few seconds.
    """
    sources = [_coords_from_manifest, _coords_from_env]
    if try_live_gps:
        sources.insert(0, _coords_from_live_gps)
    for source in sources:
        coords = source()
        if coords is not None:
            return coords
    return None


# ── CLI ─────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BirdNET V2.4 audio species classifier for Sage/Waggle",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Audio input
    audio = parser.add_argument_group("audio input")
    audio.add_argument(
        "--input", "-i",
        help="Path to audio file or directory. If not specified, records from microphone or camera.",
    )
    audio.add_argument(
        "--camera",
        help="URL for network camera audio. Supports Mobotix MxPEG, RTSP, or any ffmpeg source. "
             "Example: 'http://user:pass@IP/control/faststream.jpg?stream=MxPEG&needlength'",
    )
    audio.add_argument(
        "--duration", type=float, default=15.0,
        help="Recording duration in seconds (microphone or camera mode).",
    )
    audio.add_argument(
        "--sample-rate", type=int, default=48000,
        help="Audio sample rate in Hz.",
    )

    # Model parameters
    model = parser.add_argument_group("model parameters")
    model.add_argument(
        "--min-confidence", type=float, default=0.25,
        help="Minimum confidence threshold (0.01–0.99).",
    )
    model.add_argument(
        "--sensitivity", type=float, default=1.0,
        help="Detection sensitivity (0.5–1.5). Higher = more sensitive.",
    )
    model.add_argument(
        "--overlap", type=float, default=0.0,
        help="Overlap in seconds between 3-second analysis windows (0.0–2.9).",
    )
    model.add_argument(
        "--top-k", type=int, default=5,
        help="Max predictions per 3-second chunk.",
    )
    model.add_argument(
        "--bandpass-fmin", type=int, default=0,
        help="Bandpass filter minimum frequency in Hz. Useful to cut low-frequency noise.",
    )
    model.add_argument(
        "--bandpass-fmax", type=int, default=15000,
        help="Bandpass filter maximum frequency in Hz. Set to match audio source "
             "(e.g. 4000 for 8kHz camera mic, 15000 for full-bandwidth USB mic).",
    )
    model.add_argument(
        "--batch-size", type=int, default=1,
        help="Number of 3-second chunks to process in parallel. Increase for long recordings.",
    )

    # Location filtering
    loc = parser.add_argument_group("location filtering (eBird)")
    loc.add_argument(
        "--lat", type=float, default=-1,
        help="Latitude for species range filtering. -1 = auto-resolve "
             "(manifest/env; or live sys.gps.* with --gps-subscribe).",
    )
    loc.add_argument(
        "--lon", type=float, default=-1,
        help="Longitude for species range filtering. -1 = auto-resolve "
             "(manifest/env; or live sys.gps.* with --gps-subscribe).",
    )
    loc.add_argument(
        "--gps-subscribe", action="store_true",
        help="When auto-resolving location, also try a live GPS fix by "
             "subscribing to the node's sys.gps.* stream (only useful on "
             "GPS-equipped/mobile nodes; adds a few seconds of startup and is "
             "off by default since fixed nodes have no GPS publisher).",
    )
    loc.add_argument(
        "--week", type=str, default="auto",
        help="Week of year (1–48) for seasonal filtering. "
             "'auto' (default) = calculate from current date. -1 for year-round.",
    )
    loc.add_argument(
        "--sf-thresh", type=float, default=0.03,
        help="Species filter threshold for geo model (0.0–1.0).",
    )

    # Runtime
    runtime = parser.add_argument_group("runtime")
    runtime.add_argument(
        "--interval", type=float, default=0.0,
        help="Seconds between recording cycles. 0 = run once (or --num-recordings times).",
    )
    runtime.add_argument(
        "--num-recordings", type=int, default=1,
        help="Number of recording cycles to run. 0 = loop forever (requires --interval > 0).",
    )
    runtime.add_argument(
        "--output", "-o",
        help="Path to save CSV results (optional).",
    )
    runtime.add_argument(
        "--save-match", type=str, default="",
        help="When to SAVE (upload) the recorded audio clip. Comma-separated "
             "OR-list of 'Name:confidence' rules, e.g. "
             "\"Northern Cardinal:0.5,Barn Owl:0.4\". Name is matched "
             "case-insensitively and EXACTLY against the common OR scientific "
             "name. Use \"*:0.5\" to save any clip with a detection >=0.5. The "
             "clip is saved if ANY detection matches ANY rule. Operates only on "
             "published detections (>= --min-confidence). Omit to save no audio "
             "(detection topics + heartbeat still publish).",
    )
    runtime.add_argument(
        "--dry-run", action="store_true",
        help="Run without publishing to Waggle (for testing).",
    )

    return parser


def _get_audio(args) -> tuple[str, bool]:
    """Get audio from the configured source. Returns (path, needs_cleanup)."""
    if args.input:
        return args.input, False
    elif args.camera:
        return record_from_camera(args.camera, args.duration, args.sample_rate), True
    else:
        return record_from_microphone(args.duration, args.sample_rate), True


def _log_detections(detections: list[dict]):
    """Log detections to console."""
    for det in detections:
        logger.info(
            "  %s (%s): %.4f [%.1f-%.1fs]",
            det["scientific_name"],
            det["common_name"],
            det["confidence"],
            det["start_time"],
            det["end_time"],
        )


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Clamp parameters to valid ranges
    args.min_confidence = max(0.01, min(args.min_confidence, 0.99))
    args.sensitivity = max(0.5, min(args.sensitivity, 1.5))
    args.overlap = max(0.0, min(args.overlap, 2.9))

    # Parse --save-match up front and FAIL FAST on a malformed spec: a typo'd
    # save rule that silently saved nothing would waste an entire deployment.
    try:
        save_rules = parse_save_match(args.save_match)
    except SaveMatchError as e:
        logger.error("Invalid --save-match: %s", e)
        sys.exit(2)
    if save_rules:
        logger.info("Audio save rules (--save-match): %s",
                    ", ".join(f"{'*' if r.is_wildcard else r.name}>={r.min_confidence}"
                              for r in save_rules))
    else:
        logger.info("No --save-match rules: audio clips will NOT be saved "
                    "(detection topics + heartbeat still publish every cycle).")

    # Resolve --week: "auto" → current week, or parse as int
    if args.week.lower() == "auto":
        args.week = current_birdnet_week()
        logger.info("Auto-detected BirdNET week: %d", args.week)
    else:
        args.week = int(args.week)

    # Resolve --lat/--lon: auto-resolve from node location if not specified
    if args.lat == -1 and args.lon == -1:
        location = read_node_location(try_live_gps=args.gps_subscribe)
        if location is not None:
            args.lat, args.lon = location
            logger.info("Auto-detected node location: (%.4f, %.4f)", args.lat, args.lon)
        else:
            logger.info("No node location available (live GPS / manifest / env "
                        "all absent) — geo-filtering disabled. Pass --lat/--lon "
                        "to enable it explicitly, or mount the node manifest.")

    logger.info("BirdNET Species Classifier starting")
    logger.info(
        "  min_confidence=%.2f  sensitivity=%.1f  overlap=%.1f  top_k=%d",
        args.min_confidence, args.sensitivity, args.overlap, args.top_k,
    )
    if args.input:
        logger.info("  source=file (%s)", args.input)
    elif args.camera:
        safe_url = args.camera.split("@")[-1] if "@" in args.camera else args.camera
        logger.info("  source=camera (%s)", safe_url)
    else:
        logger.info("  source=microphone (USB)")
    if not (args.lat == -1 and args.lon == -1):
        logger.info("  location=(%.4f, %.4f)  week=%d", args.lat, args.lon,
                     args.week if args.week > 0 else -1)

    # Initialize classifier
    classifier = BirdNETClassifier(
        min_confidence=args.min_confidence,
        sensitivity=args.sensitivity,
        overlap=args.overlap,
        top_k=args.top_k,
        lat=args.lat,
        lon=args.lon,
        week=args.week,
        sf_thresh=args.sf_thresh,
        bandpass_fmin=args.bandpass_fmin,
        bandpass_fmax=args.bandpass_fmax,
        batch_size=args.batch_size,
    )

    if args.dry_run:
        logger.info("DRY RUN — will not publish to Waggle")

    def run_cycle(plugin=None):
        """Single record-classify-publish cycle."""
        timestamp = int(time.time_ns())

        # Acquire audio input, timed as plugin.duration.input (nanoseconds) —
        # the standard Sage phase metric (see avian-diversity-monitoring/TAFT).
        if plugin is not None:
            with plugin.timeit("plugin.duration.input"):
                audio_path, cleanup = _get_audio(args)
        else:
            audio_path, cleanup = _get_audio(args)

        try:
            t0 = time.time()
            # Run inference, timed as plugin.duration.inference (nanoseconds).
            if plugin is not None:
                with plugin.timeit("plugin.duration.inference"):
                    detections = classifier.classify_file(audio_path)
            else:
                detections = classifier.classify_file(audio_path)
            elapsed = time.time() - t0

            logger.info(
                "Classified %s: %d detections in %.2fs",
                os.path.basename(audio_path), len(detections), elapsed,
            )

            # Heartbeat invariant: ALWAYS publish (publish_detections emits the
            # summary even with zero detections), so the data API carries a
            # per-cycle liveness signal. Previously this call was gated behind
            # `if detections:`, so quiet cycles published NOTHING — making a live
            # job indistinguishable from a dead one.
            if plugin is not None:
                publish_detections(plugin, detections, timestamp)
            if detections:
                _log_detections(detections)
            else:
                logger.info("  No detections above threshold %.2f", args.min_confidence)

            # SAVE (selective): upload the captured audio clip only when a
            # detection matches a --save-match rule. ANY (rule x detection)
            # match saves the whole clip once. No rules => never saves.
            if plugin is not None and save_rules and should_save(
                save_rules, detections,
                name_keys=["common_name", "scientific_name"],
            ):
                try:
                    top = max(detections, key=lambda d: d["confidence"])
                    plugin.upload_file(
                        audio_path, timestamp=timestamp,
                        meta={
                            "top_species": str(top["scientific_name"]),
                            "common_name": str(top["common_name"]),
                            "confidence": str(top["confidence"]),
                        },
                    )
                    logger.info("Saved audio clip (save-match matched: %s %.4f)",
                                top["scientific_name"], top["confidence"])
                except Exception:
                    logger.exception("Audio clip upload failed")

            if args.output and detections:
                _save_csv(detections, args.output, audio_path)

            return detections

        finally:
            if cleanup and os.path.exists(audio_path):
                shutil.rmtree(os.path.dirname(audio_path), ignore_errors=True)

    def run_loop(plugin=None):
        """Run recording cycles based on --num-recordings and --interval.

        --num-recordings 1  --interval 0    Run once and exit (default)
        --num-recordings 6  --interval 5    Run 6 cycles, 5s gap between each
        --num-recordings 0  --interval 60   Loop forever, 60s between cycles
        """
        num = args.num_recordings
        if num == 0 and args.interval <= 0:
            logger.error("--num-recordings 0 (loop forever) requires --interval > 0")
            sys.exit(1)

        cycle = 0
        while True:
            cycle += 1
            if num > 1 or num == 0:
                logger.info("── Cycle %d%s ──", cycle,
                            f"/{num}" if num > 0 else "")
            run_cycle(plugin)

            # Check if we've done enough
            if num > 0 and cycle >= num:
                break

            # Sleep between cycles
            if args.interval > 0:
                logger.info("Sleeping %.1fs...", args.interval)
                time.sleep(args.interval)
            elif num > 1:
                # Multiple recordings with no explicit interval — no gap
                pass

    try:
        if args.dry_run:
            classifier.load()  # no Plugin context in dry-run; load untimed
            run_loop(plugin=None)
        else:
            from waggle.plugin import Plugin
            with Plugin() as plugin:
                # Load model + geo filter, timed as plugin.duration.loadmodel
                # (nanoseconds) — standard Sage telemetry; makes cold-start cost
                # observable for window sizing.
                with plugin.timeit("plugin.duration.loadmodel"):
                    classifier.load()
                run_loop(plugin=plugin)
    except KeyboardInterrupt:
        logger.info("Interrupted — shutting down")


def _save_csv(detections: list[dict], output_path: str, audio_path: str):
    """Append detections to a CSV file."""
    write_header = not os.path.exists(output_path)
    with open(output_path, "a") as f:
        if write_header:
            f.write("audio_file,start_time,end_time,scientific_name,common_name,confidence\n")
        for det in detections:
            f.write(
                f"{audio_path},{det['start_time']},{det['end_time']},"
                f"{det['scientific_name']},{det['common_name']},{det['confidence']:.4f}\n"
            )
    logger.info("Results appended to %s", output_path)


if __name__ == "__main__":
    main()
