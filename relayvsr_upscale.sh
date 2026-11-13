#!/usr/bin/env bash
#
# upscale.sh - 4x upscale one video with RelayVSR.
#
#   upscale.sh [options] INPUT [-- extra arguments for inference.py]
#
# Stages. On a re-run every stage skips what is already on disk, so an
# interrupted job continues from the chunk it stopped in.
#
#   1. cut      INPUT -> lossless chunks                      work/<name>/lr/
#   2. upscale  each chunk through RelayVSR's inference.py    work/<name>/sr/
#   3. stitch   join the chunks, add the original audio       OUTPUT
#   4. verify   OUTPUT has exactly as many frames as went in
#
# Functions, in the order main() calls them:
#
#   parse_args        read the command line
#   check_setup       the install is there, INPUT exists, OUTPUT does not
#   probe_source      frame rate, colour matrix and frame size of INPUT
#   prepare_workdir   work/<name>/, refuse to mix two different jobs
#   select_gpu        pin the run to one GPU when --gpu is given
#   cut_source        stage 1
#   list_chunks       collect the chunk files, print the plan
#   upscale_chunks    stage 2
#   stitch_chunks     stage 3   (uses run_stitch)
#   verify_output     stage 4
#   cleanup           remove the work directory
#
# Helpers: die, usage, video_field, count_frames, run_stitch

set -euo pipefail

# ---------------------------------------------------------------- settings --

RVSR_HOME="${RVSR_HOME:-/mnt/scratch/rvsr}"

# Frames per chunk. RelayVSR itself has no frame limit (it streams with
# bounded memory); chunks only exist so a long job can resume. Each chunk
# starts the model's temporal context from scratch, so fewer, larger chunks
# are better. 0 = never split.
CHUNK_FRAMES=5000

# ------------------------------------------------------------ derived paths --

REPO_DIR="$RVSR_HOME/RelayVSR"
CACHE_DIR="$RVSR_HOME/cache"
WEIGHTS_DIR="$REPO_DIR/weights"
PYTHON="$REPO_DIR/.venv/bin/python"

# Runtime caches stay inside the tree (compiled Triton / torch.compile kernels).
export TRITON_CACHE_DIR="$CACHE_DIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/torchinductor"
# Nothing here needs the network; make sure nothing tries.
export HF_HOME="$CACHE_DIR/hf"
export HF_HUB_OFFLINE=1

# ------------------------------------------------- set by the functions below --

INPUT=""
OUTPUT=""
GPU=""
KEEP_WORK=0
FRESH=0
EXTRA_ARGS=()

SRC_FPS=""
CUT_FILTER=()
CUT_CODEC=()
WORK_DIR=""
LR_DIR=""
SR_DIR=""
LR_CHUNKS=()
TOTAL_FRAMES=0

# ------------------------------------------------------------------ helpers --

die() {
    echo "ERROR: $*" >&2
    exit 1
}

usage() {
    cat <<EOF
usage: upscale.sh [options] INPUT [-- extra arguments for inference.py]

  -o, --output FILE   result file (default: INPUT name with _x4.mp4, next to INPUT)
  -c, --chunk N       frames per chunk, 0 = one pass, no splitting (default: $CHUNK_FRAMES)
  -g, --gpu N         GPU to use, numbered as in nvidia-smi (default: torch's first GPU)
  -k, --keep          keep the work directory after a successful run
      --fresh         throw away earlier work for this input and start over
  -h, --help          this text

Everything after "--" is passed to RelayVSR's inference.py unchanged, e.g.
  upscale.sh clip.mp4 -- --window-size 24 --dtype fp16
EOF
}

# video_field NAME FILE : one property of the first video stream.
video_field() {
    ffprobe -v error -select_streams v:0 -show_entries "stream=$1" \
        -of default=noprint_wrappers=1:nokey=1 "$2" | sed -n '1p'
}

# count_frames FILE : number of video frames (counts packets, no decoding).
count_frames() {
    ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets \
        -of default=noprint_wrappers=1:nokey=1 "$1" | sed -n '1p'
}

# ------------------------------------------------------------------- stages --

parse_args() {
    while [ $# -gt 0 ]; do
        case "$1" in
            -o|--output) OUTPUT="${2:?$1 needs a value}"; shift 2 ;;
            -c|--chunk)  CHUNK_FRAMES="${2:?$1 needs a value}"; shift 2 ;;
            -g|--gpu)    GPU="${2:?$1 needs a value}"; shift 2 ;;
            -k|--keep)   KEEP_WORK=1; shift ;;
            --fresh)     FRESH=1; shift ;;
            -h|--help)   usage; exit 0 ;;
            --)          shift; EXTRA_ARGS=("$@"); break ;;
            -*)          die "unknown option: $1 (try --help)" ;;
            *)
                if [ -n "$INPUT" ]; then
                    die "only one input file per run (got '$INPUT' and '$1')"
                fi
                INPUT="$1"
                shift
                ;;
        esac
    done

    if [ -z "$INPUT" ]; then
        usage
        exit 1
    fi
    if ! [[ "$CHUNK_FRAMES" =~ ^[0-9]+$ ]]; then
        die "--chunk must be a whole number, got '$CHUNK_FRAMES'"
    fi
}

check_setup() {
    local cmd

    [ -x "$PYTHON" ] || die "no Python environment at $PYTHON - run install.sh first"
    [ -f "$REPO_DIR/inference.py" ] || die "RelayVSR not found in $REPO_DIR - run install.sh first"
    [ -f "$WEIGHTS_DIR/wan.safetensors" ] || die "missing $WEIGHTS_DIR/wan.safetensors - run install.sh first"
    [ -f "$WEIGHTS_DIR/flash.safetensors" ] || die "missing $WEIGHTS_DIR/flash.safetensors - run install.sh first"
    for cmd in ffmpeg ffprobe; do
        command -v "$cmd" >/dev/null || die "$cmd is not on PATH"
    done

    [ -f "$INPUT" ] || die "input not found: $INPUT"
    INPUT="$(realpath "$INPUT")"

    if [ -z "$OUTPUT" ]; then
        OUTPUT="${INPUT%.*}_x4.mp4"
    fi
    OUTPUT="$(realpath -m "$OUTPUT")"
    case "$OUTPUT" in
        *.mp4|*.mkv|*.mov) ;;
        *) die "output name must end in .mp4, .mkv or .mov: $OUTPUT" ;;
    esac
    [ "$OUTPUT" != "$INPUT" ] || die "output and input are the same file"
    [ ! -e "$OUTPUT" ] || die "output already exists: $OUTPUT (remove it or pick another name with -o)"
    mkdir -p "$(dirname "$OUTPUT")"
}

probe_source() {
    local matrix width height

    # Frame rate: the average rate, which is what inference.py would use by
    # itself. It is passed on explicitly so that every chunk gets the same one.
    SRC_FPS="$(video_field avg_frame_rate "$INPUT")"
    if [ -z "$SRC_FPS" ] || [ "$SRC_FPS" = "0/0" ]; then
        SRC_FPS="$(video_field r_frame_rate "$INPUT")"
    fi
    if [ -z "$SRC_FPS" ] || [ "$SRC_FPS" = "0/0" ]; then
        die "could not read a frame rate from $INPUT"
    fi

    # Colour matrix. RelayVSR's reader (PyAV) converts YUV to RGB with the
    # matrix the file is tagged with, and assumes BT.601 when there is no tag.
    # Players assume BT.709 for untagged HD. So an untagged source gets the
    # tag a player would assume, otherwise the model sees shifted colours.
    # setparams only labels the frames, it does not change any pixel.
    width="$(video_field width "$INPUT")"
    height="$(video_field height "$INPUT")"
    matrix="$(video_field color_space "$INPUT")"
    case "$matrix" in
        ""|unknown|reserved)
            if [ "$width" -ge 1280 ] || [ "$height" -gt 576 ]; then
                matrix="bt709"
            else
                matrix="smpte170m"
            fi
            CUT_FILTER=(-vf "setparams=colorspace=$matrix")
            echo "source: no colour matrix tag, treating it as $matrix"
            ;;
    esac

    # Lossless codec for the chunks, so the model sees the pixels of INPUT.
    # H.264 at -qp 0 is exact, small and fast, but 4:2:0 H.264 cannot store an
    # odd width or height (427x240 and the like). Those sizes go to FFV1.
    if [ $((width % 2)) = 1 ] || [ $((height % 2)) = 1 ]; then
        CUT_CODEC=(-c:v ffv1 -level 3)
    else
        CUT_CODEC=(-c:v libx264 -preset ultrafast -qp 0 -bf 0 -sc_threshold 0)
    fi
}

prepare_workdir() {
    local name job

    name="$(basename "${INPUT%.*}")"
    name="${name//[^A-Za-z0-9._-]/_}"
    if [ -z "$name" ]; then
        name="job"
    fi
    WORK_DIR="$RVSR_HOME/work/$name"
    LR_DIR="$WORK_DIR/lr"
    SR_DIR="$WORK_DIR/sr"

    if [ "$FRESH" = 1 ]; then
        rm -rf "$WORK_DIR"
    fi

    # Chunks from a different file, chunk size or model setting must never be
    # stitched together. job.txt records what this work directory belongs to.
    job="input=$INPUT size=$(stat -c %s "$INPUT") mtime=$(stat -c %Y "$INPUT") chunk=$CHUNK_FRAMES args=${EXTRA_ARGS[*]}"
    if [ -f "$WORK_DIR/job.txt" ] && [ "$(cat "$WORK_DIR/job.txt")" != "$job" ]; then
        echo "work directory $WORK_DIR was made by a different job:"
        echo "  then: $(cat "$WORK_DIR/job.txt")"
        echo "  now:  $job"
        die "re-run with the same settings to resume it, or add --fresh to start over"
    fi

    mkdir -p "$LR_DIR" "$SR_DIR"
    echo "$job" > "$WORK_DIR/job.txt"
}

select_gpu() {
    if [ -n "$GPU" ]; then
        # PCI_BUS_ID makes CUDA number the cards the way nvidia-smi does.
        export CUDA_DEVICE_ORDER=PCI_BUS_ID
        export CUDA_VISIBLE_DEVICES="$GPU"
        echo "gpu: $GPU"
    fi
}

cut_source() {
    local gop="$CHUNK_FRAMES"

    if [ -f "$LR_DIR/.done" ]; then
        echo "cut: already done"
        return
    fi
    rm -f "$LR_DIR"/chunk_*.mkv

    if [ "$gop" = 0 ]; then
        gop=1000000000    # one keyframe, therefore one chunk
    fi

    # Constant frame rate (-fps_mode cfr -r): RelayVSR writes its output at one
    # fixed rate, so the frames have to be evenly spaced before they go in or
    # the original audio drifts against the result. A constant-rate source
    # passes through frame for frame. A variable-rate source (phone footage,
    # screen recordings) gets frames repeated or dropped to stay on time.
    #
    # Chunk length: the encoder puts a keyframe exactly every $gop frames and
    # nowhere else (-g), and the segment muxer starts a new file at every
    # keyframe (-segment_time 0.001). That gives chunks of exactly $gop frames.
    echo "cut: writing lossless chunks to $LR_DIR"
    ffmpeg -hide_banner -loglevel warning -stats -nostdin -y \
        -i "$INPUT" -map 0:v:0 -an -sn -dn "${CUT_FILTER[@]}" \
        -fps_mode cfr -r "$SRC_FPS" \
        "${CUT_CODEC[@]}" -g "$gop" \
        -f segment -segment_time 0.001 -reset_timestamps 1 \
        "$LR_DIR/chunk_%05d.mkv"

    touch "$LR_DIR/.done"
}

list_chunks() {
    local chunk width height

    LR_CHUNKS=("$LR_DIR"/chunk_*.mkv)
    [ -f "${LR_CHUNKS[0]}" ] || die "cutting produced no chunks in $LR_DIR"

    TOTAL_FRAMES=0
    for chunk in "${LR_CHUNKS[@]}"; do
        TOTAL_FRAMES=$((TOTAL_FRAMES + $(count_frames "$chunk")))
    done

    width="$(video_field width "${LR_CHUNKS[0]}")"
    height="$(video_field height "${LR_CHUNKS[0]}")"
    echo "plan: $TOTAL_FRAMES frames at $SRC_FPS fps in ${#LR_CHUNKS[@]} chunk(s)"
    echo "plan: ${width}x${height} -> $((width * 4))x$((height * 4))"
    echo "plan: result goes to $OUTPUT"
}

upscale_chunks() {
    local lr sr name
    local index=0

    # inference.py writes to a temporary file and renames it when the chunk is
    # complete, so an existing chunk_*.mp4 is always a finished one. Leftover
    # temporaries come from a run that was killed.
    rm -f "$SR_DIR"/.refvsr-*.mp4

    for lr in "${LR_CHUNKS[@]}"; do
        index=$((index + 1))
        name="$(basename "${lr%.mkv}")"
        sr="$SR_DIR/$name.mp4"

        if [ -f "$sr" ]; then
            echo "upscale: $index/${#LR_CHUNKS[@]} $name already done"
            continue
        fi

        echo "upscale: $index/${#LR_CHUNKS[@]} $name ($(count_frames "$lr") frames)"
        "$PYTHON" "$REPO_DIR/inference.py" \
            --input "$lr" \
            --output "$sr" \
            --fps "$SRC_FPS" \
            "${EXTRA_ARGS[@]}"
    done
}

# run_stitch OUTFILE AUDIO_ARGS... : join the chunks and add INPUT's audio.
run_stitch() {
    local out="$1"
    shift

    # -c:v copy: the chunks are joined without re-encoding.
    # h264_metadata: RelayVSR's writer encodes BT.601 but leaves the file
    # untagged, and players then decode HD as BT.709 (visibly wrong reds and
    # greens). This writes the correct tag into the stream; no pixels change.
    # "1:a?" takes every audio stream of INPUT, and is fine with none.
    ffmpeg -hide_banner -loglevel warning -nostdin -y \
        -f concat -safe 0 -i "$WORK_DIR/concat.txt" \
        -i "$INPUT" \
        -map 0:v:0 -map "1:a?" \
        -c:v copy \
        -bsf:v "h264_metadata=matrix_coefficients=6:video_full_range_flag=0" \
        "$@" \
        "$out"
}

stitch_chunks() {
    local lr sr
    local partial="${OUTPUT%.*}.partial.${OUTPUT##*.}"

    : > "$WORK_DIR/concat.txt"
    for lr in "${LR_CHUNKS[@]}"; do
        sr="$SR_DIR/$(basename "${lr%.mkv}").mp4"
        [ -f "$sr" ] || die "upscaled chunk is missing: $sr"
        echo "file '$sr'" >> "$WORK_DIR/concat.txt"
    done

    echo "stitch: joining ${#LR_CHUNKS[@]} chunk(s)"
    if ! run_stitch "$partial" -c:a copy; then
        echo "stitch: this container cannot hold the source audio as-is, re-encoding audio to AAC"
        run_stitch "$partial" -c:a aac -b:a 256k
    fi
    mv "$partial" "$OUTPUT"
}

verify_output() {
    local actual

    actual="$(count_frames "$OUTPUT")"
    if [ "$actual" != "$TOTAL_FRAMES" ]; then
        die "frame count mismatch: $TOTAL_FRAMES frames went in, $OUTPUT has $actual (work directory kept: $WORK_DIR)"
    fi
    echo "verify: $TOTAL_FRAMES frames in, $actual frames out"
}

cleanup() {
    if [ "$KEEP_WORK" = 1 ]; then
        echo "work directory kept: $WORK_DIR"
        return
    fi
    rm -rf "$WORK_DIR"
}

# --------------------------------------------------------------------- main --

main() {
    parse_args "$@"
    check_setup
    probe_source
    prepare_workdir
    select_gpu

    cut_source
    list_chunks
    upscale_chunks
    stitch_chunks
    verify_output
    cleanup

    echo "done: $OUTPUT"
}

main "$@"
