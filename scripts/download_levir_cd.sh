#!/usr/bin/env bash
#
# LEVIR-CD acquisition helper.
#
# WHAT THIS SCRIPT CANNOT DO
# --------------------------
# LEVIR-CD is published from justchenhao.github.io/LEVIR/ and hosted on Google
# Drive (research/02 §3). Google Drive serves a virus-scan interstitial for
# multi-GB files and rotates confirm tokens, so the usual curl/gdown one-liners
# break without warning and sometimes silently save an HTML error page as a .zip.
# This script therefore does NOT attempt the download. It prints the manual steps
# and then takes over: extract, verify layout, count.
#
# Usage:
#   bash scripts/download_levir_cd.sh [TARGET_ROOT] [ARCHIVE]
#
#   TARGET_ROOT  where LEVIR-CD should live.
#                Default: "${CD_DATA_ROOT:-<repo>/data}/levir_cd"
#   ARCHIVE      path to the archive you downloaded by hand.
#                Default: the first archive found in TARGET_ROOT or its parent.
#
# Idempotent: an already-complete layout is verified, not re-extracted.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

DATASET="levir_cd"
TARGET_ROOT="${1:-${CD_DATA_ROOT:-$REPO_ROOT/data}/$DATASET}"
ARCHIVE_ARG="${2:-}"

# research/02 §3: 637 VHR 1024x1024 pairs, official split 445 / 64 / 128 FULL pairs.
# The 7,120 / 1,024 / 2,048 numbers everyone quotes are the 256x256 non-overlapping
# crops derived from these (16 crops per 1024x1024 pair) -- the crop step belongs to
# cdlib.data.datasets.levir_cd, not here. This script counts full pairs only.
EXPECTED_TRAIN=445
EXPECTED_VAL=64
EXPECTED_TEST=128

# CLAUDE.md rule 3: 256x256 non-overlapping -> 7,120 / 1,024 / 2,048. Not negotiable.
CROPS_PER_PAIR=16

# tests/fixtures/synth.py::make_levir_cd reproduces exactly this layout.
SUBDIRS=(A B label)
SPLITS=(train val test)

log()  { printf '[levir-cd] %s\n' "$*"; }
warn() { printf '[levir-cd] WARNING: %s\n' "$*" >&2; }
die()  { printf '[levir-cd] ERROR: %s\n' "$*" >&2; exit 1; }

print_manual_instructions() {
    cat <<'INSTRUCTIONS'

================================================================================
  LEVIR-CD is a MANUAL download. Nothing below can be automated reliably.
================================================================================

  1. Open the official dataset page:

         https://justchenhao.github.io/LEVIR/

  2. Follow its Google Drive link and download the LEVIR-CD archive
     (the one containing train/ val/ test/, ~2 GB).

     If your browser shows "Google Drive can't scan this file for viruses",
     click "Download anyway". Do NOT curl the link -- for large files Drive
     returns the interstitial HTML, and you end up with a 3 KB "zip" that
     unzips to nothing.

  3. Move the archive to:

INSTRUCTIONS
    printf '         %s/\n' "$TARGET_ROOT"
    cat <<'INSTRUCTIONS'

  4. Re-run this script. It will extract, verify the layout and count pairs.

  Licence note (research/02 §3): academic / non-commercial use, with Google
  Earth Terms of Service applying on top of that. Not redistributable.

  LEVIR-CD+ is a DIFFERENT dataset (985 pairs) whose licence is listed as
  "unknown" on the HuggingFace mirror. Do not silently substitute it; email the
  LEVIR group first (research/02 §3).

================================================================================

INSTRUCTIONS
}

layout_present() {
    local split sub
    for split in "${SPLITS[@]}"; do
        for sub in "${SUBDIRS[@]}"; do
            [[ -d "$TARGET_ROOT/$split/$sub" ]] || return 1
        done
    done
    return 0
}

find_archive() {
    if [[ -n "$ARCHIVE_ARG" ]]; then
        [[ -f "$ARCHIVE_ARG" ]] || die "archive not found: $ARCHIVE_ARG"
        printf '%s' "$ARCHIVE_ARG"
        return 0
    fi
    local dir candidate
    for dir in "$TARGET_ROOT" "$(dirname "$TARGET_ROOT")"; do
        [[ -d "$dir" ]] || continue
        candidate="$(find "$dir" -maxdepth 1 -type f \
            \( -iname '*LEVIR*.zip' -o -iname '*LEVIR*.tar.gz' -o -iname '*LEVIR*.tgz' \
               -o -iname '*LEVIR*.tar' -o -iname '*LEVIR*.7z' \) \
            2>/dev/null | sort | head -n 1)"
        if [[ -n "$candidate" ]]; then
            printf '%s' "$candidate"
            return 0
        fi
    done
    return 1
}

# A Drive interstitial saved as a .zip is the single most common failure here, and
# it fails *later*, confusingly, inside unzip. Catch it up front.
reject_html_masquerading_as_archive() {
    local archive="$1" head_bytes
    head_bytes="$(head -c 512 "$archive" | tr -d '\0' | tr 'A-Z' 'a-z' || true)"
    case "$head_bytes" in
        *'<html'*|*'<!doctype html'*)
            die "$archive is an HTML page, not an archive -- Google Drive served you its
       virus-scan interstitial. Re-download through a browser and click
       'Download anyway'."
            ;;
    esac
}

extract_archive() {
    local archive="$1" dest="$2"
    reject_html_masquerading_as_archive "$archive"
    mkdir -p "$dest"
    log "extracting $(basename "$archive") -> $dest"
    case "$archive" in
        *.zip)
            command -v unzip >/dev/null 2>&1 || die "unzip not found; install it or extract by hand"
            unzip -q -n "$archive" -d "$dest"
            ;;
        *.tar.gz|*.tgz) tar -xzf "$archive" -C "$dest" ;;
        *.tar)          tar -xf  "$archive" -C "$dest" ;;
        *.7z)
            command -v 7z >/dev/null 2>&1 || die "7z not found; install p7zip or extract by hand"
            7z x -y -o"$dest" "$archive" >/dev/null
            ;;
        *) die "unrecognised archive type: $archive (extract it by hand into $dest)" ;;
    esac
}

flatten_single_wrapper_dir() {
    local inner
    if layout_present; then return 0; fi
    inner="$(find "$TARGET_ROOT" -mindepth 1 -maxdepth 1 -type d | head -n 2)"
    [[ "$(printf '%s\n' "$inner" | wc -l)" -eq 1 ]] || return 0
    [[ -d "$inner/train" ]] || return 0
    log "flattening wrapper directory $(basename "$inner")"
    # shellcheck disable=SC2086
    mv "$inner"/* "$TARGET_ROOT"/ && rmdir "$inner"
}

count_files() {
    local dir="$1"
    [[ -d "$dir" ]] || { printf '0'; return; }
    find "$dir" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.tif' \) | wc -l | tr -d ' '
}

verify_layout() {
    local ok=0 split sub n expected
    log "verifying layout under $TARGET_ROOT (full 1024x1024 pairs, pre-crop)"
    printf '\n  %-8s %-8s %10s %10s   %s\n' split dir found expected status
    printf '  %s\n' "--------------------------------------------------------------"
    for split in "${SPLITS[@]}"; do
        case "$split" in
            train) expected=$EXPECTED_TRAIN ;;
            val)   expected=$EXPECTED_VAL ;;
            test)  expected=$EXPECTED_TEST ;;
        esac
        for sub in "${SUBDIRS[@]}"; do
            if [[ ! -d "$TARGET_ROOT/$split/$sub" ]]; then
                printf '  %-8s %-8s %10s %10s   %s\n' "$split" "$sub" "-" "$expected" "MISSING DIR"
                ok=1
                continue
            fi
            n="$(count_files "$TARGET_ROOT/$split/$sub")"
            if [[ "$n" -eq "$expected" ]]; then
                printf '  %-8s %-8s %10s %10s   %s\n' "$split" "$sub" "$n" "$expected" "ok"
            else
                printf '  %-8s %-8s %10s %10s   %s\n' "$split" "$sub" "$n" "$expected" "MISMATCH"
                ok=1
            fi
        done
    done
    printf '\n'
    printf '  At the frozen %sx%s non-overlapping crop policy (CLAUDE.md rule 3) these\n' 256 256
    printf '  become %s / %s / %s crops.\n\n' \
        "$((EXPECTED_TRAIN * CROPS_PER_PAIR))" \
        "$((EXPECTED_VAL   * CROPS_PER_PAIR))" \
        "$((EXPECTED_TEST  * CROPS_PER_PAIR))"
    return $ok
}

main() {
    log "target root: $TARGET_ROOT"
    mkdir -p "$TARGET_ROOT"

    if layout_present; then
        log "layout already present -- skipping extraction (idempotent re-run)"
    else
        local archive
        if ! archive="$(find_archive)"; then
            print_manual_instructions
            die "no LEVIR-CD archive found. Follow the steps above, then re-run."
        fi
        log "found archive: $archive"
        extract_archive "$archive" "$TARGET_ROOT"
        flatten_single_wrapper_dir
    fi

    local layout_ok=0
    verify_layout || layout_ok=$?
    if [[ $layout_ok -eq 0 ]]; then
        log "counts match the official 445 / 64 / 128 split (research/02 §3)."
    else
        warn "counts do NOT match the official split. Our numbers are only comparable to"
        warn "the literature on the official split -- resolve this before training."
    fi

    cat <<NEXT_STEPS

Next steps (CLAUDE.md rule 9 -- track checksums and manifests, never the media):

  python scripts/verify_dataset.py --root "$TARGET_ROOT" --dataset levir_cd --write-manifest
  python scripts/verify_dataset.py --root "$TARGET_ROOT" --dataset levir_cd --summary
  bash   scripts/archive_to_drive.sh "$TARGET_ROOT"

NEXT_STEPS

    [[ $layout_ok -eq 0 ]] || exit 2
}

main "$@"
