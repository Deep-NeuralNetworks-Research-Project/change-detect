#!/usr/bin/env bash
#
# PCD / TSUNAMI acquisition helper (Sakurada & Okatani, BMVC 2015).
#
# READ THIS BEFORE RUNNING
# ------------------------
# * The ORIGINAL host (vision.is.tohoku.ac.jp) is CONFIRMED DEAD -- 404, see
#   sscdnet issue #1. The canonical page is now sakuradaken.net/pcd_dataset.html,
#   one researcher's personal site (research/02 §4).
# * Only TSUNAMI has a live link. **GSV is not hosted anywhere**, and research/02
#   §4 found NO third-party mirror -- no Kaggle, no HuggingFace, no Zenodo.
#   This script therefore never treats a missing GSV as an error.
# * That makes PCD a SINGLE POINT OF FAILURE (research/02 §4 and "Two risks",
#   P1-data-lead.md). The moment TSUNAMI lands, mirror it to team storage.
# * Do NOT confuse PCD with PSCD at sakuradaken.net/pscd/ -- a different, newer
#   dataset (770 pairs, ICRA-2020 lineage, different annotation schema).
#
# The download itself is Google-Drive-hosted and interactive, so this script does
# not pretend to fetch it. It prints the manual steps, then extracts, verifies and
# nags you to archive.
#
# Usage:
#   bash scripts/download_pcd_tsunami.sh [TARGET_ROOT] [ARCHIVE]
#
#   TARGET_ROOT  where PCD should live.
#                Default: "${CD_DATA_ROOT:-<repo>/data}/pcd"
#   ARCHIVE      path to the archive you downloaded by hand.
#
# Idempotent: an already-complete TSUNAMI tree is verified, not re-extracted.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

DATASET="pcd"
TARGET_ROOT="${1:-${CD_DATA_ROOT:-$REPO_ROOT/data}/$DATASET}"
ARCHIVE_ARG="${2:-}"

PCD_PAGE="https://sakuradaken.net/pcd_dataset.html"
PSCD_PAGE="https://sakuradaken.net/pscd/"

# research/02 §4: two 100-pair subsets, each 224x1024 equirectangular panoramas
# with binary PNG change masks. Protocol is 5-fold CV over 20-pair folds.
EXPECTED_PAIRS=100
CV_FOLDS=5

# tests/fixtures/synth.py::make_pcd reproduces exactly this layout:
#   <root>/TSUNAMI/{t0,t1,mask}/NNNNN.{jpg,jpg,bmp}
SUBDIRS=(t0 t1 mask)
SUBSET="TSUNAMI"
OPTIONAL_SUBSET="GSV"

log()  { printf '[pcd] %s\n' "$*"; }
warn() { printf '[pcd] WARNING: %s\n' "$*" >&2; }
die()  { printf '[pcd] ERROR: %s\n' "$*" >&2; exit 1; }

banner() {
    printf '\n'
    printf '################################################################################\n'
    printf '# %s\n' "$@"
    printf '################################################################################\n'
    printf '\n'
}

print_manual_instructions() {
    banner "PCD is a MANUAL download, and it is our most fragile dataset." \
            "Only TSUNAMI is obtainable. GSV is NOT hosted anywhere."
    cat <<INSTRUCTIONS
  1. Open the canonical page (the old Tohoku host is dead -- 404):

         $PCD_PAGE

     NOT $PSCD_PAGE -- that is PSCD, a different dataset with a
     different annotation schema. Downloading it by mistake is easy and the
     directory layouts look similar enough to fool you.

  2. Follow the TSUNAMI link (Google Drive) and download the archive.

     There is no GSV link, and there is no mirror. If you need GSV, email the
     authors -- and keep it off the critical path (P1-data-lead.md).

  3. Move the archive to:

         $TARGET_ROOT/

  4. Re-run this script.

  Licence note (research/02 §1/§4): research use; cite Sakurada & Okatani,
  BMVC 2015. Not redistributable outside the team.

INSTRUCTIONS
}

layout_present() {
    local sub
    for sub in "${SUBDIRS[@]}"; do
        [[ -d "$TARGET_ROOT/$SUBSET/$sub" ]] || return 1
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
            \( -iname '*TSUNAMI*.zip' -o -iname '*TSUNAMI*.tar.gz' -o -iname '*TSUNAMI*.tgz' \
               -o -iname '*pcd*.zip'  -o -iname '*pcd*.tar.gz' \) \
            2>/dev/null | sort | head -n 1)"
        if [[ -n "$candidate" ]]; then
            printf '%s' "$candidate"
            return 0
        fi
    done
    return 1
}

# PSCD ships privacy masks and instance labels; PCD does not. If we see those
# directory names we grabbed the wrong dataset, and finding that out three weeks
# later would be expensive.
reject_pscd() {
    local dest="$1"
    if find "$dest" -maxdepth 3 -type d \
            \( -iname '*privacy*' -o -iname '*instance*' -o -iname '*semantic*' \) \
            2>/dev/null | grep -q .; then
        warn "this archive contains privacy/instance/semantic directories."
        warn "That is PSCD ($PSCD_PAGE), NOT PCD."
        die  "wrong dataset -- delete it and download TSUNAMI from $PCD_PAGE"
    fi
}

extract_archive() {
    local archive="$1" dest="$2"
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
    reject_pscd "$dest"
}

# Mirrors vary: <root>/TSUNAMI/... or <root>/PCD/TSUNAMI/... . Normalise.
flatten_single_wrapper_dir() {
    local inner
    if layout_present; then return 0; fi
    inner="$(find "$TARGET_ROOT" -mindepth 1 -maxdepth 1 -type d | head -n 2)"
    [[ "$(printf '%s\n' "$inner" | wc -l)" -eq 1 ]] || return 0
    [[ -d "$inner/$SUBSET" ]] || return 0
    log "flattening wrapper directory $(basename "$inner")"
    # shellcheck disable=SC2086
    mv "$inner"/* "$TARGET_ROOT"/ && rmdir "$inner"
}

count_files() {
    local dir="$1"
    [[ -d "$dir" ]] || { printf '0'; return; }
    find "$dir" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.bmp' \) | wc -l | tr -d ' '
}

verify_layout() {
    local ok=0 sub n
    log "verifying $SUBSET layout under $TARGET_ROOT"
    printf '\n  %-10s %-8s %10s %10s   %s\n' subset dir found expected status
    printf '  %s\n' "----------------------------------------------------------------"
    for sub in "${SUBDIRS[@]}"; do
        if [[ ! -d "$TARGET_ROOT/$SUBSET/$sub" ]]; then
            printf '  %-10s %-8s %10s %10s   %s\n' "$SUBSET" "$sub" "-" "$EXPECTED_PAIRS" "MISSING DIR"
            ok=1
            continue
        fi
        n="$(count_files "$TARGET_ROOT/$SUBSET/$sub")"
        if [[ "$n" -eq "$EXPECTED_PAIRS" ]]; then
            printf '  %-10s %-8s %10s %10s   %s\n' "$SUBSET" "$sub" "$n" "$EXPECTED_PAIRS" "ok"
        else
            printf '  %-10s %-8s %10s %10s   %s\n' "$SUBSET" "$sub" "$n" "$EXPECTED_PAIRS" "MISMATCH"
            ok=1
        fi
    done

    # GSV absence is the EXPECTED state, not a failure (research/02 §4).
    if [[ -d "$TARGET_ROOT/$OPTIONAL_SUBSET" ]]; then
        n="$(count_files "$TARGET_ROOT/$OPTIONAL_SUBSET/t0")"
        printf '  %-10s %-8s %10s %10s   %s\n' "$OPTIONAL_SUBSET" "t0" "$n" "$EXPECTED_PAIRS" "present (bonus)"
    else
        printf '  %-10s %-8s %10s %10s   %s\n' "$OPTIONAL_SUBSET" "-" "0" "n/a" "not hosted -- expected"
    fi
    printf '\n'
    printf '  Protocol: %s-fold CV over %s-pair folds (research/02 §4). Build the folds at\n' \
        "$CV_FOLDS" "$((EXPECTED_PAIRS / CV_FOLDS))"
    printf '  panorama level, BEFORE any patch cropping -- CLAUDE.md rule 2.\n\n'
    return $ok
}

archive_reminder() {
    banner \
        "ARCHIVE THIS TO TEAM STORAGE NOW. NOT LATER. NOW." \
        "" \
        "PCD is a single point of failure: one researcher's personal site, one" \
        "Google Drive link, no Kaggle/HF/Zenodo mirror anywhere (research/02 §4)." \
        "If that link dies before we copy it, this dataset is gone from the project." \
        "" \
        "  bash scripts/archive_to_drive.sh \"$TARGET_ROOT\"" \
        "" \
        "And email the authors about GSV in the same sitting -- it has the longest" \
        "lead time of anything left in the acquisition track."
}

main() {
    log "target root: $TARGET_ROOT"
    mkdir -p "$TARGET_ROOT"

    if layout_present; then
        log "$SUBSET layout already present -- skipping extraction (idempotent re-run)"
    else
        local archive
        if ! archive="$(find_archive)"; then
            print_manual_instructions
            die "no PCD/TSUNAMI archive found. Follow the steps above, then re-run."
        fi
        log "found archive: $archive"
        extract_archive "$archive" "$TARGET_ROOT"
        flatten_single_wrapper_dir
    fi

    local layout_ok=0
    verify_layout || layout_ok=$?
    if [[ $layout_ok -eq 0 ]]; then
        log "$SUBSET complete: $EXPECTED_PAIRS panoramas per directory."
    else
        warn "$SUBSET counts do NOT match the expected $EXPECTED_PAIRS panoramas."
        warn "Re-download before trusting anything computed from this copy."
    fi

    # Print the archive nag even on a count mismatch: a partial TSUNAMI copy is
    # still more than we would have if the link dies tomorrow.
    archive_reminder

    cat <<NEXT_STEPS
Next steps (CLAUDE.md rule 9 -- track checksums and manifests, never the media):

  python scripts/verify_dataset.py --root "$TARGET_ROOT" --dataset pcd --write-manifest
  python scripts/verify_dataset.py --root "$TARGET_ROOT" --dataset pcd --summary
  bash   scripts/archive_to_drive.sh "$TARGET_ROOT"

NEXT_STEPS

    [[ $layout_ok -eq 0 ]] || exit 2
}

main "$@"
