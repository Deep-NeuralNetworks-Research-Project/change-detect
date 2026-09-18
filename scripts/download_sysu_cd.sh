#!/usr/bin/env bash
#
# SYSU-CD acquisition helper.
#
# WHAT THIS SCRIPT CANNOT DO
# --------------------------
# SYSU-CD is distributed only via BaiduYun (password `mlls`) and OneDrive, both
# linked from the liumency/SYSU-CD GitHub README (research/02 §2). Both are
# interactive: BaiduYun requires an account and its own client, OneDrive hands out
# short-lived, session-bound direct links. There is no stable direct URL to curl,
# so this script does NOT pretend to download anything. It prints the manual steps,
# then takes over the moment the archive is on disk: extract, verify layout, count.
#
# Usage:
#   bash scripts/download_sysu_cd.sh [TARGET_ROOT] [ARCHIVE]
#
#   TARGET_ROOT  where SYSU-CD should live.
#                Default: "${CD_DATA_ROOT:-<repo>/data}/sysu_cd"
#   ARCHIVE      path to the archive you downloaded by hand.
#                Default: the first archive found in TARGET_ROOT or its parent.
#
# Idempotent: if the expected layout is already present and complete, it verifies
# and exits 0 without re-extracting.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

DATASET="sysu_cd"
TARGET_ROOT="${1:-${CD_DATA_ROOT:-$REPO_ROOT/data}/$DATASET}"
ARCHIVE_ARG="${2:-}"

# research/02 §2 and §1: 20,000 pairs, 256x256, split 12,000 / 4,000 / 4,000 (6:2:2).
EXPECTED_TRAIN=12000
EXPECTED_VAL=4000
EXPECTED_TEST=4000

# Subdirectory names as published in the archive. tests/fixtures/synth.py::make_sysu_cd
# reproduces exactly this layout, and cdlib.data.datasets.sysu_cd reads the names from
# config -- so a surprise here is a config fix, not a code change.
SUBDIRS=(time1 time2 label)
SPLITS=(train val test)

log()  { printf '[sysu-cd] %s\n' "$*"; }
warn() { printf '[sysu-cd] WARNING: %s\n' "$*" >&2; }
die()  { printf '[sysu-cd] ERROR: %s\n' "$*" >&2; exit 1; }

print_manual_instructions() {
    cat <<'INSTRUCTIONS'

================================================================================
  SYSU-CD is a MANUAL download. Nothing below can be automated.
================================================================================

  1. Open the official repository README:

         https://github.com/liumency/SYSU-CD

  2. Pick ONE of the two hosts linked there:

         BaiduYun   password: mlls        (needs a Baidu account + their client)
         OneDrive                          (browser download; link is session-bound)

  3. Download the archive (~ a few GB).

  4. Move it to:

INSTRUCTIONS
    printf '         %s/\n' "$TARGET_ROOT"
    cat <<'INSTRUCTIONS'

  5. Re-run this script. It will extract, verify the layout and count the files.

  Licence note (research/02 §1): SYSU-CD ships NO LICENSE file. Treat it as
  research-only, cite the SYSU-CD paper, and do not redistribute.

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
        # -maxdepth 1: never recurse into an already-extracted tree.
        candidate="$(find "$dir" -maxdepth 1 -type f \
            \( -iname '*SYSU*.zip' -o -iname '*SYSU*.tar.gz' -o -iname '*SYSU*.tgz' \
               -o -iname '*SYSU*.tar' -o -iname '*SYSU*.7z'  -o -iname '*SYSU*.rar' \) \
            2>/dev/null | sort | head -n 1)"
        if [[ -n "$candidate" ]]; then
            printf '%s' "$candidate"
            return 0
        fi
    done
    return 1
}

extract_archive() {
    local archive="$1" dest="$2"
    mkdir -p "$dest"
    log "extracting $(basename "$archive") -> $dest"
    case "$archive" in
        *.zip)
            command -v unzip >/dev/null 2>&1 || die "unzip not found; install it or extract by hand"
            # -n: never clobber, so a re-run after a partial extract resumes cheaply.
            unzip -q -n "$archive" -d "$dest"
            ;;
        *.tar.gz|*.tgz) tar -xzf "$archive" -C "$dest" ;;
        *.tar)          tar -xf  "$archive" -C "$dest" ;;
        *.7z)
            command -v 7z >/dev/null 2>&1 || die "7z not found; install p7zip or extract by hand"
            7z x -y -o"$dest" "$archive" >/dev/null
            ;;
        *.rar)
            if   command -v unrar >/dev/null 2>&1; then unrar x -y "$archive" "$dest/" >/dev/null
            elif command -v unar  >/dev/null 2>&1; then unar -q -o "$dest" "$archive"
            else die "no unrar/unar found; install one or extract $archive by hand"
            fi
            ;;
        *) die "unrecognised archive type: $archive (extract it by hand into $dest)" ;;
    esac
}

# Some mirrors nest everything one level deep (<root>/SYSU-CD/train/...). Flatten it
# so downstream config never has to care which mirror the operator used.
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
    find "$dir" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.bmp' -o -iname '*.tif' \) \
        | wc -l | tr -d ' '
}

verify_layout() {
    local ok=0 split sub n expected
    log "verifying layout under $TARGET_ROOT"
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
            die "no SYSU-CD archive found. Follow the steps above, then re-run."
        fi
        log "found archive: $archive"
        extract_archive "$archive" "$TARGET_ROOT"
        flatten_single_wrapper_dir
    fi

    local layout_ok=0
    verify_layout || layout_ok=$?
    if [[ $layout_ok -eq 0 ]]; then
        log "counts match the official 12,000 / 4,000 / 4,000 split (research/02 §2)."
    else
        warn "counts do NOT match the official split. Do not report numbers off this copy"
        warn "until the mismatch is explained -- a truncated archive is the usual cause."
    fi

    cat <<NEXT_STEPS

Next steps (CLAUDE.md rule 9 -- track checksums and manifests, never the media):

  python scripts/verify_dataset.py --root "$TARGET_ROOT" --dataset sysu_cd --write-manifest
  python scripts/verify_dataset.py --root "$TARGET_ROOT" --dataset sysu_cd --summary
  bash   scripts/archive_to_drive.sh "$TARGET_ROOT"

NEXT_STEPS

    # A count mismatch is a data problem the operator must see, so it is a non-zero
    # exit -- but a *soft* one (2), distinct from the hard failures above (1).
    [[ $layout_ok -eq 0 ]] || exit 2
}

main "$@"
