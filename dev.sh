#!/bin/bash
# =============================================================================
# Persistent Dev Container for OGL Development
# =============================================================================
# Instead of rebuilding the Docker image for every .C file change (~12 min),
# this runs a long-lived container with your source mounted as a volume.
# Incremental wmake recompiles only changed files (~30-60 sec).
#
# Usage:
#   ./dev.sh start          Start the persistent dev container
#   ./dev.sh compile         Incremental wmake of OGL module
#   ./dev.sh compile-all     Full OpenFOAM + OGL recompile
#   ./dev.sh shell           Interactive shell inside container
#   ./dev.sh stop            Stop and remove the container
#   ./dev.sh status          Check if container is running
#   ./dev.sh benchmark ...   Run benchmark.py inside the container
#
# First-time setup:
#   docker build -t mixfoam:latest .    # Build the base image once
#   ./dev.sh start                       # Start the persistent container
#   ./dev.sh compile                     # Incremental compile after code edits
# =============================================================================

set -euo pipefail

CONTAINER_NAME="ogl-dev"
BASE_IMAGE="mixfoam:latest"
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Source directory inside the container where OGL source is mounted
OGL_SRC_MOUNT="/workspace/modules/OGL"
# The original compiled OGL in the image (used as build cache seed)
OGL_IMG="/opt/OpenFOAM-13/modules/OGL"

# Named Docker volumes for build artifacts (survive container stop/start)
OBJ_VOLUME="ogl-obj-cache"       # wmake object files + .dep dependency tracking
LIB_VOLUME="ogl-lib-cache"       # Final libOGL.so output
OBJ_PATH="/opt/OpenFOAM-13/platforms/linux64GccDPInt32Opt/modules/OGL"
LIB_PATH="/root/OpenFOAM/root-13/platforms/linux64GccDPInt32Opt/lib"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info()  { echo -e "${GREEN}[dev]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[dev]${NC} $*"; }
log_error() { echo -e "${RED}[dev]${NC} $*"; }

# -----------------------------------------------------------------------------
# Check if the container is running
# -----------------------------------------------------------------------------
is_running() {
    docker inspect --format='{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null | grep -q true
}

is_exists() {
    docker inspect "$CONTAINER_NAME" &>/dev/null
}

# -----------------------------------------------------------------------------
# Start the persistent dev container
# -----------------------------------------------------------------------------
cmd_start() {
    if is_running; then
        log_info "Container '$CONTAINER_NAME' is already running."
        return 0
    fi

    if is_exists; then
        log_info "Starting existing container '$CONTAINER_NAME'..."
        docker start "$CONTAINER_NAME"
        return 0
    fi

    log_info "Creating persistent dev container '$CONTAINER_NAME'..."
    log_info "  Base image: $BASE_IMAGE"
    log_info "  OGL source: $REPO_ROOT/modules/OGL -> $OGL_SRC_MOUNT"

    docker run -d \
        --gpus all \
        --name "$CONTAINER_NAME" \
        -v "$REPO_ROOT/modules/OGL/src:${OGL_SRC_MOUNT}/src:ro" \
        -v "$REPO_ROOT/tutorials:/workspace/tutorials:ro" \
        -v "$REPO_ROOT/benchmarks:/workspace/benchmarks" \
        -v "${OBJ_VOLUME}:${OBJ_PATH}" \
        -v "${LIB_VOLUME}:${LIB_PATH}" \
        "$BASE_IMAGE" \
        sleep infinity

    # Named volumes auto-populate from the image on first creation (Docker
    # copies the image's directory contents into the empty volume). On
    # subsequent container recreations the volumes retain cached .o / .dep
    # files, so wmake only recompiles what actually changed.
    log_info "Build cache volumes: ${OBJ_VOLUME}, ${LIB_VOLUME}"

    # Copy OGL source from mounted volume to the build location
    # (wmake needs writable source tree for .dep files)
    log_info "Syncing OGL source into build tree..."
    docker exec "$CONTAINER_NAME" bash -c "
        rm -rf ${OGL_IMG}/src && cp -a ${OGL_SRC_MOUNT}/src ${OGL_IMG}/src
        cp ${OGL_SRC_MOUNT}/../Allwmake ${OGL_IMG}/Allwmake 2>/dev/null || true
    "

    log_info "Container '$CONTAINER_NAME' is ready."
    log_info "Run './dev.sh compile' after editing OGL source files."
}

# -----------------------------------------------------------------------------
# Incremental compile: only recompile changed OGL .C files
# -----------------------------------------------------------------------------
cmd_compile() {
    if ! is_running; then
        log_error "Container '$CONTAINER_NAME' is not running. Run './dev.sh start' first."
        exit 1
    fi

    log_info "Syncing OGL source and running incremental wmake..."
    local start_time=$SECONDS

    docker exec "$CONTAINER_NAME" bash -c "
        export FOAM_INST_DIR=/opt
        source /opt/OpenFOAM-13/etc/bashrc SCOTCH_TYPE=system
        export GINKGO_ROOT=/opt/ginkgo
        export WM_NCOMPPROCS=\$(nproc)

        # Sync only changed source files from the read-only mount
        rm -rf ${OGL_IMG}/src && cp -a ${OGL_SRC_MOUNT}/src ${OGL_IMG}/src

        # Incremental wmake — only recompiles changed .C files
        cd ${OGL_IMG}
        wmake src 2>&1
    "
    local rc=$?
    local elapsed=$((SECONDS - start_time))

    if [ $rc -eq 0 ]; then
        log_info "OGL compile completed in ${elapsed}s"
    else
        log_error "OGL compile FAILED (exit code $rc) after ${elapsed}s"
        exit $rc
    fi
}

# -----------------------------------------------------------------------------
# Full recompile: OpenFOAM + OGL
# -----------------------------------------------------------------------------
cmd_compile_all() {
    if ! is_running; then
        log_error "Container '$CONTAINER_NAME' is not running. Run './dev.sh start' first."
        exit 1
    fi

    log_info "Running full OpenFOAM + OGL compile..."
    local start_time=$SECONDS

    docker exec "$CONTAINER_NAME" bash -c "
        export FOAM_INST_DIR=/opt
        source /opt/OpenFOAM-13/etc/bashrc SCOTCH_TYPE=system
        export GINKGO_ROOT=/opt/ginkgo
        export WM_NCOMPPROCS=\$(nproc)

        cd /opt/OpenFOAM-13
        ./Allwmake -j\$(nproc) 2>&1 | tee /tmp/build.log | tail -40

        # Sync and recompile OGL
        rm -rf ${OGL_IMG}/src && cp -a ${OGL_SRC_MOUNT}/src ${OGL_IMG}/src
        cd ${OGL_IMG}
        wmake src 2>&1
    "
    local elapsed=$((SECONDS - start_time))
    log_info "Full compile completed in ${elapsed}s"
}

# -----------------------------------------------------------------------------
# Run a benchmark inside the dev container
# -----------------------------------------------------------------------------
cmd_benchmark() {
    if ! is_running; then
        log_error "Container '$CONTAINER_NAME' is not running. Run './dev.sh start' first."
        exit 1
    fi

    # Pass all remaining arguments to benchmark.py
    local bench_args=("$@")

    log_info "Running benchmark: ${bench_args[*]}"

    docker exec "$CONTAINER_NAME" bash -c "
        export FOAM_INST_DIR=/opt
        source /opt/OpenFOAM-13/etc/bashrc
        export GINKGO_ROOT=/opt/ginkgo
        cd /workspace/benchmarks
        python3 benchmark.py ${bench_args[*]}
    "
}

# -----------------------------------------------------------------------------
# Interactive shell
# -----------------------------------------------------------------------------
cmd_shell() {
    if ! is_running; then
        log_error "Container '$CONTAINER_NAME' is not running. Run './dev.sh start' first."
        exit 1
    fi

    log_info "Opening shell in '$CONTAINER_NAME'..."
    docker exec -it "$CONTAINER_NAME" bash -c "
        export FOAM_INST_DIR=/opt
        source /opt/OpenFOAM-13/etc/bashrc
        export GINKGO_ROOT=/opt/ginkgo
        exec bash
    "
}

# -----------------------------------------------------------------------------
# Stop the container
# -----------------------------------------------------------------------------
cmd_stop() {
    if is_running; then
        log_info "Stopping container '$CONTAINER_NAME'..."
        docker stop "$CONTAINER_NAME"
    fi
    if is_exists; then
        log_info "Removing container '$CONTAINER_NAME'..."
        docker rm "$CONTAINER_NAME"
    fi
    log_info "Container removed."
}

# -----------------------------------------------------------------------------
# Status check
# -----------------------------------------------------------------------------
cmd_status() {
    if is_running; then
        log_info "Container '$CONTAINER_NAME' is running."
        docker exec "$CONTAINER_NAME" bash -c "
            echo '  Uptime:' \$(cat /proc/uptime | cut -d' ' -f1)s
            echo '  GPU:' \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'not available')
            echo '  OGL lib:' \$(ls -la /root/OpenFOAM/root-13/platforms/linux64GccDPInt32Opt/lib/libOGL.so 2>/dev/null || echo 'not found')
        "
    elif is_exists; then
        log_warn "Container '$CONTAINER_NAME' exists but is stopped. Run './dev.sh start'."
    else
        log_warn "Container '$CONTAINER_NAME' does not exist. Run './dev.sh start'."
    fi
}

# -----------------------------------------------------------------------------
# Clean build cache volumes (use after Docker image rebuild)
# -----------------------------------------------------------------------------
cmd_clean_cache() {
    log_info "Removing build cache volumes..."
    docker volume rm "$OBJ_VOLUME" 2>/dev/null && log_info "  Removed $OBJ_VOLUME" || log_warn "  $OBJ_VOLUME not found"
    docker volume rm "$LIB_VOLUME" 2>/dev/null && log_info "  Removed $LIB_VOLUME" || log_warn "  $LIB_VOLUME not found"
    log_info "Build cache cleared. Next './dev.sh start' will re-seed from image."
}

# -----------------------------------------------------------------------------
# Main dispatch
# -----------------------------------------------------------------------------
case "${1:-help}" in
    start)       cmd_start ;;
    compile)     cmd_compile ;;
    compile-all) cmd_compile_all ;;
    benchmark)   shift; cmd_benchmark "$@" ;;
    shell)       cmd_shell ;;
    stop)        cmd_stop ;;
    clean-cache) cmd_clean_cache ;;
    status)      cmd_status ;;
    help|--help|-h)
        echo "Usage: ./dev.sh <command> [args]"
        echo ""
        echo "Commands:"
        echo "  start          Start the persistent dev container"
        echo "  compile        Incremental OGL compile (~30-60s)"
        echo "  compile-all    Full OpenFOAM + OGL compile"
        echo "  shell          Interactive shell inside container"
        echo "  benchmark ...  Run benchmark.py with given arguments"
        echo "  stop           Stop and remove the container"
        echo "  clean-cache    Remove build cache volumes (after image rebuild)"
        echo "  status         Check container status"
        echo ""
        echo "Workflow:"
        echo "  1. docker build -t mixfoam:latest .  # Once (or after Dockerfile changes)"
        echo "  2. ./dev.sh start                     # Start persistent container"
        echo "  3. <edit OGL source files>"
        echo "  4. ./dev.sh compile                   # ~30-60s incremental"
        echo "  5. ./dev.sh benchmark scaling ...     # Run benchmark"
        echo "  6. Repeat steps 3-5"
        ;;
    *)
        log_error "Unknown command: $1"
        echo "Run './dev.sh help' for usage."
        exit 1
        ;;
esac
