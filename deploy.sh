#!/usr/bin/env bash

# ==============================================================================
# FindWay CNN Moderator — Production Deployment Pipeline (Multiplexed SSH)
# Target Host: 46.224.148.229 (Hetzner Infrastructure)
# ==============================================================================

set -euo pipefail

REMOTE_USER="root"
REMOTE_HOST="46.224.148.229"
REMOTE_DIR="~/moderator_CNN"
SERVICE_NAME="moderator"
CONTAINER_NAME="cnn_moderator"
PORT="8001"
MAX_HEALTH_RETRIES=10
RETRY_INTERVAL=3

# SSH Multiplexing settings (сохраняет соединение открытым, пароль вводится 1 раз)
SSH_MUX_DIR="/tmp/ssh_mux_findway"
mkdir -p "$SSH_MUX_DIR"
SSH_OPTS="-o ControlMaster=auto -o ControlPath=$SSH_MUX_DIR/mux-%r@%h:%p -o ControlPersist=5m -o ServerAliveInterval=30"

# Cleanup background master connection on exit
trap "ssh $SSH_OPTS -O exit ${REMOTE_USER}@${REMOTE_HOST} &>/dev/null || true" EXIT

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ------------------------------------------------------------------------------
# Phase 1: Pre-flight & Master Connection Initialization
# ------------------------------------------------------------------------------
log_info "Establishing persistent SSH control master connection..."
# Здесь пароль вводится ОДИН РАЗ, дальше соединение живет в фоне
ssh $SSH_OPTS -N -f "${REMOTE_USER}@${REMOTE_HOST}"

log_info "SSH connectivity verified."

# ------------------------------------------------------------------------------
# Phase 2: Code Synchronization via Rsync (использует тот же сокет)
# ------------------------------------------------------------------------------
log_info "Synchronizing code to remote target (${REMOTE_DIR})..."

rsync -avz \
    -e "ssh $SSH_OPTS" \
    --exclude='runs/' \
    --exclude='data/' \
    --exclude='.git/' \
    --exclude='.gitignore' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.venv/' \
    --exclude='venv/' \
    --exclude='.idea/' \
    --exclude='.vscode/' \
    ./ "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

log_info "Rsync synchronization complete."

# ------------------------------------------------------------------------------
# Phase 3: Isolated Service Build & Restart
# ------------------------------------------------------------------------------
log_info "Rebuilding and restarting service [${SERVICE_NAME}] on remote host..."

ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" "
    cd ${REMOTE_DIR}
    mkdir -p weights data
    docker compose up -d --build --no-deps ${SERVICE_NAME}
"

# ------------------------------------------------------------------------------
# Phase 4: Active Healthcheck Verification
# ------------------------------------------------------------------------------
log_info "Waiting for ${CONTAINER_NAME} to initialize and pass healthcheck on port ${PORT}..."

HEALTHY=false

# Исправленный блок Phase 4 в скрипте деплоя
HEALTH_CHECK_RESULT=$(ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" bash -s <<EOF
    for i in \$(seq 1 ${MAX_HEALTH_RETRIES}); do
        STATUS=\$(curl -s -o /dev/null -w "%{http_code}" http://localhost:${PORT}/docs || echo "000")
        if [ "\$STATUS" -eq 200 ]; then
            echo "OK:\$STATUS"
            exit 0
        fi
        sleep ${RETRY_INTERVAL}
    done
    echo "FAIL"
    exit 1
EOF
)

if [[ "$HEALTH_CHECK_RESULT" == *"OK"* ]]; then
    HTTP_STATUS=$(echo "$HEALTH_CHECK_RESULT" | cut -d':' -f2)
    log_info "Healthcheck passed successfully (HTTP ${HTTP_STATUS})."
else
    log_error "Deployment verification failed! Container ${CONTAINER_NAME} did not become ready."
    log_error "Fetching last 50 lines of container logs..."
    ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" "docker logs --tail 50 ${CONTAINER_NAME}"
    exit 1
fi

# ------------------------------------------------------------------------------
# Phase 5: Resource Allocation Report
# ------------------------------------------------------------------------------
log_info "Fetching runtime status and memory metrics..."

ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" "
    echo '--------------------------------------------------------------------------------'
    docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' --filter 'name=${CONTAINER_NAME}' --filter 'name=findway_elasticsearch'
    echo '--------------------------------------------------------------------------------'
    docker stats --no-stream ${CONTAINER_NAME} findway_elasticsearch
    echo '--------------------------------------------------------------------------------'
"

log_info "Deployment completed successfully."