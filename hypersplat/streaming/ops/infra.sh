#!/usr/bin/env bash
# Local Kafka (KRaft, one node) + PostgreSQL with podman/docker run, for development and the
# per-unit e2e tests. Production-style startup with every layer is ops/up.sh + docker-compose.yml.
#
#   ops/infra.sh up [DBS...]   start both (idempotent); create extra databases, e.g. hypersplat_u1
#   ops/infra.sh status        container state + port checks
#   ops/infra.sh down          stop and remove both containers (data is not kept)
set -euo pipefail
RUN=$(command -v docker >/dev/null && docker compose version >/dev/null 2>&1 && echo docker || echo podman)
command -v "$RUN" >/dev/null || RUN=docker
KAFKA_PORT=${HYPERSPLAT_KAFKA_PORT:-9094}
DB_PORT=${HYPERSPLAT_DB_PORT:-5433}
KAFKA_IMAGE=docker.io/apache/kafka:3.9.1
DB_IMAGE=docker.io/library/postgres:16-alpine

running() { [[ "$($RUN inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" == "true" ]]; }
port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

up() {
    if ! running hs-kafka; then
        $RUN rm -f hs-kafka >/dev/null 2>&1 || true
        $RUN run -d --name hs-kafka -p "$KAFKA_PORT:9094" \
            -e KAFKA_NODE_ID=1 -e KAFKA_PROCESS_ROLES=broker,controller \
            -e KAFKA_LISTENERS=INTERNAL://:29092,EXTERNAL://:9094,CONTROLLER://:9093 \
            -e KAFKA_ADVERTISED_LISTENERS="INTERNAL://hs-kafka:29092,EXTERNAL://localhost:$KAFKA_PORT" \
            -e KAFKA_LISTENER_SECURITY_PROTOCOL_MAP=INTERNAL:PLAINTEXT,EXTERNAL:PLAINTEXT,CONTROLLER:PLAINTEXT \
            -e KAFKA_CONTROLLER_LISTENER_NAMES=CONTROLLER -e KAFKA_INTER_BROKER_LISTENER_NAME=INTERNAL \
            -e KAFKA_CONTROLLER_QUORUM_VOTERS=1@localhost:9093 \
            -e KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1 -e KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR=1 \
            -e KAFKA_TRANSACTION_STATE_LOG_MIN_ISR=1 -e KAFKA_GROUP_INITIAL_REBALANCE_DELAY_MS=0 \
            "$KAFKA_IMAGE" >/dev/null
    fi
    if ! running hs-db; then
        $RUN rm -f hs-db >/dev/null 2>&1 || true
        $RUN run -d --name hs-db -p "$DB_PORT:5432" -e POSTGRES_USER=hypersplat \
            -e POSTGRES_PASSWORD=hypersplat -e POSTGRES_DB=hypersplat "$DB_IMAGE" >/dev/null
    fi
    for _ in $(seq 60); do
        port_open "$KAFKA_PORT" && $RUN exec hs-db pg_isready -U hypersplat -q 2>/dev/null && break
        sleep 1
    done
    for db in "$@"; do
        $RUN exec hs-db psql -U hypersplat -tAc "SELECT 1 FROM pg_database WHERE datname='$db'" | grep -q 1 \
            || $RUN exec hs-db psql -U hypersplat -qc "CREATE DATABASE \"$db\""
    done
    status
}

status() {
    for c in hs-kafka hs-db; do printf '%-9s %s\n' "$c" "$(running "$c" && echo running || echo stopped)"; done
    printf 'kafka  localhost:%s %s\n' "$KAFKA_PORT" "$(port_open "$KAFKA_PORT" && echo open || echo closed)"
    printf 'db     localhost:%s %s\n' "$DB_PORT" "$(port_open "$DB_PORT" && echo open || echo closed)"
}

case "${1:-status}" in
    up) shift; up "$@" ;;
    status) status ;;
    down) $RUN rm -f hs-kafka hs-db ;;
    *) sed -n '2,8p' "$0"; exit 1 ;;
esac
