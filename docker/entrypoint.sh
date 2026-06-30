#!/usr/bin/env bash
# Entrypoint for the ATLAS orchestration container.
#
# Waits for PostgreSQL and RabbitMQ, then (on first run) creates an AiiDA
# profile pointing at those services and starts the daemon. Idempotent: an
# existing profile is reused.
set -euo pipefail

POSTGRES_HOST="${POSTGRES_HOST:-postgres}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_USER="${POSTGRES_USER:-aiida}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-aiida_pw}"
RABBITMQ_HOST="${RABBITMQ_HOST:-rabbitmq}"
RABBITMQ_PORT="${RABBITMQ_PORT:-5672}"
PROFILE_NAME="${ATLAS_AIIDA_PROFILE:-atlas}"

wait_for() {
    host="$1"; port="$2"; name="$3"
    echo "[atlas] Waiting for ${name} at ${host}:${port}..."
    for _ in $(seq 1 60); do
        if nc -z "${host}" "${port}" 2>/dev/null; then
            echo "[atlas] ${name} is up."
            return 0
        fi
        sleep 2
    done
    echo "[atlas] WARNING: ${name} not reachable at ${host}:${port}." >&2
    return 1
}

wait_for "${POSTGRES_HOST}" "${POSTGRES_PORT}" "PostgreSQL" || true
wait_for "${RABBITMQ_HOST}" "${RABBITMQ_PORT}" "RabbitMQ" || true

if verdi profile show "${PROFILE_NAME}" >/dev/null 2>&1; then
    echo "[atlas] AiiDA profile '${PROFILE_NAME}' already configured."
else
    echo "[atlas] Creating AiiDA profile '${PROFILE_NAME}'..."
    # verdi presto auto-detects RabbitMQ only on localhost, so create the
    # profile against PostgreSQL first, then point the broker at the rabbitmq
    # service explicitly.
    verdi presto \
        --profile-name "${PROFILE_NAME}" \
        --use-postgres \
        --postgres-hostname "${POSTGRES_HOST}" \
        --postgres-port "${POSTGRES_PORT}" \
        --postgres-username "${POSTGRES_USER}" \
        --postgres-password "${POSTGRES_PASSWORD}"

    verdi profile configure-rabbitmq "${PROFILE_NAME}" \
        --broker-host "${RABBITMQ_HOST}" \
        --broker-port "${RABBITMQ_PORT}" \
        --force

    verdi profile setdefault "${PROFILE_NAME}" || true
fi

# Start the daemon (best-effort; harmless if already running).
verdi daemon start 2>/dev/null || true

verdi status || true

exec "$@"
