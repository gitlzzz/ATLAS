# ATLAS orchestration image.
#
# This image runs the ATLAS workflow/orchestration layer (AiiDA + the atl_*
# CLIs). The heavy MACE/VASP calculations themselves run on compute nodes via
# AiiDA's ContainerizedCode (see container_settings in the config), not inside
# this image.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    QT_QPA_PLATFORM=offscreen \
    QT_NO_DBUS=1 \
    AIIDA_PATH=/atl_data/.aiida

# System libraries: BLAS/LAPACK for numpy/scipy/torch, the PostgreSQL client and
# netcat for the AiiDA database wait/setup, ssh for remote HPC codes, git for the
# setuptools-git-versioning version string.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        libopenblas-dev \
        liblapack-dev \
        postgresql-client \
        openssh-client \
        netcat-openbsd \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/atlas
COPY . /opt/atlas

# Core install by default. Build with --build-arg INSTALL_MACE=true to also pull
# the MACE extra (large; includes torch + cuequivariance, CUDA-oriented).
ARG INSTALL_MACE=false
RUN python -m pip install --upgrade pip setuptools wheel \
    && if [ "$INSTALL_MACE" = "true" ]; then \
         pip install ".[mace]"; \
       else \
         pip install .; \
       fi

RUN mkdir -p /atl_data
WORKDIR /atl_data

COPY docker/entrypoint.sh /usr/local/bin/atlas-entrypoint
RUN chmod +x /usr/local/bin/atlas-entrypoint

# Flask monitoring dashboard (atl_monitor_al_loop) default port.
EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/atlas-entrypoint"]
CMD ["verdi", "status"]
