# The statistics sandbox: the guardrail's analyses (analysis.py) behind a JSON entry point
# (runner.py). Built by scripts/70_build_sandbox.py, which tags the image with a hash of this
# directory, and run by src/stats/sandbox.py with no network, a read-only file system, a small
# scratch directory, a non-root user, no capabilities and limits on time, memory and processes.
#
# The base image is pinned by digest, and every package by version and hash
# (requirements.txt, exported from the project's lock file: the `sandbox` dependency group).
FROM python:3.12-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f

# One thread for every numerical library, so sums are added in the same order on every run.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    HOME=/scratch \
    TMPDIR=/scratch

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --no-deps --require-hashes --only-binary=:all: \
        -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt \
    && useradd --system --uid 10001 --user-group --no-create-home \
        --shell /usr/sbin/nologin sandbox

COPY runner.py analysis.py /opt/sandbox/
RUN python -m compileall -q /opt/sandbox

WORKDIR /scratch
USER 10001:10001
ENTRYPOINT ["python", "-I", "-B", "/opt/sandbox/runner.py"]
