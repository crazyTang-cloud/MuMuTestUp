FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MAVEN_VERSION=3.9.6

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    git \
    gnupg \
    procps \
    unzip \
    wget \
    xz-utils \
    zip \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt /data/david/maven /data/david/java /data/david/maven_repo /data/david/project/mumutestup

RUN curl -fsSL "https://archive.apache.org/dist/maven/maven-3/${MAVEN_VERSION}/binaries/apache-maven-${MAVEN_VERSION}-bin.tar.gz" \
    | tar -xz -C /opt \
    && ln -sfn "/opt/apache-maven-${MAVEN_VERSION}" "/data/david/maven/apache-maven-3.9.6"

ENV MAVEN_HOME=/data/david/maven/apache-maven-3.9.6
ENV PATH="${MAVEN_HOME}/bin:${PATH}"

WORKDIR /data/david/project/mumutestup

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x /data/david/project/mumutestup/entrypoint.sh

ENTRYPOINT ["/data/david/project/mumutestup/entrypoint.sh"]
CMD ["python", "run_complete_beam.py"]
