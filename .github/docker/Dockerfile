FROM ubuntu:24.04

LABEL org.opencontainers.image.source=https://github.com/Kuree/bflang
LABEL org.opencontainers.image.description="Container image for bflang"
LABEL org.opencontainers.image.licenses=MIT

ARG BUILD_DIR

RUN apt-get update && \
    apt-get install -y --no-install-recommends llvm-18 && \
    apt-get clean

COPY ${BUILD_DIR}/install /usr/local/bflang

ENV PATH=${PATH}:/usr/local/bflang/bin
