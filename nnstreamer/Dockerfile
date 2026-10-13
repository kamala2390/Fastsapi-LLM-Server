# Ubuntu + NNStreamer environment for running the bridge pipeline on a machine
# without a native NNStreamer package (e.g. macOS). See README.md.
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository -y ppa:nnstreamer/ppa \
    && apt-get update && apt-get install -y \
       nnstreamer nnstreamer-python3 \
       gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
       python3-numpy python3-pil python3-requests \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /work
