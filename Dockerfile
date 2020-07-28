FROM debian:buster

ENV DEBIAN_FRONTEND=noninteractive
ENV TF_CPP_MIN_LOG_LEVEL=2


# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    git \
    gnupg \
    python3 \
    python3-dev \
    python3-pip && \
    python3 -m pip install --upgrade \
        pip \
        setuptools \
        wheel 

# Install the Edge TPU runtime
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" > /etc/apt/sources.list.d/coral-edgetpu.list && \
    wget -q -O - https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        libedgetpu1-std \
        python3-edgetpu

EXPOSE 5000

COPY . /opt/cusca

WORKDIR /opt/cusca

RUN python3 -m pip install -r requirements.txt 

RUN sh install_models.sh

CMD ["python3", "app.py"]
