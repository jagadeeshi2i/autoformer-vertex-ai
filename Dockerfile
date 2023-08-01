FROM pytorch/pytorch:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update  -y --fix-missing && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    curl \
    unrar \
    unzip \
    git && \
    apt-get upgrade -y libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Create user and set ownership and permissions as required
RUN useradd -ms /bin/bash demo
RUN mkdir /home/demo/app/ && chown -R demo:demo /home/demo/app
WORKDIR /home/demo/app/
USER demo

ENV PATH="/home/demo/.local/bin:${PATH}"

COPY --chown=demo:demo . .

RUN pip install --user --requirement requirements.txt

CMD [ "/bin/bash"]