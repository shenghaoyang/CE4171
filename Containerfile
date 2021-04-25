FROM docker.io/archlinux:latest
ARG wheel=dlserver-0.0.4-py2.py3-none-any.whl
LABEL maintainer="me@shenghaoyang.info"

RUN pacman -Syu --noconfirm python python-pip python-wheel sqlite && \
    pacman -Scc --noconfirm
COPY server_container.ini /etc/dlserver/server.ini
COPY train/data/mini_speech_commands /usr/lib/dlserver/initial_data
COPY train/saved/audiorecog /usr/lib/dlserver/initial_model
RUN mkdir -p /var/lib/dlserver /var/lib/dlserver/uploaded/training \
             /var/lib/dlserver/uploaded/infer
COPY dist/${wheel} /tmp
RUN pip install --no-cache-dir /tmp/${wheel}
RUN rm /tmp/${wheel}

EXPOSE 55221/tcp
VOLUME ["/var/lib/dlserver", "/etc/dlserver/"]
ENTRYPOINT ["/bin/dlserver", "/etc/dlserver/server.ini"]
