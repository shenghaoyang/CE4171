FROM archlinux:latest
RUN pacman -Syu --noconfirm python python-pip sqlite && \
    pacman -Scc --noconfirm
RUN pip install
COPY server_container.ini /etc/dlserver/server.ini
COPY train/data/mini_speech_commands /usr/lib/dlserver/initial_data
COPY train/saved/audiorecog /usr/lib/dlserver/initial_model
RUN mkdir /var/lib/dlserver /var/lib/dlserver/uploaded/training \
          /var/lib/dlserver/uploaded/infer
