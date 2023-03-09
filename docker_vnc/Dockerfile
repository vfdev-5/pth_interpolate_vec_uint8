FROM intel/oneapi-basekit

COPY start.bash /start.bash

RUN apt-get update && ln -fs /usr/share/zoneinfo/Europe/Paris /etc/localtime && \
    apt-get install -y tzdata && \
    dpkg-reconfigure --frontend noninteractive tzdata

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y x11vnc xvfb \
    && apt install --no-install-recommends -y ubuntu-desktop \
    && apt install --no-install-recommends -y supervisor tini \
    && chmod +x /start.bash \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt install --no-install-recommends -y lxpanel openbox openbox-lxde-session

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

ENTRYPOINT ["/start.bash"]