ARG ALPINE_VERSION=latest
FROM alpine:${ALPINE_VERSION}

RUN mkdir ./NSGA

RUN set -eux && \
    sed -i 's/dl-cdn.alpinelinux.org/mirrors.ustc.edu.cn/g' /etc/apk/repositories && \
    apk add py3-numpy


COPY /data /NSGA/data
COPY /src /NSGA/src

WORKDIR /NSGA

ENTRYPOINT ["python3","./src/NSGA_II.py"]
CMD ["-ipd","./data/input.csv","-mag","100","-opd","./data/naga_output.csv"]
