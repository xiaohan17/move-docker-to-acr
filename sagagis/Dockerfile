##
# registry.cn-hangzhou.aliyuncs.com/onesis-geomodels/sagagis-base:alpine-8.1.0
#
# Copyright 2022 Liang-Jun Zhu <zlj@lreis.ac.cn>
##

# Use alpine as the build container
ARG ALPINE_VERSION=3.16
FROM alpine:${ALPINE_VERSION} AS builder

LABEL maintainer="Liang-Jun Zhu <zlj@lreis.ac.cn>"

# Copy source directory
WORKDIR /saga

ARG VERSION=8.1.0

RUN wget https://github.com/saga-gis/saga-gis/archive/refs/tags/v${VERSION}.tar.gz -O saga-gis.tar.gz -c

RUN tar xzf saga-gis.tar.gz -C /saga

RUN mv /saga/saga-gis-${VERSION}/saga-gis /saga

# Replace alpine repository source cdn to accelarate access speed;
# Compile SAGA-GIS base, without GUI, without Python binding...
#RUN sed -i 's/dl-cdn.alpinelinux.org/mirrors.aliyun.com/g' /etc/apk/repositories \
RUN apk update && apk upgrade \
    && apk add cmake make g++ libgomp gdal-dev proj-dev vigra-dev wxgtk-dev \
    && cd /saga/saga-gis && mkdir build && mkdir dist && cd build \
    && cmake .. -DWITH_GUI=OFF \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    && make -j8 && make install

# # Build final image
FROM alpine:${ALPINE_VERSION} AS final

# Replace alpine repository source cdn; Add SAGA-GIS runtime libraries
# RUN sed -i 's/dl-cdn.alpinelinux.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apk/repositories
#RUN sed -i 's/dl-cdn.alpinelinux.org/mirrors.aliyun.com/g' /etc/apk/repositories \
RUN apk update && apk upgrade \
    && apk add --no-cache libgomp gdal proj vigra wxgtk

# Copy compiled SAGA-GIS to the final image
COPY --from=builder /usr/local /usr/local

ENV SAGA_TLB=/usr/local/lib/saga
ENV LD_LIBRARY_PATH=/usr/local/lib

ENTRYPOINT ["saga_cmd"]
