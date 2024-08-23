##
# registry.cn-hangzhou.aliyuncs.com/onesis-geomodels/jdk:17-alpine
#
# From: https://github.com/nirmata/tomcat9-jre17-alpine/blob/main/Dockerfile
#

# Use alpine as the build container
ARG ALPINE_VERSION=3.15
FROM alpine:${ALPINE_VERSION} AS builder

LABEL maintainer="Liang-Jun Zhu <zlj@lreis.ac.cn>"

# Replace alpine repository source cdn to accelarate access speed; Add openjdk17
# RUN sed -i 's/dl-cdn.alpinelinux.org/mirrors.aliyun.com/g' /etc/apk/repositories && \

# For GitHub Actions, there is no need to replace alpine repository source cdn!
RUN apk update && apk upgrade && \
    apk add openjdk17
