FROM alpine:3.13

COPY data /app/data
COPY src /app/src

WORKDIR /app/src

RUN set -eux && \
    sed -i 's/dl-cdn.alpinelinux.org/mirrors.ustc.edu.cn/g' /etc/apk/repositories && \
	apk add py3-scikit-learn

ENTRYPOINT ["python3", "./svm.py"]
CMD ["-C", "1.0"]
