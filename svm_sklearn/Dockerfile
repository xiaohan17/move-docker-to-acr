FROM alpine:3.13

COPY data /app/data
COPY src /app/src

WORKDIR /app/src

RUN set -eux && \
    sed -i 's/dl-cdn.alpinelinux.org/mirrors.ustc.edu.cn/g' /etc/apk/repositories && \
	apk add py3-scikit-learn

ENTRYPOINT ["python3", "./svm.py"]
CMD ["-train_data","../data/train_data.csv",
    "-train_label","../data/train_label.csv",
    "-test_data","../data/test_data.csv",
    "-test_label","../data/test_label.csv",
    "-C", "1.0",
    "-output","./c_support_vector_classification.csv"]
