FROM igormcsouza/ml:scala-spark-python3-polynote

COPY requirements.txt /opt/requirements.txt
RUN pip3 install -r /opt/requirements.txt

WORKDIR /age-detection/

COPY scripts/ /age-detection/
COPY data/ /data/

CMD ["/bin/bash"]
