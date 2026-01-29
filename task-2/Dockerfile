FROM apache/nifi:2.5.0

USER root

# install python
RUN apt-get update && \
    apt-get install -y python3 python3-venv python3-pip && \
    apt-get clean

# copy requirements
COPY requirements.txt /tmp/requirements.txt

# create venv + install deps
RUN python3 -m venv /opt/python-venv && \
    /opt/python-venv/bin/pip install --upgrade pip && \
    /opt/python-venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt

# permissions
RUN chown -R nifi:nifi /opt/python-venv

USER nifi
