FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /

# Install Python dependencies (Worker Template)
COPY builder/requirements.txt /requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /requirements.txt && \
    rm /requirements.txt

# Add src files (Worker Template)
ADD src .

CMD [ "python", "-u", "/rp_handler.py" ]
