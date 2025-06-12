FROM rayproject/ray:2.41.0

RUN sudo apt update -y && sudo apt install -y mc iproute2 curl
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install mlflow==2.22.1
