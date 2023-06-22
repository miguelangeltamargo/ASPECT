FROM ubuntu

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common \
    gcc \
    git \
    python3 \
    python3-distutils \
    python3-pip \
    python3-apt \
    nvidia-cuda-toolkit \
    nano \
    bedtools

RUN add-apt-repository -y ppa:deadsnakes/ppa

RUN pip3 install \
    pandas \
    numpy \
    tensorflow \
    keras \
    scikit-learn \
    pybedtools

RUN echo "alias python='python3'" >> ~/.bashrc

EXPOSE 8888

# Set the working directory
WORKDIR /aspect

# Clone the first repository
# RUN git clone https://github.com/miguelangeltamargo/ASPECT.git /aspect/home

# Clone the second repository
# RUN git clone https://github.com/louadi/DSC.git /aspect/dsc

# Start the Jupyter Notebook server
# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
