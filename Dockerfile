# Use a valid CUDA base image
FROM nvidia/cuda:11.8.0-base-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=xterm
ENV LC_ALL=C.UTF-8
ENV PATH="/home/docker/.local/bin:$PATH"

# Install required packages
RUN apt-get update && apt-get install -y \
    zsh \
    software-properties-common \
    gcc \
    git \
    python3 \
    python3-distutils \
    python3-pip \
    python3-apt \
    nano \
    neovim \
    bedtools \
    screen \
    curl \
    sudo && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install NVIDIA Container Toolkit
RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && \
    distribution=ubuntu22.04 && \
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb #deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] #' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list && \
    apt-get update && apt-get install -y nvidia-container-toolkit && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Install Python packages
RUN pip3 install --no-cache-dir \
    pandas \
    numpy \
    torch \
    keras \
    scikit-learn \
    pybedtools \
    wandb \
    transformers \
    seaborn
RUN pip3 install --no-cache-dir transformers[torch]

# Add a non-root user and give it sudo privileges
RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo && \
    echo "docker ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Switch to the new user
USER docker

# Create necessary files for the non-root user
RUN touch ~/.zshrc && touch ~/.bashrc

# Install Oh My Zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# Install Powerlevel10k theme
RUN git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/themes/powerlevel10k && \
    sed -i 's/ZSH_THEME="robbyrussell"/ZSH_THEME="powerlevel10k\/powerlevel10k"/g' ~/.zshrc

# Set Python alias
RUN echo "alias python='python3'" >> ~/.bashrc

# Expose port 8888
EXPOSE 8888

# Default command
CMD ["/bin/zsh"]
