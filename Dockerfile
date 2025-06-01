FROM python:3.11

# Install system packages required for scientific computing and PDF export
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libatlas-base-dev \
    libglib2.0-0 \
    libxrender1 \
    libsm6 \
    libxext6 \
    wget \
    git \
    texlive-xetex \
    texlive-fonts-recommended \
    texlive-plain-generic \
    pandoc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python packages
RUN pip install --upgrade pip \
    && pip install --no-cache-dir \
        mne \
        PyWavelets \
        numpy \
        matplotlib \
        pandas \
        pyarrow \
        scipy \
        antropy \
        seaborn \
        jupyter

# Expose Jupyter port
EXPOSE 8888

# Run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
