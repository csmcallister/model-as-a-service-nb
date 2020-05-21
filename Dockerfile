FROM tensorflow/tensorflow:2.2.0-gpu

# deps for opencv
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev

# install the SageMaker Training Toolkit 
RUN pip3 install sagemaker-training

# install our dependencies
RUN pip3 install keras_ocr scikit-learn

# copy the training script inside the container
COPY train.py /opt/ml/code/train.py

# define train.py as the script entry point
ENV SAGEMAKER_PROGRAM train.py