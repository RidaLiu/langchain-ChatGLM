FROM python:3.8
# FROM registry.baidubce.com/paddlepaddle/paddle:2.4.2

MAINTAINER "chatGLM"

COPY agent /chatGLM/agent

COPY chains /chatGLM/chains

COPY configs /chatGLM/configs

COPY content /chatGLM/content

COPY models /chatGLM/models

COPY nltk_data /chatGLM/content

COPY requirements.txt /chatGLM/

COPY cli_demo.py /chatGLM/

COPY textsplitter /chatGLM/

COPY webui.py /chatGLM/

WORKDIR /chatGLM

RUN pip --default-timeout=500 install --user torch torchvision tensorboard cython 
# RUN pip install --user 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'
# install detectron2
# RUN git clone https://github.com/facebookresearch/detectron2
RUN pip install --upgrade pip
RUN pip --default-timeout=500 install -r requirements.txt 

RUN apt-get update && apt-get -y install libgl1

RUN pip install accelerate

CMD ["python","-u", "webui.py"]
