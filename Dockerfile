FROM python:3.9.19-slim

####################################
# To enable CUDA from the container (at least on our machines, it might be different in other settings)
ENV PATH /usr/local/cuda/bin/:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib:/usr/local/cuda/lib64
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
LABEL com.nvidia.volumes.needed="nvidia_driver"
#####################################

RUN pip install --no-cache-dir torch==2.2.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 44334

COPY . .

CMD ["fastapi", "run", "main.py", "--port", "44334"]