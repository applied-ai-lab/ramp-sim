FROM nvcr.io/nvidia/isaac-sim:2022.2.1
WORKDIR /app
RUN DEBIAN_FRONTEND=noninteractive apt-get -y update \
    && DEBIAN_FRONTEND=noninteractive apt-get -y install \
    default-jre git \
    software-properties-common \
    && DEBIAN_FRONTEND=noninteractive add-apt-repository  ppa:potassco/stable \
    && DEBIAN_FRONTEND=noninteractive apt -y update \
    && DEBIAN_FRONTEND=noninteractive apt -y install clingo
RUN git clone https://github.com/iensen/sparc.git
ENV SPARC_PATH=/app/sparc
COPY . /app
ENV PLANNER_PATH=/app/planner
ENV PYTHON_PATH=/isaac-sim/python.sh
ENV PYTHONPATH="${PYTHONPATH}:/app/planner"
RUN cd planner \
    && /isaac-sim/python.sh -m pip install -r requirements.txt \
    && /isaac-sim/python.sh setup.py build \
    && /isaac-sim/python.sh setup.py install

