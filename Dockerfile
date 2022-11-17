# FROM rocker/r-ubuntu:18.04
FROM continuumio/anaconda3

LABEL MAINTAINER CBJ <choi328328@naver.com> 

RUN mkdir -p /home/client/FederatedPLP
COPY . /home/client/FederatedPLP

## install java
RUN apt-get update && \
	apt-get install -y default-jdk wget apt-utils vim ssh git build-essential r-base libcurl4-gnutls-dev libxml2-dev \
	libssl-dev libgit2-dev libfontconfig1-dev libcairo2-dev libsnappy-dev
RUN R CMD javareconf
RUN R -e "install.packages('rJava', dependencies = TRUE)"

RUN R -e "install.packages('dplyr')"
RUN R -e "install.packages('arrow')"
RUN R -e "install.packages('shiny')"
RUN R -e "install.packages('shinycssloaders')"
RUN R -e "install.packages('shinydashboard')"
RUN R -e "install.packages('R.utils')"
RUN R -e "install.packages('rvg')"
RUN R -e "install.packages('remotes')"

# # # install PLP Packages 
RUN R -e 'remotes::install_github("OHDSI/OhdsiSharing",ref="v0.2.2")'
RUN R -e 'remotes::install_github("OHDSI/FeatureExtraction",ref="v3.1.0")'
RUN R -e 'remotes::install_github("OHDSI/PatientLevelPrediction",ref="v4.0.5", upgrade="never")'

# # install JDBC driver
RUN mkdir -p /home/client/jdbc 
RUN R -e "DatabaseConnector::downloadJdbcDrivers(dbms = 'oracle',pathToDriver = '/home/client/jdbc')"
RUN R -e "DatabaseConnector::downloadJdbcDrivers(dbms = 'postgresql',pathToDriver = '/home/client/jdbc')"
RUN R -e "DatabaseConnector::downloadJdbcDrivers(dbms = 'sql server',pathToDriver = '/home/client/jdbc')"

WORKDIR /home/client
RUN conda install -y pytorch=1.12.1 torchvision torchaudio cpuonly -c pytorch 
RUN pip3 install deepctr_torch pytorch-lightning==1.3.6 black mypy loguru pyvacy opacus flwr==1.0 pyarrow torchmetrics==0.6.0 && \
	pip3 install hydra-core --upgrade --pre && \ 
	pip3 install -U scikit-learn



