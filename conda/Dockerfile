FROM mambaorg/micromamba
COPY --chown=micromamba:micromamba xarray-behave xarray-behave
COPY --chown=micromamba:micromamba xarray-behave-nogui xarray-behave-nogui

# USER root
# RUN apt update
# RUN apt install ffmpeg libsm6 libxext6 libgl1-mesa-glx -y
# USER 1001

RUN micromamba install -y -n base -c conda-forge \
       python=3.8 \
       boa anaconda-client conda-build conda-verify
RUN micromamba clean --all --yes
RUN conda config --set anaconda_upload yes

RUN conda mambabuild xarray-behave-nogui -c conda-forge -c ncb -c anaconda --python 3.7 --user ncb
RUN conda mambabuild xarray-behave -c conda-forge -c ncb -c anaconda --python 3.7 --user ncb
RUN conda mambabuild xarray-behave-nogui -c conda-forge -c ncb -c anaconda --python 3.8 --user ncb
RUN conda mambabuild xarray-behave -c conda-forge -c ncb -c anaconda --python 3.8 --user ncb
RUN conda mambabuild xarray-behave-nogui -c conda-forge -c ncb -c anaconda --python 3.9 --user ncb
RUN conda mambabuild xarray-behave -c conda-forge -c ncb -c anaconda --python 3.9 --user ncb
