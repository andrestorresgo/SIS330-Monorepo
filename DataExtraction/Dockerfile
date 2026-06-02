FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    vitaldb \
    pandas \
    scipy \
    matplotlib \
    jupyter \
    notebook

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
