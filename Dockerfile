FROM python:3.11-slim-bookworm

WORKDIR /app

RUN pip install -U pip

COPY requirements.txt .

RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "main.py" ]
