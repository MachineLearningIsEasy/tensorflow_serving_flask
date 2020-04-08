# Сборка основного контейнера
docker-compose up --build

#Start tensorflow serving
docker run -p 8501:8501 --name tfserving_document --mount type=bind,source=/root/models/document,target=/models/document -e MODEL_NAME=document -t tensorflow/serving &

#Параметры модели на tensorflow serving
saved_model_cli show --dir /root/models/document --all