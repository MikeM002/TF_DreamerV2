@echo off
REM Salir al directorio anterior
cd ..

REM Construir la imagen de Docker nuevamente
docker build -t dreamerv2 .

REM Volver a entrar al directorio de dreamerv2
cd dreamerv2

REM Ejecutar el contenedor Docker con el comando de entrenamiento
docker run -it --rm --gpus all -v %USERPROFILE%/logdir:/logdir dreamerv2 python3 dreamerv2/train.py --logdir /logdir/atari_pong/dreamerv2/1 --configs atari --task atari_ms_pacman
