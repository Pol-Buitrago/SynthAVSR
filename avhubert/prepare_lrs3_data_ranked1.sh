#!/bin/bash
#SBATCH --job-name=lrs3_prepare         # Nombre de la tarea
#SBATCH --account=bsc88       
#SBATCH -t 2-00:00:00                   # Tiempo (ajustar según el tamaño de los datos)
#SBATCH -q acc_bscls                    # Quality of Service para GPUs
#SBATCH -c 80                           # CPUs por tarea
#SBATCH --gres=gpu:4                    # GPUS
#SBATCH --ntasks=1                      # Número de tareas
#SBATCH -N 1                            # Número de nodos
#SBATCH --output=/gpfs/projects/bsc88/speech/research/repos/av_hubert/output/lrs3_prepare_%j.log
#SBATCH --error=/gpfs/projects/bsc88/speech/research/repos/av_hubert/error/lrs3_prepare_%j.err

###########################################################################################################################
########################################### LRS3 data preparation script ##################################################
###########################################################################################################################

# Este script realiza todos los pasos necesarios para preparar los archivos de manifiesto (*.tsv, *.wrd)
# y los archivos necesarios para realizar inferencias con AV-HuBERT en el conjunto de datos LRS3.

# Activar el entorno conda donde tienes las dependencias
source activate avhubert

# Rutas base para los archivos
DATA_DIR="/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/data"

# Variables - Editar rutas según el entorno
LRS3_DIR="$DATA_DIR/datasets/dataset_EN/lrs3"                                                   # Directorio donde están almacenados los datos LRS3 (contiene pretrain, trainval y test)
FFMPEG_PATH="/gpfs/home/bsc/bsc915220/.conda/envs/avhubert/bin/ffmpeg"                          # Ruta hacia ffmpeg 
DLIB_CNN_DETECTOR="$DATA_DIR/misc/mmod_human_face_detector.dat"             # Ruta al detector CNN
DLIB_FACE_DETECTOR="$DATA_DIR/misc/shape_predictor_68_face_landmarks.dat"   # Ruta al predictor de landmarks
MEAN_FACE_PATH="$DATA_DIR/misc/20words_mean_face.npy"                       # Ruta hacia el archivo mean_face.npy (descargado del repo de AV-HuBERT)

N_SHARD=10       # Número de shards (partes en que dividimos los datos para procesamiento)
RANK=2          # Índice de shard a procesar (0 si solo usas un shard)



# Paso 1: Preparación de los datos
# -----------------------------------------------------------------------------
# Ejecuta el script lrs3_prepare.py para generar la lista de archivos y transcripciones
# -----------------------------------------------------------------------------
: <<'EOF'
echo "=== Paso 1: Preparación de los datos ==="

# Bucle para ejecutar cada paso
for step in {1..4}; do
  echo "Ejecutando paso $step"
  python preparation/lrs3_prepare.py \
    --lrs3 "$LRS3_DIR" \
    --ffmpeg "$FFMPEG_PATH" \
    --rank $RANK \
    --nshard $N_SHARD \
    --step $step
done
EOF

# Explicación:
# --lrs3: Ruta hacia los datos LRS3.
# --ffmpeg: Ruta hacia ffmpeg para procesar los videos.
# --rank: Índice de shard (0 en este caso, ya que no estamos dividiendo los datos en varios shards).
# --nshard: Número de shards (1 significa que procesaremos todos los datos de una sola vez).
# --step: Especifica que queremos generar los archivos file.list y label.list (paso 4).

# Los archivos file.list y label.list se generan en el directorio LRS3_DIR.

# Paso 2: Detección de landmarks faciales y recorte de la región de la boca (ROI)
# -----------------------------------------------------------------------------
# Ejecuta los scripts detect_landmark.py y align_mouth.py para detectar landmarks
# faciales y recortar la región de la boca (ROI) en los vídeos.
# -----------------------------------------------------------------------------

echo "=== Paso 2: Detección de landmarks faciales ==="
python preparation/detect_new_landmark.py \
  --root "$LRS3_DIR" \
  --landmark "$LRS3_DIR/landmark" \
  --manifest "$LRS3_DIR/file.list" \
  --cnn_detector "$DLIB_CNN_DETECTOR" \
  --face_predictor "$DLIB_FACE_DETECTOR" \
  --ffmpeg "$FFMPEG_PATH" \
  --rank $RANK \
  --nshard $N_SHARD

# Explicación:
# --root: Ruta raíz donde están almacenados los datos de LRS3.
# --landmark: Directorio donde se almacenarán los landmarks faciales detectados.
# --manifest: Archivo file.list generado en el paso anterior.
# --cnn_detector: Ruta hacia el modelo CNN de detección de rostros de dlib.
# --face_detector: Ruta hacia el predictor de landmarks faciales de dlib.
# --ffmpeg: Ruta hacia ffmpeg.
# --rank y --nshard: Similar a los valores anteriores; procesamos solo un shard.

echo "=== Paso 2.1: Alineación de la región de la boca (ROI) ==="
python preparation/align_mouth.py \
  --video-direc "$LRS3_DIR" \
  --landmark "$LRS3_DIR/landmark" \
  --filename-path "$LRS3_DIR/file.list" \
  --save-direc "$LRS3_DIR/video" \
  --mean-face "$MEAN_FACE_PATH" \
  --ffmpeg "$FFMPEG_PATH" \
  --rank $RANK \
  --nshard $N_SHARD

# Explicación:
# --video-direc: Directorio de videos LRS3.
# --landmark: Directorio donde se almacenaron los landmarks.
# --filename-path: Archivo file.list con los IDs de los videos.
# --save-direc: Directorio donde se almacenarán las regiones de la boca (ROI).
# --mean-face: Ruta al archivo mean_face.npy necesario para la alineación.
# --ffmpeg: Ruta hacia ffmpeg.
# --rank y --nshard: Procesamos un solo shard (0 de 1).

# Paso 3: Contar el número de frames por clip
# -----------------------------------------------------------------------------
# Ejecuta count_frames.py para contar el número de frames de audio y vídeo por clip.
# Los resultados se guardarán en los archivos nframes.audio y nframes.video.
# -----------------------------------------------------------------------------

echo "=== Paso 3: Conteo de frames por clip ==="
python preparation/count_frames.py \
  --root "$LRS3_DIR" \
  --manifest "$LRS3_DIR/file.list" \
  --nshard $N_SHARD \
  --rank $RANK

# Explicación:
# --root: Directorio raíz de LRS3.
# --manifest: Archivo file.list generado anteriormente.
# --nshard: Número de shards.
# --rank: Índice del shard a procesar (0 de 1).

# Una vez que todos los shards han sido procesados (en este caso solo 1), puedes combinar los resultados:

echo "=== Paso 3.1: Combinando resultados de los frames ==="
cat "$LRS3_DIR/nframes.audio.$RANK" > "$LRS3_DIR/nframes.audio"
cat "$LRS3_DIR/nframes.video.$RANK" > "$LRS3_DIR/nframes.video"

# Explicación:
# Combina los resultados del conteo de frames para audio y vídeo. Como solo procesamos un shard,
# los resultados ya están completos en nframes.audio y nframes.video.

# Paso 4: Generación de archivos de manifiesto (train.tsv, valid.tsv, test.tsv) y transcripciones (*.wrd)
# -----------------------------------------------------------------------------
# Ejecuta lrs3_manifest.py para crear los archivos *.tsv y *.wrd, necesarios para la inferencia.
# -----------------------------------------------------------------------------

VOCAB_SIZE=1000  # Ajusta el tamaño del vocabulario según tu necesidad
VALID_IDS="/ruta/a/valid_ids"  # Ruta al archivo con los IDs de clips reservados para validación (puedes crear uno si es necesario)

echo "=== Paso 4: Generación de manifiestos y transcripciones ==="
python lrs3_manifest.py \
  --lrs3 "$LRS3_DIR" \
  --manifest "$LRS3_DIR/file.list" \
  --valid-ids "$VALID_IDS" \
  --vocab-size $VOCAB_SIZE

# Explicación:
# --lrs3: Directorio raíz de los datos LRS3.
# --manifest: Archivo file.list generado anteriormente.
# --valid-ids: Archivo que contiene los IDs de clips reservados para la validación.
# --vocab-size: Tamaño del vocabulario (por ejemplo, 1000 o 2000 palabras).

# Los archivos train.tsv, valid.tsv, test.tsv, train.wrd, valid.wrd y test.wrd se generarán
# en el directorio LRS3_DIR.

echo "Preparación de datos LRS3 completada con éxito."
