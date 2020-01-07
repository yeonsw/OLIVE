EXEC_NAME=w2v
gcc ./code/olive.c -o "${EXEC_NAME}" -lm -g -pthread -O2

DIMENSION=100
LR=0.5
WINDOW=5
SAMPLE=1e-5
K=50.0
MINC=2
THREADS=28
ITER=200
MINN=2
MAXN=7
CACHE=0

DIR=./data/wiki/
CORPUS="${DIR}"wiki_small.txt
VOCAB="${DIR}"wiki_small_vocab.txt
COOCC="${DIR}"wiki_small_co.txt
VECTOR_DIR="${DIR}"vectors/

mkdir "${VECTOR_DIR}"
VEC_U="${VECTOR_DIR}"vector_u.bin
VEC_V="${VECTOR_DIR}"vector_v.bin
VEC_ZU="${VECTOR_DIR}"vector_zu.bin
VEC_ZV="${VECTOR_DIR}"vector_zv.bin

./"${EXEC_NAME}" -corpus "${CORPUS}" -vocab-fname "${VOCAB}" -co-occ-fname "${COOCC}" -vector-u-fname "${VEC_U}" -vector-v-fname "${VEC_V}" -vector-zu-fname "${VEC_ZU}" -vector-zv-fname "${VEC_ZV}" -dimension $DIMENSION -lr $LR -window $WINDOW -min-count $MINC -sample $SAMPLE -k $K -threads $THREADS -iter $ITER -minn $MINN -maxn $MAXN -use-cache-file $CACHE

