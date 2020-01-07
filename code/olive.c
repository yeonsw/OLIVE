/*
Copyright 2019-present NAVER Corp. and KAIST (Korea Advanced Institute of Science and Technology)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Reference code: https://code.google.com/archive/p/word2vec/source/default/source (word2vec/trunk/word2vec.c)
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <stdbool.h>
#include <float.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000

typedef struct subword_index_node {
    long long i;
    struct subword_index_node *next;
}SIN;

typedef struct vocab_index_node {
    long long i;
    struct vocab_index_node *next;
}VIN;

struct vocab_word {
  long long cn;
  char *word;
  SIN *subws;
  long long subn;
};

typedef struct vocab_subw {
    long long cn;
    char *subw;
    VIN *vocabs;
}VS;

typedef struct y_dtype {
    long long n;
    int n_thread;
    double lr;
    double k;
    struct vocab_word *vocab;
    struct vocab_subw *v_subw;
    long long vocab_size;
    long long subw_size;
    long long start_v, end_v;
    long long start_z, end_z;
    long long *ws, *ss;
    float *word_vector1;
    float *word_vector2;
    float *word_vector;
    float *subword_vector;
    float *grad_word;
    float *exp_table;
    int dimension;
    struct n_given_i_header *co_matrix;
    double pair_cn;
    double *co_occ_sum;
}th_dtype;

typedef struct n_given_i {
  double cn;
  int word;
  struct n_given_i *next;
}NGI;

typedef struct n_given_i_header {
    struct n_given_i *first;
    struct n_given_i *last;
    unsigned int uw;
}NGI_H;

typedef struct address_given_ij {
    int w1;
    int w2;
    long long cn;    
    struct address_given_ij *next;
}AGIJ;

typedef struct subw2words {
    int *words;
}S2WS;

typedef struct SUBWORD_HASH_ELEMENT {
    long long si;
    char *subw;
    struct SUBWORD_HASH_ELEMENT *next;
}SHM;

typedef struct SUBWORD_HASH {
    struct SUBWORD_HASH_ELEMENT *head;
}SH;

typedef struct exp_settings {
    clock_t start_t;
    int dimension;
    char *corpus;
    char *vocab_fname;
    char *co_occ_fname;
    char *vector_u_fname;
    char *vector_v_fname;
    char *vector_zu_fname;
    char *vector_zv_fname;
    
    double lr, sample, k;
    int window, min_count;
    int num_threads;
    int iter, cache, minn, maxn;
    
    struct vocab_word *vocab;
    int *vocab_hash;
    int vocab_size;
    int vocab_hash_size;
    long long vocab_max_size;
    
    struct vocab_subw *v_subw;
    struct SUBWORD_HASH *subw_hash;
    long long subw_size;
    long long subw_hash_size;
    long long subw_max_size;

    double *vocab_weight;
    
    struct address_given_ij **co_hash;
    long long co_hash_size;

    struct n_given_i_header *co_matrix;
    double *co_occ_sum;
    float *U, *V, *ZU, *ZV;

    double pair_cn;
    long long pair_size;
    long long train_words; 
} EXPSETTING;


void ReadVocab(EXPSETTING *expsetting);
void read_co_occ_stat(EXPSETTING *expsetting);
void SortVocab(EXPSETTING *expsetting);
int VocabCompare(const void *a, const void *b);
int ReadWordIndex(FILE *fin, EXPSETTING *expsetting);
int SearchVocab(char *word, EXPSETTING *expsetting);
AGIJ *SearchPair(int word1, int word2, EXPSETTING *expsetting);
unsigned long long GetWordHash(char *word, int vocab_hash_size);
unsigned long long GetPairHash(int word1, int word2, EXPSETTING *expsetting);
int AddWordToVocab(char *word, EXPSETTING *expsetting);
AGIJ *AddPairToCo(int word1, int word2, EXPSETTING *expsetting);
void ReadWord(char *word, FILE *fin);
void SaveVocab(EXPSETTING *expsetting);
void LearnVocabFromTrainFile(EXPSETTING *expsetting);
void get_co_occ_stat(EXPSETTING *expsetting);
void save_co_occ_stat(EXPSETTING *expsetting);
void InitNet(EXPSETTING *expsetting);
void TrainModel(EXPSETTING *expsetting);
void insert(NGI_H *h, int i, int j, double cn);
void read_co_as_grouped(EXPSETTING *expsetting);
double inner_product(float *a, float *b, long long ai, long long bi, int n, bool bound, int dimension);
void save_word_vector_uv(char *vector_u_fname, char *vector_v_fname, EXPSETTING *expsetting);
void save_subword_vector_uv(char *vector_zu_fname, char *vector_zv_fname, EXPSETTING *expsetting);
void set_weight_vector(EXPSETTING *expsetting);
void get_subw_and_add(long long word_index, EXPSETTING *expsetting);
long long search_subw(char *subw, EXPSETTING *expsetting);
int add_subword_to_subv(char *subw, EXPSETTING *expsetting);
void read_subword(EXPSETTING *expsetting);
void link_vocab_subword(long long word_index, long long subw_index, EXPSETTING *expsetting);
void rand_shuffle(long long *seq, long long s, long long e);
void init_arg(th_dtype *arg, EXPSETTING *expsetting, long long *ws, long long *ss);

void *calculate_grad_word(void *id);
void *update_subword_vector(void *id);
void *update_word_vector(void *id);
void update_weight(EXPSETTING *expsetting);
float* precompute_exptable();

void init_expsettings_from_arg(int argc, char **argv, EXPSETTING *expsetting);
int arg_pos(char *str, int argc, char **argv);

void free_expsettings(EXPSETTING *expsetting);
void free_co_occ_stat(EXPSETTING *expsetting);
void free_subw_hash(SH *subw_hash, long long subw_size);
void free_expsettings(EXPSETTING *expsetting);
void free_co_matrix(EXPSETTING *expsetting);
void free_vocab(struct vocab_word *vocab, long long vocab_size);
void free_vocab_sub(VS *v_subw, long long subw_size);

int main(int argc, char **argv) {
    int i = 0;
    EXPSETTING expsetting;
    
    init_expsettings_from_arg(argc, argv, &expsetting);
    srand(time(NULL));
    TrainModel(&expsetting);
    free_expsettings(&expsetting);
    
    return 0;
}

void free_expsettings(EXPSETTING *expsetting) {
    free(expsetting -> corpus);
    free(expsetting -> vocab_fname);
    free(expsetting -> co_occ_fname);
    free(expsetting -> vector_u_fname);
    free(expsetting -> vector_v_fname);
    free(expsetting -> vector_zu_fname);
    free(expsetting -> vector_zv_fname);
    
    free(expsetting -> vocab_hash);
    free(expsetting -> vocab_weight);
    free(expsetting -> co_occ_sum);
    
    free(expsetting -> U);
    free(expsetting -> V);
    free(expsetting -> ZU);
    free(expsetting -> ZV);

    free_vocab(expsetting -> vocab, expsetting -> vocab_size);
    free_subw_hash(expsetting-> subw_hash, expsetting -> subw_size);
    free_vocab_sub(expsetting -> v_subw, expsetting -> subw_size);
    free_co_occ_stat(expsetting);
    free_co_matrix(expsetting);
    return;
}

void free_co_matrix(EXPSETTING *expsetting) {
    long long i;
    NGI *current = NULL, *next = NULL;
    for (i = 0; i < expsetting -> vocab_size; i++) {
        if (!(expsetting -> co_matrix)[i].first) continue;
        current = (expsetting -> co_matrix)[i].first;
        while (current) {
            next = current -> next;
            free(current);
            current = next;
        }
    }
    free(expsetting -> co_matrix);
}

void free_co_occ_stat(EXPSETTING *expsetting) {
    long long i;
    AGIJ *current = NULL, *next = NULL;
    if (!(expsetting -> co_hash)) return;
    for (i = 0; i < expsetting -> co_hash_size; i++) {
        current = (expsetting -> co_hash)[i];
        while (current) {
            next = current -> next;
            free(current);
            current = next;
        }
    }
    free(expsetting -> co_hash);
    expsetting -> co_hash = NULL;
}

void free_subw_hash(SH *subw_hash, long long subw_size) {
    long long i = 0;
    SHM *current = NULL, *next = NULL;
    for (i =  0; i < subw_size; i++) {
        if (!subw_hash[i].head) continue;
        current = subw_hash[i].head;
        while(current) {
            next = current -> next;
            free(current -> subw);
            free(current);
            current = next;
        }
    }
    free(subw_hash);
}

void free_vocab(struct vocab_word *vocab, long long vocab_size) {
    long long i = 0;
    SIN *current = NULL, *next = NULL;
    for(i = 0; i < vocab_size; i++) {
        free(vocab[i].word);
        current = vocab[i].subws;
        while(current) {
            next = current -> next;
            free(current);
            current = next;
        }
    }
    free(vocab);
    return;
}

void free_vocab_sub(VS *v_subw, long long subw_size) {
    long long i = 0;
    VIN *current = NULL, *next = NULL;
    for(i = 0; i < subw_size; i++) {
        free(v_subw[i].subw);
        current = v_subw[i].vocabs;
        while(current) {
            next = current -> next;
            free(current);
            current = next;
        }
    }
    free(v_subw);
    return;
}

void init_expsettings_from_arg(int argc, char **argv, EXPSETTING *expsetting) {
    long long i = 0;
    expsetting -> U = NULL;
    expsetting -> V = NULL;
    expsetting -> ZU = NULL;
    expsetting -> ZV = NULL;

    expsetting -> corpus = (char*)malloc(sizeof(char) * (MAX_STRING + 1));
    expsetting -> vocab_fname = (char*)malloc(sizeof(char) * (MAX_STRING + 1));
    expsetting -> co_occ_fname = (char*)malloc(sizeof(char) * (MAX_STRING + 1));
    expsetting -> vector_u_fname = (char*)malloc(sizeof(char) * (MAX_STRING + 1));
    expsetting -> vector_v_fname = (char*)malloc(sizeof(char) * (MAX_STRING + 1));
    expsetting -> vector_zu_fname = (char*)malloc(sizeof(char) * (MAX_STRING + 1));
    expsetting -> vector_zv_fname = (char*)malloc(sizeof(char) * (MAX_STRING + 1));
    
    expsetting -> dimension = 100;
    expsetting -> lr = 0.5;
    expsetting -> window = 5;
    expsetting -> min_count = 5;
    expsetting -> sample = 1e-5;
    expsetting -> k = 100.0;
    expsetting -> num_threads = 28;
    expsetting -> iter = 500;
    expsetting -> minn = 2;
    expsetting -> maxn = 7;
    expsetting -> cache = 1;

    if ((i = arg_pos((char *)"-corpus", argc, argv)) > 0) strcpy(expsetting -> corpus, argv[i + 1]);
    if ((i = arg_pos((char *)"-vocab-fname", argc, argv)) > 0) strcpy(expsetting -> vocab_fname, argv[i + 1]);
    if ((i = arg_pos((char *)"-co-occ-fname", argc, argv)) > 0) strcpy(expsetting -> co_occ_fname, argv[i + 1]);
    if ((i = arg_pos((char *)"-vector-u-fname", argc, argv)) > 0) strcpy(expsetting -> vector_u_fname, argv[i + 1]);
    if ((i = arg_pos((char *)"-vector-v-fname", argc, argv)) > 0) strcpy(expsetting -> vector_v_fname, argv[i + 1]);
    if ((i = arg_pos((char *)"-vector-zu-fname", argc, argv)) > 0) strcpy(expsetting -> vector_zu_fname, argv[i + 1]);
    if ((i = arg_pos((char *)"-vector-zv-fname", argc, argv)) > 0) strcpy(expsetting -> vector_zv_fname, argv[i + 1]);

    if ((i = arg_pos((char *)"-dimension", argc, argv)) > 0) expsetting -> dimension = atoi(argv[i + 1]);
    if ((i = arg_pos((char *)"-lr", argc, argv)) > 0) expsetting -> lr = atof(argv[i + 1]);
    if ((i = arg_pos((char *)"-window", argc, argv)) > 0) expsetting -> window = atoi(argv[i + 1]);
    if ((i = arg_pos((char *)"-min-count", argc, argv)) > 0) expsetting -> min_count = atoi(argv[i + 1]);
    if ((i = arg_pos((char *)"-sample", argc, argv)) > 0) expsetting -> sample = atof(argv[i + 1]);
    if ((i = arg_pos((char *)"-k", argc, argv)) > 0) expsetting -> k = atof(argv[i + 1]);
    if ((i = arg_pos((char *)"-threads", argc, argv)) > 0) expsetting -> num_threads = atoi(argv[i + 1]);
    if ((i = arg_pos((char *)"-iter", argc, argv)) > 0) expsetting -> iter = atoi(argv[i + 1]);
    if ((i = arg_pos((char *)"-minn", argc, argv)) > 0) expsetting -> minn = atoi(argv[i + 1]);
    if ((i = arg_pos((char *)"-maxn", argc, argv)) > 0) expsetting -> maxn = atoi(argv[i + 1]);
    if ((i = arg_pos((char *)"-use-cache-file", argc, argv)) > 0) expsetting -> cache = atoi(argv[i + 1]);

    expsetting -> vocab_max_size = 1000;
    expsetting -> vocab_size = 0;
    expsetting -> vocab_hash_size = 30000000;
    expsetting -> vocab = (struct vocab_word *) calloc(expsetting -> vocab_max_size, sizeof(struct vocab_word));
    expsetting -> vocab_hash = (int *) calloc(expsetting -> vocab_hash_size, sizeof(int));
    
    expsetting -> subw_max_size = 1000;
    expsetting -> subw_size = 0;
    expsetting -> subw_hash_size = 30000000;
    expsetting -> v_subw = (VS *) calloc(expsetting -> subw_max_size, sizeof(VS));
    expsetting -> subw_hash = (SH*) calloc(expsetting -> subw_hash_size, sizeof(SH));
    
    expsetting -> co_hash_size = 3000000000;
    expsetting -> co_hash = (AGIJ **) calloc(expsetting -> co_hash_size, sizeof(AGIJ*));
    
    expsetting -> pair_cn = 0.0;
    expsetting -> pair_size = 0;
    expsetting -> train_words = 0;
    return;
}

int arg_pos(char *str, int argc, char **argv) {
    int i = 0;
    for (i = 1; i < argc; i++)
        if (!strcmp(str, argv[i])) {
            if (i == argc - 1) {
                printf("Argument missing for %s\n", str);
                exit(1);
            }
            return i;
        }
    return -1;
}

void set_weight_vector(EXPSETTING *expsetting) {
    long long i = 0;
    double sample = expsetting -> sample;
    double nw = expsetting -> train_words;
    double golden = (sqrt(5) + 1.0) / 2.0;
    double thresh = golden * golden * nw * sample;
    double ns = sample * nw;
    double sqrt_ns = sqrt(ns);
    for (i = 1; i < expsetting -> vocab_size; i++) {
        if (thresh > (double)(expsetting -> vocab)[i].cn) (expsetting -> vocab_weight)[i] = 1.0;
        else (expsetting -> vocab_weight)[i] = (sqrt(ns / (expsetting -> vocab)[i].cn) + ns / (expsetting -> vocab)[i].cn);
    }
    return;
}

void *calculate_grad_word(void *id) {
    th_dtype *arg = (th_dtype*)id;
    long long start = arg -> start_v;
    long long end = arg -> end_v;
    float *word_vector1 = arg -> word_vector1;
    float *word_vector2 = arg -> word_vector2;
    float *grad_word = arg -> grad_word;
    float *exp_table = arg -> exp_table;
    int dimension = arg -> dimension;
    long long *ws = arg -> ws;
    double k = arg -> k;
    NGI_H *co_matrix = arg -> co_matrix;
    double *co_occ_sum = arg -> co_occ_sum;
    double pair_cn = arg -> pair_cn;
    long long ri = 0, d = 0, word_i = 0, current = 0, word_index = 0, co_word_index = 0; 
    double pre_computed = 0.0, inner_p = 0.0, divider = 1000.0;
    double sig = 0.0, sig_inv = 0.0, exp_table_const = EXP_TABLE_SIZE / (double) MAX_EXP / 2.0;
    NGI *tmp = NULL;

    for(ri = start; ri < end; ri++) {
        word_i = ws[ri];
        word_index = ws[ri] * (dimension);
        for(d = 0; d < dimension; d++) grad_word[word_index + d] = 0.0;
        tmp = co_matrix[word_i].first;
        while(tmp) {
            current = tmp -> word;
            inner_p = inner_product(word_vector1, word_vector2, word_i, current, dimension, true, dimension);
            sig = exp_table[(int)((inner_p + MAX_EXP) * exp_table_const)]; sig_inv = exp_table[(int)(( - inner_p + MAX_EXP) * exp_table_const)];
            pre_computed = 1.0 / divider * (sig_inv * sig * co_occ_sum[current] / (double)pair_cn * co_occ_sum[word_i] * k / 4 - sig_inv * sig_inv * tmp -> cn);
            co_word_index = current * dimension;
            for(d = 0; d < dimension; d++) grad_word[word_index + d] += word_vector2[co_word_index + d] * pre_computed;
            tmp = tmp -> next;
        }
    }
    pthread_exit(NULL);
    return;
}

void *update_subword_vector(void *id) {
    th_dtype *arg = (th_dtype*)id;
    float lr = arg -> lr;
    long long start = arg -> start_z;
    long long end = arg -> end_z;
    long long *ss = arg -> ss;
    int dimension = arg -> dimension;
    float *subword_vector = arg -> subword_vector;
    float *grad_word = arg -> grad_word;
    VS *v_subw = arg -> v_subw;
    long long ri = 0, d = 0, z_index = 0, i = 0, w_index = 0;
    double len = 0.0;
    double *grad = (double *)calloc(dimension, sizeof(double));
    VIN *element = NULL;
    double rmax1 = 0.0;
    for(ri = start; ri < end; ri++) {
        z_index = ss[ri] * dimension;
        for (d = 0; d < dimension; d++) grad[d] = 0.0;
        element = v_subw[ss[ri]].vocabs;
        while(element) {
            w_index = (element -> i) * dimension;
            for(d = 0; d < dimension; d++) grad[d] += grad_word[w_index + d];
            element = element -> next;
        }
        for(d = 0; d < dimension; d++) if (fabs(grad[d]) > rmax1) rmax1 = fabs(grad[d]);
        len = 1e-8; for (d = 0; d < dimension; d++) len += grad[d] * grad[d]; len = sqrt(len);
        for(d = 0; d < dimension; d++) subword_vector[z_index + d] -= grad[d] / len * lr; 
    }
    free(grad);
    pthread_exit(NULL);
    return;
}

void *update_word_vector(void *id) {
    th_dtype *arg = (th_dtype*)id;
    long long start = arg -> start_v;
    long long end = arg -> end_v;
    long long *ws = arg -> ws;
    float *word_vector = arg -> word_vector;
    float *subword_vector = arg -> subword_vector;
    struct vocab_word *vocab = arg -> vocab;
    int dimension = arg -> dimension;
    long long ri = 0, d = 0, word_index = 0, word_i = 0, subw_i = 0, z_index = 0;
    SIN *element = NULL;

    for(ri = start; ri < end; ri++) {
        word_i = ws[ri];
        word_index = word_i * dimension;
        for(d = 0; d < dimension; d++) word_vector[word_index + d] = 0.0;
        element = vocab[word_i].subws;
        while(element) {
            subw_i = element -> i;
            z_index = subw_i * dimension;
            for(d = 0; d < dimension; d++) word_vector[word_index + d] += subword_vector[z_index + d];
            element = element -> next;
        }
        for(d = 0; d < dimension; d++) word_vector[word_index + d] /= vocab[word_i].subn;
    }
    pthread_exit(NULL);
    return;
}

void rand_shuffle(long long *seq, long long s, long long e) {
    long long i = 0, ri = 0, tmp = 0;
    unsigned long long next_random = rand();
    for(i = s; i < e; i++) {
        next_random = next_random * (unsigned long long)25214903917 + 11;
        ri = next_random % ((e - s) - (i - s)) + i;
        tmp = seq[i];
        seq[i] = seq[ri];
        seq[ri] = tmp;
    }
    return;
}

void init_arg(th_dtype *arg, EXPSETTING *expsetting, long long *ws, long long *ss) {
    int i = 0;
    long long block_size = 0;
    for(i = 0; i < expsetting -> num_threads; i++) {
        arg[i].n = i; arg[i].k = expsetting -> k;
        arg[i].lr = expsetting -> lr;
        arg[i].vocab = expsetting -> vocab;
        arg[i].v_subw = expsetting -> v_subw;
        arg[i].subw_size = expsetting -> subw_size;
        arg[i].vocab_size = expsetting -> vocab_size;
        arg[i].start_v = 0; arg[i].end_v = 0;
        arg[i].start_z = 0; arg[i].end_z = 0;
        arg[i].ws = ws; arg[i].ss = ss;
        arg[i].n_thread = expsetting -> num_threads;
        arg[i].word_vector1 = NULL;
        arg[i].word_vector2 = NULL;
        arg[i].word_vector = NULL;
        arg[i].subword_vector = NULL;
        arg[i].dimension = expsetting -> dimension;
        arg[i].co_matrix = expsetting -> co_matrix;
        arg[i].pair_cn = expsetting -> pair_cn;
        arg[i].co_occ_sum = expsetting -> co_occ_sum;
    }

    block_size = (long long)(((expsetting -> vocab_size) - 1) / (double)(expsetting -> num_threads)) + 1;
    for(i = 0; i < expsetting -> num_threads; i++) {
        arg[i].start_v = block_size * i; arg[i].end_v = block_size * (i + 1);
    }
    arg[0].start_v = 1;
    arg[expsetting -> num_threads - 1].end_v = expsetting -> vocab_size;
    
    block_size = (long long)(((expsetting -> subw_size) - 1) / (double)(expsetting -> num_threads)) + 1;
    for(i = 0; i < expsetting -> num_threads; i++) {
        arg[i].start_z = block_size * i; arg[i].end_z = block_size * (i + 1);
    }
    arg[expsetting -> num_threads - 1].end_z = expsetting -> subw_size;
    return;
}

float* precompute_exptable() {
    int i = 0;
    float *exp_table = NULL;
    exp_table = (float *)malloc((EXP_TABLE_SIZE + 1) * sizeof(float));
    for (i = 0; i < EXP_TABLE_SIZE; i++) {
        exp_table[i] = exp((i / (float)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        exp_table[i] = exp_table[i] / (exp_table[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }
    return exp_table;
}

void update_weight(EXPSETTING *expsetting) {
    int  a = 0;
    long long i = 0;
    time_t p1, p2;
    long long *ws = (long long *)calloc((expsetting -> vocab_size), sizeof(long long));
    long long *ss = (long long *)calloc((expsetting -> subw_size), sizeof(long long));
    float *grad_word = NULL;
    float *exp_table = NULL;
    float *U = expsetting -> U;
    float *V = expsetting -> V;
    float *ZU = expsetting -> ZU;
    float *ZV = expsetting -> ZV;
    
    exp_table = precompute_exptable();
    
    a = posix_memalign((void **)&grad_word, 128, (long long)(expsetting -> vocab_size) * expsetting -> dimension * sizeof(float));
    if (grad_word == NULL) { printf("Memory allocation failed\n"); exit(1); }
    
    pthread_t *pt = NULL;
    th_dtype *arg = NULL;
    
    for(i = 0; i < (expsetting -> vocab_size); i++) ws[i] = i;
    rand_shuffle(ws, 1, (expsetting -> vocab_size));
    for(i = 0; i < (expsetting -> subw_size); i++) ss[i] = i;
    rand_shuffle(ss, 0, (expsetting -> subw_size));

    arg = (th_dtype*)malloc(expsetting -> num_threads * sizeof(th_dtype));
    init_arg(arg, expsetting, ws, ss);
    
    pt = (pthread_t *)malloc(expsetting -> num_threads * sizeof(pthread_t));
    for(a = 1; a < expsetting -> iter + 1; a++) {
        for (i = 0; i < expsetting -> num_threads; i++) { 
            arg[i].lr = expsetting -> lr - (float) (expsetting -> lr) / expsetting -> iter * (a - 1);
            arg[i].grad_word = grad_word;
            arg[i].exp_table = exp_table;
        }
        time(&p1);
        for (i = 0; i < expsetting -> num_threads; i++) {
            arg[i].word_vector1 = U; arg[i].word_vector2 = V; arg[i].word_vector = U;
            arg[i].subword_vector = ZU;
        }
        for(i = 0; i < expsetting -> num_threads; i++) pthread_create(&pt[i], NULL, calculate_grad_word, (void*)&arg[i]);
        for(i = 0; i < expsetting -> num_threads; i++) pthread_join(pt[i], NULL);
        
        for(i = 0; i < expsetting -> num_threads; i++) pthread_create(&pt[i], NULL, update_subword_vector, (void*)&arg[i]);
        for(i = 0; i < expsetting -> num_threads; i++) pthread_join(pt[i], NULL);
        
        for(i = 0; i < expsetting -> num_threads; i++) pthread_create(&pt[i], NULL, update_word_vector, (void*)&arg[i]);
        for(i = 0; i < expsetting -> num_threads; i++) pthread_join(pt[i], NULL);
        
        for (i = 0; i < expsetting -> num_threads; i++) {
            arg[i].word_vector1 = V; arg[i].word_vector2 = U; arg[i].word_vector = V;
            arg[i].subword_vector = ZV;
        }
        for(i = 0; i < expsetting -> num_threads; i++) pthread_create(&pt[i], NULL, calculate_grad_word, (void*)&arg[i]);
        for(i = 0; i < expsetting -> num_threads; i++) pthread_join(pt[i], NULL);
        
        for(i = 0; i < expsetting -> num_threads; i++) pthread_create(&pt[i], NULL, update_subword_vector, (void*)&arg[i]);
        for(i = 0; i < expsetting -> num_threads; i++) pthread_join(pt[i], NULL);
        
        for(i = 0; i < expsetting -> num_threads; i++) pthread_create(&pt[i], NULL, update_word_vector, (void*)&arg[i]);
        for(i = 0; i < expsetting -> num_threads; i++) pthread_join(pt[i], NULL);
        time(&p2);
        printf("Iteration: %03d, learning_rate: %.6lf, time (sec): %d\n", a, arg[0].lr, p2 - p1);
        if (a % 50 == 0) {
            save_word_vector_uv(expsetting -> vector_u_fname, expsetting -> vector_v_fname, expsetting);
            save_subword_vector_uv(expsetting -> vector_zu_fname, expsetting -> vector_zv_fname, expsetting);
            printf("iter %03d: U, V saved\n", a);
        }
    }
    free(grad_word);
    free(exp_table);
    free(ws); free(ss);
    free(pt);
    free(arg);
    return;
}

double inner_product(float *a, float *b, long long ai, long long bi, int n, bool bound, int dimension) {
    int i = 0;
    double r = 0.0;
    bool pos = true;
    for (i = 0; i < n; i++) r += a[ai * dimension + i] * b[bi * dimension + i];
    if (bound) {
        if (r >= MAX_EXP) r = MAX_EXP - 0.1;
        if (r <= -MAX_EXP) r = - MAX_EXP + 0.1;
    }
    return r;
}

void read_co_as_grouped(EXPSETTING *expsetting) {
    int i, j;
    expsetting -> pair_size = 0;
    expsetting -> pair_cn = 0.0;
    char c;
    char word1[MAX_STRING], word2[MAX_STRING];
    long long cn = 0;
    double cn_double = 0.0;
    FILE *fin = fopen(expsetting -> co_occ_fname, "rb");
    if (fin == NULL) {
        printf("Co-occ file not found\n");
        exit(1);
    }
    while (1) {
        if ((expsetting -> pair_size) % 1000000 == 0) {
            printf("%cProgress: %d ", 13, expsetting -> pair_size);
            fflush(stdout);
        }
        ReadWord(word1, fin);
        ReadWord(word2, fin);
        
        if (feof(fin)) break;
        fscanf(fin, "%lld%c", &cn, &c);
        i = SearchVocab(word1, expsetting);
        j = SearchVocab(word2, expsetting);
        if (i == -1 || j == -1) continue;
        cn_double = cn * ((expsetting -> vocab_weight)[i] * (expsetting -> vocab_weight)[j]);
        
        insert(expsetting -> co_matrix, i, j, cn_double);
        insert(expsetting -> co_matrix, j, i, cn_double);
        (expsetting -> co_occ_sum)[i] += cn_double;
        (expsetting -> co_occ_sum)[j] += cn_double;
        (expsetting -> pair_cn) += cn_double;
        (expsetting -> pair_size)++;
    }
    fclose(fin);
    printf("pair size: %lld %lf\n", expsetting -> pair_size, expsetting -> pair_cn);
    return;
}

void insert(NGI_H *h, int i, int j, double cn) {
    NGI *new = (NGI*) malloc(sizeof(NGI));
    new -> word = j;
    new -> cn = cn;
    new -> next = NULL;
    if (!h[i].first && !h[i].last) {
        h[i].first = new;
        h[i].last = new;
    } else {
        h[i].last -> next = new;
        h[i].last = new;
    }
    h[i].uw++;
    return;
}

void TrainModel(EXPSETTING *expsetting) {
    long long i = 0;
    time_t p1, p2;

    printf("Starting training using file %s\n", expsetting -> corpus);
    if (expsetting -> cache) {
        ReadVocab(expsetting);
    } else {
        LearnVocabFromTrainFile(expsetting);
        SaveVocab(expsetting);
    }
    read_subword(expsetting);

    expsetting -> start_t = clock();
    
    printf("counting co-occ stat....\n");
    time(&p1);
    if (!(expsetting -> cache)) {
        get_co_occ_stat(expsetting);
        save_co_occ_stat(expsetting);
        free_co_occ_stat(expsetting);
    }
    time(&p2);
    printf("end counting... comp time: %lld\n", p2 - p1);
    
    expsetting -> vocab_weight = (double*)calloc(expsetting -> vocab_size, sizeof(double));

    expsetting -> co_matrix = (NGI_H*) calloc(expsetting -> vocab_size, sizeof(NGI_H));
    for (i = 0; i < expsetting -> vocab_size; i++) {
        expsetting -> co_matrix[i].first = NULL;
        expsetting -> co_matrix[i].last = NULL;
        expsetting -> co_matrix[i].uw = 0;
    }
    expsetting -> co_occ_sum = (double *)calloc(expsetting -> vocab_size, sizeof(double));

    set_weight_vector(expsetting);
    printf("grouping co-occ stat\n");
    time(&p1); 
    read_co_as_grouped(expsetting);
    time(&p2);
    printf("end grouping... comp time: %lld\n", p2 - p1);
    
    InitNet(expsetting);
    
    printf("Training...\n");
    time(&p1);
    update_weight(expsetting);
    time(&p2);
    printf("Training end... comp time:  %lld\n", p2 - p1);
    
    printf("Saving word vectors\n");
    time(&p1);
    save_word_vector_uv(expsetting -> vector_u_fname, expsetting -> vector_v_fname, expsetting);
    save_subword_vector_uv(expsetting -> vector_zu_fname, expsetting -> vector_zv_fname, expsetting);
    time(&p2);
    printf("Saving end... comp time:  %lld\n", p2 - p1);
    
    return;
}

void save_subword_vector_uv(char *vector_zu_fname, char *vector_zv_fname, EXPSETTING *expsetting) {
    long a = 0, b = 0;
    FILE *fo_u = NULL, *fo_v = NULL;
    float *ZU = expsetting -> ZU;
    float *ZV = expsetting -> ZV;
    fo_u = fopen(vector_zu_fname, "wb");
    fprintf(fo_u, "%lld %lld\n", expsetting -> subw_size, expsetting -> dimension);
    for (a = 0; a < expsetting -> subw_size; a++) {
        fprintf(fo_u, "%s ", (expsetting -> v_subw)[a].subw);
        for (b = 0; b < expsetting -> dimension; b++) fwrite(&ZU[a * (expsetting -> dimension) + b], sizeof(float), 1, fo_u);
        fprintf(fo_u, "\n");
    }
    
    fo_v = fopen(vector_zv_fname, "wb");
    fprintf(fo_v, "%lld %lld\n", expsetting -> subw_size, expsetting -> dimension);
    for (a = 0; a < expsetting -> subw_size; a++) {
        fprintf(fo_v, "%s ", (expsetting -> v_subw)[a].subw);
        for (b = 0; b < expsetting -> dimension; b++) fwrite(&ZV[a * (expsetting -> dimension) + b], sizeof(float), 1, fo_v);
        fprintf(fo_v, "\n");
    }
    fclose(fo_u);
    fclose(fo_v);
    return;
}
void save_word_vector_uv(char *vector_u_fname, char *vector_v_fname, EXPSETTING *expsetting) {
    long a = 0, b = 0;
    FILE *fo_u = NULL, *fo_v = NULL;
    float *U = expsetting -> U;
    float *V = expsetting -> V;

    fo_u = fopen(vector_u_fname, "wb");
    fprintf(fo_u, "%lld %lld\n", expsetting -> vocab_size, expsetting -> dimension);
    for (a = 0; a < expsetting -> vocab_size; a++) {
        fprintf(fo_u, "%s ", (expsetting -> vocab)[a].word);
        for (b = 0; b < expsetting -> dimension; b++) fwrite(&U[a * (expsetting -> dimension) + b], sizeof(float), 1, fo_u);
        fprintf(fo_u, "\n");
    }
    
    fo_v = fopen(vector_v_fname, "wb");
    fprintf(fo_v, "%lld %lld\n", expsetting -> vocab_size, expsetting -> dimension);
    for (a = 0; a < expsetting -> vocab_size; a++) {
        fprintf(fo_v, "%s ", (expsetting -> vocab)[a].word);
        for (b = 0; b < expsetting -> dimension; b++) fwrite(&V[a * (expsetting -> dimension) + b], sizeof(float), 1, fo_v);
        fprintf(fo_v, "\n");
    }
    fclose(fo_u);
    fclose(fo_v);
    return;
}

void InitNet(EXPSETTING *expsetting) {
    long long a, b;
    unsigned long long next_random = rand();

    a = posix_memalign((void **)&(expsetting -> U), 128, (long long)(expsetting -> vocab_size) * (expsetting -> dimension) * sizeof(float));
    if (expsetting -> U == NULL) {printf("Memory allocation failed\n"); exit(1);}
  
    a = posix_memalign((void **)&(expsetting -> V), 128, (long long)(expsetting -> vocab_size) * (expsetting -> dimension) * sizeof(float));
    if (expsetting -> V == NULL) {printf("Memory allocation failed\n"); exit(1);}
    
    a = posix_memalign((void **)&(expsetting -> ZU), 128, (long long)(expsetting -> subw_size) * (expsetting -> dimension) * sizeof(float));
    if (expsetting -> ZU == NULL) {printf("Memory allocation failed\n"); exit(1);}

    a = posix_memalign((void **)&(expsetting -> ZV), 128, (long long)(expsetting -> subw_size) * (expsetting -> dimension) * sizeof(float));
    if (expsetting -> ZV == NULL) {printf("Memory allocation failed\n"); exit(1);}
    
    for (a = 0; a < expsetting -> vocab_size; a++) for (b = 0; b < expsetting -> dimension; b++) {
        next_random = next_random * (unsigned long long)25214903917 + 11;
        (expsetting -> U)[a * (expsetting -> dimension) + b] = (((next_random & 0xFFFF) / (float)65536) - 0.5) / expsetting -> dimension;
    }
    for (a = 0; a < expsetting -> vocab_size; a++) for (b = 0; b < expsetting -> dimension; b++) {
        next_random = next_random * (unsigned long long)25214903917 + 11;
        (expsetting -> V)[a * (expsetting -> dimension) + b] = (((next_random & 0xFFFF) / (float)65536) - 0.5) / expsetting -> dimension;
    }

    for (a = 0; a < expsetting -> subw_size; a++) for (b = 0; b < expsetting -> dimension; b++) {
        next_random = next_random * (unsigned long long)25214903917 + 11;
        (expsetting -> ZU)[a * (expsetting -> dimension) + b] = (((next_random & 0xFFFF) / (float)65536) - 0.5) / expsetting -> dimension;
    }
    
    for (a = 0; a < expsetting -> subw_size; a++) for (b = 0; b < expsetting -> dimension; b++) {
        next_random = next_random * (unsigned long long)25214903917 + 11;
        (expsetting -> ZV)[a * (expsetting -> dimension) + b] = (((next_random & 0xFFFF) / (float)65536) - 0.5) / expsetting -> dimension;
    }
    return;
}

void read_co_occ_stat(EXPSETTING *expsetting) {
    long long a, i, j;
    expsetting -> pair_size = 0;
    char c;
    char word1[MAX_STRING], word2[MAX_STRING];
    AGIJ *tmp = NULL;
    FILE *fin = fopen(expsetting -> co_occ_fname, "rb");
    if (fin == NULL) {
        printf("Co-occ file not found\n");
        exit(1);
    }
    while (1) {
        ReadWord(word1, fin);
        ReadWord(word2, fin);
        if (feof(fin)) break;
        i = SearchVocab(word1, expsetting);
        j = SearchVocab(word2, expsetting);
        tmp = AddPairToCo(i, j, expsetting);
        fscanf(fin, "%lld%c", &(tmp -> cn), &c);
        (expsetting -> pair_size)++;
    }
    fclose(fin);
    printf("pair size: %lld\n", expsetting -> pair_size);
}

void save_co_occ_stat(EXPSETTING *expsetting) {
    long long i;
    FILE *fo = fopen(expsetting -> co_occ_fname, "wb");
    AGIJ *tmp = NULL;
    time_t s, e;
    time(&s);
    printf("start_saving: %lld\n", s);
    for (i = 0; i < expsetting -> co_hash_size; i++) {
        tmp = (expsetting -> co_hash)[i];
        while(tmp) {
            fprintf(fo, "%s %s %lld\n", (expsetting -> vocab)[tmp -> w1].word, (expsetting -> vocab)[tmp -> w2].word, tmp -> cn);
            tmp = tmp -> next;
        }
    }
    fclose(fo);
    time(&e);
    printf("end_saving: %lld\n", e);
}

void get_co_occ_stat(EXPSETTING *expsetting) {
    AGIJ *tmp = NULL;
    long long i = 0, word, last_word, sentence_length = 0, sentence_position = 0;
    long long c, word_count = 0, last_word_count = 0;
    long long sen[MAX_SENTENCE_LENGTH + 1];
    clock_t now;
    clock_t start_t = expsetting -> start_t;

    FILE *fi = fopen(expsetting -> corpus, "rb");
    while(!feof(fi)) {
        if (word_count - last_word_count > 10000) {
            last_word_count = word_count;
            now=clock();
            printf("%cProgress: %.2f%%  Words/thread/sec: %.2fk  ", 13, 
                    word_count / (float)((expsetting -> train_words) + 1) * 100, 
                    word_count / ((float)(now - start_t + 1) / (float)CLOCKS_PER_SEC * 1000));
            fflush(stdout);
        }
        if (sentence_length == 0) {
            while (1) {
                word = ReadWordIndex(fi, expsetting);
                if (feof(fi)) break;
                if (word == -1) continue;
                word_count++;
                if (word == 0) break;
                sen[sentence_length] = word;
                sentence_length++;
                if (sentence_length >= MAX_SENTENCE_LENGTH) break;
            }
            sentence_position = 0;
        }
        word = sen[sentence_position];
        
        for (i = 0; i < (expsetting -> window) * 2 + 1; i++) if (i != expsetting -> window) {
            c = sentence_position - (expsetting -> window) + i;
            if (c < 0 || c >= sentence_length) continue;
            last_word = sen[c];
            if (last_word == -1) continue;

            tmp = SearchPair(word, last_word, expsetting);
            if (!tmp) {
                tmp = AddPairToCo(word, last_word, expsetting);
                tmp -> cn = 1;
            } else tmp -> cn ++;
        }
        sentence_position++;
        if (sentence_position >= sentence_length) {
            sentence_length = 0;
            continue;
        }
    }
    fclose(fi);
    return;
}

void LearnVocabFromTrainFile(EXPSETTING *expsetting) {
    char word[MAX_STRING];
    FILE *fin;
    long long a, i;
    for (a = 0; a < expsetting -> vocab_hash_size; a++) (expsetting -> vocab_hash)[a] = -1;
    fin = fopen(expsetting -> corpus, "rb");
    if (fin == NULL) {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }
    expsetting -> vocab_size = 0;
    AddWordToVocab((char *)"</s>", expsetting);
    while (1) {
        ReadWord(word, fin);
        if (feof(fin)) break;
        (expsetting -> train_words)++;
        
        if ((expsetting -> train_words) % 100000 == 0) {
            printf("%lldK%c", (expsetting -> train_words) / 1000, 13);
            fflush(stdout);
        }
        
        i = SearchVocab(word, expsetting);
        if (i == -1) {
            a = AddWordToVocab(word, expsetting);
            (expsetting -> vocab)[a].cn = 1;
        } else (expsetting -> vocab)[i].cn++;
    }
    
    SortVocab(expsetting);
    printf("Vocab size: %lld\n", expsetting -> vocab_size);
    printf("Words in train file: %lld\n", expsetting -> train_words);
    fclose(fin);
}


void SaveVocab(EXPSETTING *expsetting) {
    long long i;
    FILE *fo = fopen(expsetting -> vocab_fname, "wb");
    for (i = 0; i < expsetting -> vocab_size; i++) fprintf(fo, "%s %lld\n", (expsetting -> vocab)[i].word, (expsetting -> vocab)[i].cn);
    fclose(fo);
    return;
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
    int a = 0, ch;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if (ch == 13) continue;
        if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
            if (a > 0) {
                if (ch == '\n') ungetc(ch, fin);
                break;
            }
            if (ch == '\n') {
                strcpy(word, (char *)"</s>");
                return;
            } else continue;
        }
        word[a] = ch;
        a++;
        if (a >= MAX_STRING - 1) a--;   // Truncate too long words
    }
    word[a] = 0;
}

AGIJ *AddPairToCo(int word1, int word2, EXPSETTING *expsetting) {
    unsigned long long hash = GetPairHash(word1, word2, expsetting);

    AGIJ *new = (AGIJ *) malloc(sizeof(AGIJ));
    new -> cn = 0;
    new -> w1 = word1;
    new -> w2 = word2;
    new -> next = NULL;

    AGIJ *tmp = (expsetting -> co_hash)[hash];
    new -> next = tmp;
    (expsetting -> co_hash)[hash] = new;
    return new;
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word, EXPSETTING *expsetting) {
    unsigned long long hash = 0;
    unsigned int length = strlen(word) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    (expsetting -> vocab)[expsetting -> vocab_size].word = (char *)calloc(length, sizeof(char));
    strcpy((expsetting -> vocab)[expsetting -> vocab_size].word, word);
    (expsetting -> vocab)[expsetting -> vocab_size].cn = 0;
    (expsetting -> vocab)[expsetting -> vocab_size].subws = NULL;
    (expsetting -> vocab)[expsetting -> vocab_size].subn = 0;
    (expsetting -> vocab_size)++;
    // Reallocate memory if needed
    if ((expsetting -> vocab_size) + 2 >= expsetting -> vocab_max_size) {
        (expsetting -> vocab_max_size) += 1000;
        expsetting -> vocab = (struct vocab_word *)realloc(expsetting -> vocab, (expsetting -> vocab_max_size) * sizeof(struct vocab_word));
    }
    hash = GetWordHash(word, expsetting -> vocab_hash_size);
    while ((expsetting -> vocab_hash)[hash] != -1) hash = (hash + 1) % (expsetting -> vocab_hash_size);
    (expsetting -> vocab_hash)[hash] = (expsetting -> vocab_size) - 1;
    return (expsetting -> vocab_size) - 1;
}

unsigned long long GetPairHash(int word1, int word2, EXPSETTING *expsetting) {
    unsigned long long hash1 = word1 * (unsigned long long)25214903917 + 11;
    unsigned long long hash2 = word2 * (unsigned long long)25214903917 + 11;
    hash1 = hash1 * (unsigned long long)25214903917 + 11;
    hash2 = hash2 * (unsigned long long)25214903917 + 11;
    hash1 = hash1 * hash2;
    hash1 = hash1 % (expsetting -> co_hash_size);
    return hash1;
}

// Returns hash value of a word
unsigned long long GetWordHash(char *word, int vocab_hash_size) {
    unsigned long long hash = 0;
    char s_len = strlen(word);
    char i = 0;
    for (i = 0; i < s_len; i++) hash = hash * 257 + word[i];
    hash = hash % vocab_hash_size;
    return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
AGIJ *SearchPair(int word1, int word2, EXPSETTING *expsetting) {
    unsigned long long hash = GetPairHash(word1, word2, expsetting);
    AGIJ *tmp = (expsetting -> co_hash)[hash];
    while (tmp) {
        if ((word1 == tmp -> w1 && word2 == tmp -> w2) || (word2 == tmp -> w1 && word1 == tmp -> w2)) return tmp;
        tmp = tmp -> next;
    }
    return NULL;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word, EXPSETTING *expsetting) {
  unsigned long long hash = GetWordHash(word, expsetting -> vocab_hash_size);
  while (1) {
    if ((expsetting -> vocab_hash)[hash] == -1) return -1;
    if (!strcmp(word, (expsetting -> vocab)[(expsetting -> vocab_hash)[hash]].word)) return (expsetting -> vocab_hash)[hash];
    hash = (hash + 1) % (expsetting -> vocab_hash_size);
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin, EXPSETTING *expsetting) {
    char word[MAX_STRING];
    ReadWord(word, fin);
    if (feof(fin)) return -1;
    return SearchVocab(word, expsetting);
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab(EXPSETTING *expsetting) {
    int a;
    long long size;
    unsigned long long hash;
    size = expsetting -> vocab_size;
    // Sort the vocabulary and keep </s> at the first position
    qsort(&((expsetting -> vocab)[1]), (expsetting -> vocab_size) - 1, sizeof(struct vocab_word), VocabCompare);
    for (a = 0; a < expsetting -> vocab_hash_size; a++) (expsetting -> vocab_hash)[a] = -1;
    expsetting -> train_words = 0;
    for (a = 0; a < size; a++) {
        // Words occuring less than min_count times will be discarded from the vocab
        if (((expsetting -> vocab)[a].cn < expsetting -> min_count) && (a != 0)) {
            (expsetting -> vocab_size)--;
            free((expsetting -> vocab)[a].word);
        } else {
            // Hash will be re-computed, as after the sorting it is not actual
            hash=GetWordHash((expsetting -> vocab)[a].word, expsetting -> vocab_hash_size);
            while ((expsetting -> vocab_hash)[hash] != -1) hash = (hash + 1) % (expsetting -> vocab_hash_size);
            (expsetting -> vocab_hash)[hash] = a;
            (expsetting -> train_words) += (expsetting -> vocab)[a].cn;
        }
    }
    expsetting -> vocab = (struct vocab_word *)realloc(expsetting -> vocab, ((expsetting -> vocab_size) + 1) * sizeof(struct vocab_word));
    return;
}

void ReadVocab(EXPSETTING *expsetting) {
    long long a;
    char c;
    char word[MAX_STRING];
    FILE *fin = fopen(expsetting -> vocab_fname, "rb");
    if (fin == NULL) {
        printf("Vocabulary file not found\n");
        exit(1);
    }
    for (a = 0; a < expsetting -> vocab_hash_size; a++) (expsetting -> vocab_hash)[a] = -1;
    
    expsetting -> vocab_size = 0;
    while (1) {
        ReadWord(word, fin);
        if (feof(fin)) break;
        a = AddWordToVocab(word, expsetting);
        fscanf(fin, "%lld%c", &((expsetting -> vocab)[a].cn), &c);
    }
    fclose(fin);
    
    SortVocab(expsetting);
    printf("Vocab size: %lld\n", expsetting -> vocab_size);
}

void read_subword(EXPSETTING *expsetting) {
    long long i;
    char c;
    char word[MAX_STRING];
    
    for (i = 0; i < expsetting -> subw_hash_size; i++) (expsetting -> subw_hash)[i].head = NULL;
    for (i = 0; i < expsetting -> vocab_size; i++) {
        get_subw_and_add(i, expsetting);
        if (i % 5000 == 0) {
            printf("%lld/%lld, N subword: %lld%c", i, expsetting -> vocab_size, (expsetting -> subw_size), 13);
            fflush(stdout);
        }
    }
    printf("\nsubw_size: %lld\n", expsetting -> subw_size);
}

void link_vocab_subword(long long word_index, long long subw_index, EXPSETTING *expsetting) {
    VIN *tmpv = NULL;
    SIN *tmps = NULL;
    
    tmpv = (VIN*)malloc(sizeof(VIN));
    tmpv -> i = word_index;
    tmpv -> next = NULL;

    tmps = (SIN*)malloc(sizeof(SIN));
    tmps -> i = subw_index;
    tmps -> next = NULL;

    tmps -> next = (expsetting -> vocab)[word_index].subws;
    (expsetting -> vocab)[word_index].subws = tmps;
    (expsetting -> vocab)[word_index].subn++;

    tmpv -> next = (expsetting -> v_subw)[subw_index].vocabs;
    (expsetting -> v_subw)[subw_index].vocabs = tmpv;
    return;
}

void get_subw_and_add(long long word_index, EXPSETTING *expsetting) {
    long long a = 0, subw_index = 0;
    int n = 0, i = 0, j = 0, len = 0;
    
    char *b_word_e = (char*)malloc(sizeof(char) * (strlen((expsetting -> vocab)[word_index].word) + 3));
    sprintf(b_word_e, "%s%s%s", "<", (expsetting -> vocab)[word_index].word, ">"); len = strlen(b_word_e);

    char *ngram = (char*)malloc(sizeof(char) * (len + 1));
    for (i = 0; i < len; i++) {
        for (j = i, n = 1; j < len && n <= expsetting -> maxn; n++) {
            ngram[n - 1] = b_word_e[j++];
            if (n >= expsetting -> minn) {
                ngram[n] = 0;
                subw_index = search_subw(ngram, expsetting);
                if (subw_index == -1) subw_index = add_subword_to_subv(ngram, expsetting);
                link_vocab_subword(word_index, subw_index, expsetting);
            }
        }
    }
    subw_index = search_subw(b_word_e, expsetting);
    if (subw_index == -1) subw_index = add_subword_to_subv(b_word_e, expsetting);
    link_vocab_subword(word_index, subw_index, expsetting);
    free(b_word_e); free(ngram);
    return;
}

long long search_subw(char *subw, EXPSETTING *expsetting) {
    unsigned long long hash = GetWordHash(subw, expsetting -> subw_hash_size);
    SHM *tmp = (expsetting -> subw_hash)[hash].head;
    while (tmp) {
        if (!strcmp(subw, tmp -> subw)) return tmp -> si;
        tmp = tmp -> next;
    }
    return -1;
}
int add_subword_to_subv(char *subw, EXPSETTING *expsetting) {
    unsigned long long hash = 0;
    unsigned int length = strlen(subw) + 1;
    SHM *new = NULL;
    SHM *tmp = NULL;
    if (length > MAX_STRING) length = MAX_STRING;

    (expsetting -> v_subw)[expsetting -> subw_size].subw = (char *)calloc(length, sizeof(char));
    strcpy((expsetting -> v_subw)[expsetting -> subw_size].subw, subw);
    (expsetting -> v_subw)[expsetting -> subw_size].cn = 0;
    (expsetting -> v_subw)[expsetting -> subw_size].vocabs = NULL;
    (expsetting -> subw_size)++;
    
    // Reallocate memory if needed
    if ((expsetting -> subw_size) + 2 >= expsetting -> subw_max_size) {
        (expsetting -> subw_max_size) += 1000;
        expsetting -> v_subw = (VS *)realloc(expsetting -> v_subw, (expsetting -> subw_max_size) * sizeof(VS));
    }

    new = (SHM*)malloc(sizeof(SHM));
    new -> subw = (char*)malloc(sizeof(char) * length);
    strcpy(new -> subw, subw);
    new -> next = NULL;
    new -> si = (expsetting -> subw_size) - 1;

    hash = GetWordHash(subw, expsetting -> subw_hash_size);
    tmp = (expsetting -> subw_hash)[hash].head;
    if (!tmp) (expsetting -> subw_hash)[hash].head = new;
    else {
        new -> next = (expsetting -> subw_hash)[hash].head;
        (expsetting -> subw_hash)[hash].head = new;
    }
    return (expsetting -> subw_size) - 1;
}

