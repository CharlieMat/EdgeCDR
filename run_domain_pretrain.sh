ROOT="${root_path}experiments/fedct";

mkdir -p ${ROOT}/env_models;
mkdir -p ${ROOT}/env_logs;

task_name="NextItemTopK";
METRIC="_AUC";
device=2;

model_name="MF";
BS=2048;
LOSS="pairwisebpr";
DIM="32 32 32 32 16 16 32 16 16 16 16 16 8 8 8 8"
DIM_LIST=($DIM)
DOMAINS="Books Clothing_Shoes_and_Jewelry Home_and_Kitchen Electronics Sports_and_Outdoors Tools_and_Home_Improvement Movies_and_TV Toys_and_Games Automotive Pet_Supplies Kindle_Store Office_Products Patio_Lawn_and_Garden Grocery_and_Gourmet_Food CDs_and_Vinyl Video_Games"
DOMAIN_LIST=($DOMAINS)

reader_name="NextItemReader";
NNEG=2;

for domain_id in 5 7 11
do
    domain=${DOMAIN_LIST[${domain_id}]};

    for LR in 0.000003 0.00001
    do
        for REG in 1.0 0.1 0.01
        do
            python pretrain_env.py\
                --proctitle "Doctor Strange"\
                --seed 19\
                --model ${model_name}\
                --task ${task_name}\
                --n_round 1\
                --train_and_eval\
                --optimizer "Adam"\
                --cuda ${device}\
                --epoch 30\
                --batch_size ${BS}\
                --eval_batch_size 16\
                --lr ${LR}\
                --val_sample_p 0.5\
                --with_val \
                --temper 10\
                --stop_metric ${METRIC}\
                --save_reader\
                --n_worker 8\
                --reader_path ${ROOT}/domain_data/reader/${reader_name}_${domain}.rdr\
                --model_path ${ROOT}/env_models/fedct_${model_name}_${domain}_lr${LR}_reg${REG}_${LOSS}.pkl\
                --loss ${LOSS}\
                --l2_coef ${REG}\
                --emb_size ${DIM_LIST[${domain_id}]}\
                > ${ROOT}/env_logs/fedct_train_and_eval_${model_name}_${domain}_lr${LR}_reg${REG}_loss${LOSS}.log
        done
    done
done
    
    
    
    
    
    
    
    
    
    
    
