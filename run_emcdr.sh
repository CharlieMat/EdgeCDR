ROOT="${root_path}experiments/fedct"; # root_path must be the same as ROOT_PATH in preprocess.py

mkdir -p ${ROOT}/transfer_models;
mkdir -p ${ROOT}/transfer_logs;

task_name="ColdStartTopK";
METRIC="_AUC";
device=0;

model_name="Matching_Linear";
BS=256;
LOSS="rmse";
DIM="32 32 32 32 16 16 32 16 16 16 16 16 8 8 8 8"
DIM_LIST=($DIM)
DOMAINS="Books Clothing_Shoes_and_Jewelry Home_and_Kitchen Electronics Sports_and_Outdoors Tools_and_Home_Improvement Movies_and_TV Toys_and_Games Automotive Pet_Supplies Kindle_Store Office_Products Patio_Lawn_and_Garden Grocery_and_Gourmet_Food CDs_and_Vinyl Video_Games"
DOMAIN_LIST=($DOMAINS)

reader_name="ColdStartTransferEnvironment";

for target_domain_id in 0 1 3 # 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
do
    for LR in 0.00001 0.00003
    do 
        for REG in 1.0 0.1 0 # 10.0 0.1 0.01
        do
            domain=${DOMAIN_LIST[${target_domain_id}]};

            python train_transfer.py\
                --proctitle "Captain Marvel"\
                --model ${model_name}\
                --reader ${reader_name}\
                --seed 19\
                --cuda ${device}\
                --train_and_eval\
                --task ${task_name}\
                --n_round 1\
                --optimizer "Adam"\
                --n_worker 4\
                --epoch 30\
                --batch_size ${BS}\
                --eval_batch_size 8\
                --lr ${LR}\
                --val_sample_p 1.0\
                --with_val \
                --temper 6\
                --stop_metric ${METRIC}\
                --model_path ${ROOT}/transfer_models/fedct_${model_name}_small_${domain}_lr${LR}_reg${REG}_loss${LOSS}.pkl\
                --loss ${LOSS}\
                --l2_coef ${REG}\
                --data_file ${ROOT}/transfer_data_small/\
                --domain_model_file ${ROOT}/transfer_data_small/domain_model_logs.txt\
                --target ${domain}\
                --n_neg_val 100\
                --n_neg_test 1000\
                > ${ROOT}/transfer_logs/fedct_train_and_eval_${model_name}_small_${domain}_lr${LR}_reg${REG}_loss${loss}.log
        done
    done
done
    
    
    
    
    
    
    
    
    
    
    
