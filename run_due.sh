ROOT="${root_path}experiments/fedct"; # root_path must be the same as ROOT_PATH in preprocess.py
data_key="transfer_data_small";

mkdir -p ${ROOT}/transfer_models;
mkdir -p ${ROOT}/transfer_logs;

task_name="FedCTColdStart";
METRIC="_AUC";
BS=64;

model_name="DUE_VAE";
LOSS="vae";
DIM="32 32 32 32 16 16 32 16 16 16 16 16 8 8 8 8"
DIM_LIST=($DIM)
DOMAINS="Books Clothing_Shoes_and_Jewelry Home_and_Kitchen Electronics Sports_and_Outdoors Tools_and_Home_Improvement Movies_and_TV Toys_and_Games Automotive Pet_Supplies Kindle_Store Office_Products Patio_Lawn_and_Garden Grocery_and_Gourmet_Food CDs_and_Vinyl Video_Games"
DOMAIN_LIST=($DOMAINS)

reader_name="MultiTransferEnvironment";
DUE_DIM=32;

export NUM_NODES=1;
export NUM_GPUS_PER_NODE=6;
export NODE_RANK=0;
export WORLD_SIZE=$(($NUM_NODES * $NUM_GPUS_PER_NODE));
export OMP_NUM_THREADS=1;

for REG in 0.1 1.0 0 # 0.1 0 
do
    for target_domain_id in 3 1 0 # 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
    do
        for LR in 1.0 0.3 # 0.00003
        do 
            domain=${DOMAIN_LIST[${target_domain_id}]};

            CUDA_VISIBLE_DEVICES=0,2,3,4,6,7 torchrun\
                --nproc_per_node=$NUM_GPUS_PER_NODE \
                --nnodes=$NUM_NODES \
                --node_rank $NODE_RANK \
                train_transfer_multi_gpu.py\
                    --proctitle "Professor X"\
                    --model ${model_name}\
                    --reader ${reader_name}\
                    --seed 19\
                    --train_and_eval\
                    --task ${task_name}\
                    --n_round 1\
                    --optimizer "Adam"\
                    --n_worker 8\
                    --epoch 50\
                    --batch_size ${BS}\
                    --eval_batch_size 8\
                    --lr ${LR}\
                    --val_sample_p 1.0\
                    --with_val \
                    --temper 6\
                    --stop_metric ${METRIC}\
                    --step_eval -1\
                    --n_sync_per_epoch 0\
                    --model_path ${ROOT}/transfer_models/fedct_${model_name}_${domain}_lr${LR}_reg${REG}_loss${LOSS}.pkl\
                    --due_dim ${DUE_DIM}\
                    --loss ${LOSS}\
                    --l2_coef ${REG}\
                    --device_dropout_p 0\
                    --n_local_step 1\
                    --data_file ${ROOT}/${data_key}/\
                    --domain_model_file ${ROOT}/${data_key}/domain_model_logs.txt\
                    --target ${domain}\
                    --n_neg_val 100\
                    --n_neg_test 1000\
                    > ${ROOT}/transfer_logs/fedct_train_and_eval_${model_name}_${domain}_lr${LR}_reg${REG}_loss${LOSS}.log
        done
    done
done
    
    
    
    
    
    
    
    
    
    
    
