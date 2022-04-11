ROOT="${root_path}experiments/fedct"; # root_path must be the same as ROOT_PATH in preprocess.py

mkdir -p ${ROOT}/domain_data/reader;

DOMAINS="Books Clothing_Shoes_and_Jewelry Home_and_Kitchen Electronics Sports_and_Outdoors Tools_and_Home_Improvement Movies_and_TV Toys_and_Games Automotive Pet_Supplies Kindle_Store Office_Products Patio_Lawn_and_Garden Grocery_and_Gourmet_Food CDs_and_Vinyl Video_Games"
DOMAIN_LIST=($DOMAINS)

reader_name="NextItemReader";
NNEG=2;
NNEG_VAL=100;
NNEG_TEST=1000;

for domain_id in 0 1 2 3
do
    domain=${DOMAIN_LIST[${domain_id}]};

    python prepare_reader.py\
        --reader ${reader_name}\
        --reader_path ${ROOT}/domain_data/reader/${reader_name}_${domain}.rdr\
        --seed 19\
        --data_file ${ROOT}/domain_data/tsv_data/${domain}_\
        --n_worker 4\
        --user_meta_data ${ROOT}/domain_data/meta_data/${domain}_user.meta\
        --item_meta_data ${ROOT}/domain_data/meta_data/${domain}_item.meta\
        --user_fields_meta_file ${ROOT}/domain_data/meta_data/${domain}_user_fields.meta\
        --item_fields_meta_file ${ROOT}/domain_data/meta_data/${domain}_item_fields.meta\
        --user_fields_vocab_file ${ROOT}/domain_data/meta_data/${domain}_user_fields.vocab\
        --item_fields_vocab_file ${ROOT}/domain_data/meta_data/${domain}_item_fields.vocab\
        --n_neg ${NNEG}\
        --n_neg_val 100\
        --n_neg_test 1000
done
    
    
    
    
    
    
    
    
    
    
    
