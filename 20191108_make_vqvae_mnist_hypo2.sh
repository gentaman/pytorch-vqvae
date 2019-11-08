
# DISABLE Data Augumentation
# hypothesis 2
# kが小さくなりすぎるとおかしくなる
# k = {16, 32, 64, 128, 256, 512}

hidden_size=256
dataset='mnist'
kfold=5
out_prefix='test_20191108_vqvae_'${hidden_size}'_'${dataset}'_hypo2_cvk'${kfold}
N_max=7
k[0]=8
for ((i=1; i < N_max ; i++))
do
    k[$i]=$(( k[i-1] * 2))
    echo ${k[$i]}
done

init_model=$HOME'/data2/20191028_vqvae_init/models/init/init.model.pt'
init_predictor=$HOME'/data2/20191028_vqvae_init/models/init/init.predictor.pt'

image_size=32

# ls ${init_model[0]}
for ((j=1; j < N_max; j++))
do
    i=1
    # output_dir[$i]=$dataset'_im'${image_size}'_gap_k'${k[$j]}'_e100_no_recon'
    # args[$i]='--recon_coeff 0.0'
    # i=$((i+1))
    # output_dir[$i]=$dataset'_im'${image_size}'_gap_k'${k[$j]}'_e100_no_pred'
    # args[$i]='--gamma 0.0'
    # i=$((i+1))
    output_dir[$i]=$dataset'_im'${image_size}'_gap_k'${k[$j]}'_e100'
    args[$i]=''
    i=$((i+1))
    max=$i
    for ((i=1; i < $max; i++))
    do
        echo ${output_dir[$i]}
        python vqvae_predict.py --data-folder ~/data2/dataset/ --dataset $dataset --image-size ${image_size} --hidden-size ${hidden_size} --num-epochs 100 --root ~/data1/${out_prefix}_vqvae/ --output-folder ${output_dir[$i]} --gap --k ${k[$j]} --device cuda --batch-size 256 --resblock-transpose ${args[$i]} --predictor ${init_predictor} --model ${init_model} --off-da --kfold $kfold
    done
done