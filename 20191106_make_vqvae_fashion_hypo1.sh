
# DISABLE Data Augumentation
# hypothesis 1
# hidden_size と k の関係によって、過学習の度合いが変わる
# hidden_size > k ==>: EmpLoss(x, z_e) >= EmpLoss(x, z_e)
# hidden_size < k ==>: EmpLoss(x, z_e) <= EmpLoss(x, z_e)
# hidden_size = k ==>: EmpLoss(x, z_e) ~= EmpLoss(x, z_e)

hidden_size=256
out_prefix='20191106_vqvae_'${hidden_size}
k[1]=$(( hidden_size / 2 ))
k[2]=${hidden_size}
d=$(( k[2] - k[1] ))
k[3]=$(( hidden_size + d))

init_model=$HOME'/data2/20191028_vqvae_init/models/init/init.model.pt'
init_predictor=$HOME'/data2/20191028_vqvae_init/models/init/init.predictor.pt'

image_size=32
dataset='fashion-mnist'

# ls ${init_model[0]}
for ((j=1; j < 4; j++))
do
    i=1
    output_dir[$i]=$dataset'_im'${image_size}'_gap_k'${k[$j]}'_e100_no_recon'
    args[$i]='--recon_coeff 0.0'
    i=$((i+1))
    output_dir[$i]=$dataset'_im'${image_size}'_gap_k'${k[$j]}'_e100_no_pred'
    args[$i]='--gamma 0.0'
    i=$((i+1))
    output_dir[$i]=$dataset'_im'${image_size}'_gap_k'${k[$j]}'_e100'
    args[$i]=''
    i=$((i+1))
    max=$i
    for ((i=1; i < $max; i++))
    do
        echo ${output_dir[$i]}
        python vqvae_predict.py --data-folder ~/data2/dataset/ --dataset $dataset --image-size ${image_size} --hidden-size ${hidden_size} --num-epochs 100 --root ~/data2/${out_prefix}_vqvae/ --output-folder ${output_dir[$i]} --gap --k ${k[$j]} --device cuda --batch-size 256 --resblock-transpose ${args[$i]} --predictor ${init_predictor} --model ${init_model} --off-da
    done
done