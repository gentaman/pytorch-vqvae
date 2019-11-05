
out_prefix='20191105_vqvae'


k[1]=235
k[2]=234

init_model=$HOME'/data2/20191028_vqvae_init/models/init/init.model.pt'
init_predictor=$HOME'/data2/20191028_vqvae_init/models/init/init.predictor.pt'

dataset='fashion-mnist'


# ls ${init_model[0]}
for ((j=1; j < 2; j++))
do
    i=1
    output_dir[$i]=$dataset'_im64_gap_k'${k[$j]}'_e100_no_recon'
    args[$i]='--recon_coeff 0.0'
    i=$((i+1))
    output_dir[$i]=$dataset'_im64_gap_k'${k[$j]}'_e100_no_pred'
    args[$i]='--gamma 0.0'
    i=$((i+1))
    output_dir[$i]=$dataset'_im64_gap_k'${k[$j]}'_e100'
    args[$i]=''
    i=$((i+1))
    max=$i
    for ((i=1; i < $max; i++))
    do
        echo ${output_dir[$i]}
        python vqvae_predict.py --data-folder ~/data2/dataset/ --dataset $dataset --image-size 64 --hidden-size 256 --num-epochs 100 --root ~/data2/${out_prefix}_vqvae/ --output-folder ${output_dir[$i]} --gap --k ${k[$j]} --device cuda --batch-size 256 --resblock-transpose ${args[$i]} --predictor ${init_predictor} --model ${init_model}
    done
done