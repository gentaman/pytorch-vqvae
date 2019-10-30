

out_prefix='20191030_transpose_vqvae'


k[0]=4
init_model[0]=$HOME'/data2/20191028_vqvae_init/models/init_k4/init.model.pt'
init_predictor[0]=$HOME'/data2/20191028_vqvae_init/models/init_k4/init.predictor.pt'
k[1]=32
init_model[1]=$HOME'/data2/20191028_vqvae_init/models/init/init.model.pt'
init_predictor[1]=$HOME'/data2/20191028_vqvae_init/models/init/init.predictor.pt'

for ((j=1; j < 2; j++))
do
    i=1
    output_dir[$i]='mnist_im64_gap_k'${k[$j]}'_e100_no_recon'
    args[$i]='--recon_coeff 0.0'
    i=$((i+1))
    output_dir[$i]='mnist_im64_gap_k'${k[$j]}'_e100_no_pred'
    args[$i]='--gamma 0.0'
    i=$((i+1))
    output_dir[$i]='mnist_im64_gap_k'${k[$j]}'_e100'
    args[$i]=''
    i=$((i+1))
    max=$i

    # ls ${init_model[0]}
    for ((i=1; i < $max; i++))
    do
        echo ${output_dir[$i]}
        python vqvae_predict.py --data-folder ~/data2/dataset/ --dataset mnist --image-size 64 --hidden-size 256 --num-epochs 100 --root ~/data2/${out_prefix}_vqvae/ --output-folder ${output_dir[$i]} --gap --k ${k[$j]} --device cuda --batch-size 256 --resblock-transpose ${args[$i]} --predictor ${init_predictor[$j]} --model ${init_model[$j]}
    done
done