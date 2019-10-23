

out_prefix='20191023_transpose_ae'

k[0]=4
k[1]=32
for ((j=0; j < 2; j++))
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

    for ((i=1; i < $max; i++))
    do
        echo ${output_dir[$i]}
        python ae_predict.py --data-folder ~/data2/dataset/ --dataset mnist --image-size 64 --hidden-size 256 --num-epochs 100 --root ~/data2/${out_prefix}_vqvae/ --output-folder ${output_dir[$i]} --gap --k $k --device cuda --batch-size 256 --resblock-transpose $args
    done
done