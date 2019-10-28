

out_prefix='20191028_transpose_ae'

init_model=$HOME'/data2/20191028_vqvae_init/models/init/init.model.pt'
init_predictor=$HOME'/data2/20191028_vqvae_init/models/init/init.predictor.pt'

i=1
output_dir[$i]='mnist_im64_gap_k_e100_no_recon'
args[$i]='--recon_coeff 0.0'
i=$((i+1))
output_dir[$i]='mnist_im64_gap_k_e100_no_pred'
args[$i]='--gamma 0.0'
i=$((i+1))
output_dir[$i]='mnist_im64_gap_k_e100'
args[$i]=''
i=$((i+1))
max=$i

for ((i=1; i < $max; i++))
do
    echo ${output_dir[$i]}
    python ae_predict.py --data-folder ~/data2/dataset/ --dataset mnist --image-size 64 --hidden-size 256 --num-epochs 100 --root ~/data2/${out_prefix}_vqvae/ --output-folder ${output_dir[$i]} --gap --device cuda --batch-size 256 --resblock-transpose ${args[$i]} --predictor $init_predictor --model $init_model
done