

out_prefix='20191102_vqvae'


k=32
init_model=$HOME'/data2/20191102_vqvae_init/models/init_off-bias/init.model.pt'
init_predictor=$HOME'/data2/20191102_vqvae_init/models/init_off-bias/init.predictor.pt'

i=1
output_dir[$i]='fashion-mnist_im64_gap_k'${k}'_e100_no_recon'
args[$i]='--recon_coeff 0.0'
i=$((i+1))
output_dir[$i]='fashion-mnist_im64_gap_k'${k}'_e100_no_pred'
args[$i]='--gamma 0.0'
i=$((i+1))
output_dir[$i]='fashion-mnist_im64_gap_k'${k}'_e100'
args[$i]=''
i=$((i+1))
max=$i

# ls ${init_model[0]}
for ((i=1; i < $max; i++))
do
    echo ${output_dir[$i]}
    python vqvae_predict.py --data-folder ~/data2/dataset/ --dataset fashion-mnist --image-size 64 --hidden-size 256 --num-epochs 100 --root ~/data2/${out_prefix}_vqvae/ --output-folder ${output_dir[$i]} --gap --k ${k} --device cuda --batch-size 256 --resblock-transpose ${args[$i]} --predictor ${init_predictor} --model ${init_model} --off-bn --off-bias
done