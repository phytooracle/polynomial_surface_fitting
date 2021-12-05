#!/bin/bash
#SBATCH --job-name=3D-Plant-Centering
#SBATCH --account=lyons-lab
#SBATCH --partition=standard
#SBATCH --ntasks=15
#SBATCH --ntasks-per-node=15
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --gres=gpu:0
#SBATCH --time=10:00:00
#SBATCH -o  /xdisk/ericlyons/ariyanzarei/plant_centering/logs/%x_%j.out

. pipeline.config

echo "::: Processing $scan_date from season $season"

echo "::: Cleaning up the directories."

rm -r $data_path/*
rm -r $result_base_path/*

echo "::: Downloading the input data."

ssh filexfer "cd $data_path; iget -rKVPT $cyverse_input_path"

echo "::: Decompressing the input data."

cd $data_path
tar -xvf ${scan_date}_combined_pointclouds_plants.tar
rm ${scan_date}_combined_pointclouds_plants.tar

# echo "::: Beginning the preprocessing, i.e. down sampling, merging and initializing alignment."

# mkdir -p $preprocessing_output
# mkdir -p $alignment_output

# cd $preprocessing_input

# count=$(ls -d * |wc -l)
# count=$((count-1))

# echo "#!/bin/bash
# #SBATCH --job-name=3D-Batch-Preprocessing
# #SBATCH --account=lyons-lab
# #SBATCH --partition=standard
# #SBATCH --ntasks=15
# #SBATCH --ntasks-per-node=15
# #SBATCH --nodes=1
# #SBATCH --mem=100GB
# #SBATCH --gres=gpu:0
# #SBATCH --time=00:20:00
# #SBATCH --array 0-$count
# #SBATCH --wait
# #SBATCH -o /xdisk/kobus/ariyanzarei/3d_geocorrection/logs/%x_%A_%a.out

# code=$preprocessing_code

# cd $preprocessing_input
# folders=(\$(ls -d */))
# dir_name=\${folders[\${SLURM_ARRAY_TASK_ID}]}

# singularity exec $image python3 $preprocessing_code -i $preprocessing_input/\$dir_name -o $preprocessing_output -l $lids">tmp.sh

# chmod +x tmp.sh
# sbatch tmp.sh
# rm tmp.sh

# echo "::: Beginning the sequential alignment process."

# singularity exec $image python3 $alignment_code -i $preprocessing_output -o $alignment_output

# echo "::: Preprocessing finished."

# echo "::: Compressing the preprocessing results."

# cd $preprocessing_output

# tar -cvf ${scan_date}_east_preprocessed.tar east/
# tar -cvf ${scan_date}_east_downsampled_preprocessed.tar east_downsampled/
# tar -cvf ${scan_date}_west_preprocessed.tar west/
# tar -cvf ${scan_date}_west_downsampled_preprocessed.tar west_downsampled/
# tar -cvf ${scan_date}_merged_preprocessed.tar merged/
# tar -cvf ${scan_date}_merged_downsampled_preprocessed.tar merged_downsampled/
# tar -cvf ${scan_date}_metadata.tar metadata/

# cd $alignment_output

# tar -cvf ${scan_date}_east_aligned.tar east/
# tar -cvf ${scan_date}_east_downsampled_aligned.tar east_downsampled/
# tar -cvf ${scan_date}_west_aligned.tar west/
# tar -cvf ${scan_date}_west_downsampled_aligned.tar west_downsampled/
# tar -cvf ${scan_date}_merged_aligned.tar merged/
# tar -cvf ${scan_date}_merged_downsampled_aligned.tar merged_downsampled/
# tar -cvf ${scan_date}_metadata.tar metadata/

# echo "::: Uploading the compressed preprocessing results."
# ssh filexfer "cd $preprocessing_output; imkdir -p ${cyverse_base_preprocessing_output}/${scan_date}/preprocessing; iput -fKVPT ${scan_date}_merged_downsampled_preprocessed.tar ${cyverse_base_preprocessing_output}/${scan_date}/preprocessing/.;iput -fKVPT ${scan_date}_metadata.tar ${cyverse_base_preprocessing_output}/${scan_date}/preprocessing/."
