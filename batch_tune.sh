celltypes=("EXC" "INH" "OLI" "OPC" "END" "AST" "MIC")

for celltype in ${celltypes[@]}
do
  sbatch tune.sh $celltype 04722
  sbatch tune.sh $celltype 04064
  sbatch tune.sh $celltype 04151
  sbatch tune.sh $celltype 04010
  sbatch tune.sh $celltype 04020
done
