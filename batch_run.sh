celltypes=("EXC" "INH" "OLI" "OPC" "END" "AST" "MIC")

for celltype in ${celltypes[@]}
do
  python stn_cvx.py run $celltype True False
done
