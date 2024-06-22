celltypes=("EXC" "INH" "OLI" "OPC" "END" "AST" "MIC")
keggs=('04722' '04064' '04151' '04010' '04020')
fc='False'
fc_print='F'
kegg_only='False'

for kegg in ${keggs[@]}
do
  if [ $kegg_only == 'True' ]
  then
    rm -f Results/NF/Parameters/*$fc_print"$kegg"ONLY.txt
  else
    rm -f Results/NF/Parameters/*$fc_print$kegg.txt
  fi
  for celltype in ${celltypes[@]}
  do
    python stn_cvx.py plot $celltype $fc $kegg $kegg_only
  done
done
