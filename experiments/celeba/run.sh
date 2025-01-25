mkdir -p ./save
mkdir -p ./trainlogs

method=fairgrad
#method=famo
alpha=2.0
seed=0

nohup python -u trainer.py --method=$method --seed=$seed --loc='False' --stch='True' --alpha=$alpha > trainlogs/$method-alpha$alpha-sd$seed.log 2>&1 &
