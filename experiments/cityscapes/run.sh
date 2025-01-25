mkdir -p ./save
mkdir -p ./trainlogs

method=famo
#method=fairgrad
alpha=2.0
seed=0

loc=False
stch=False

#nohup 
python -u trainer.py --method=$method --seed=$seed --loc=$loc --stch=$stch --alpha=$alpha  > trainlogs/$method-alpha$alpha-sd$seed.log 2>&1 &
