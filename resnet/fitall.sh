cd resnet50_freeze_stdMLP/
#python fitResNet50.py && bash cleanCheckpoints.sh
cd ../
cd resnet50_freeze_stdMLP_dropout
#python fitResNet50.py && bash cleanCheckpoints.sh
cd ../ 
cd resnet50_freeze_stdMLP_proj
#python fitResNet50.py && bash cleanCheckpoints.sh
cd ../
cd resnet50_nonfreeze_stdMLP
python fitResNet50.py && bash cleanCheckpoints.sh
cd ../
cd resnet50_nonfreeze_stdMLP_dropout
python fitResNet50.py && bash cleanCheckpoints.sh
cd ../
cd resnet50_nonfreeze_stdMLP_proj
python fitResNet50.py && bash cleanCheckpoints.sh
