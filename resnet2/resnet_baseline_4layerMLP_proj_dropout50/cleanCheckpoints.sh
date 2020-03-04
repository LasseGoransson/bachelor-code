#ls -1 checkpoints/ | tail -n 1 | xargs -I {} mv checkpoints/{} .
rm checkpoints/*
