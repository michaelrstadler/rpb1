script_path='/Users/michaelstadler/Bioinformatics/Projects/rpb1/bin/train_siamese_cnn.py'
f='/Users/michaelstadler/Bioinformatics/Projects/rpb1/results/testsims_uPoNivMJ_10' # Folder containing training data
n='modelname' # model name
e=2 # num epochs
z=1 # dataset size
y=8 # num CNN layers
r=0.0001 # initial learning rate
R=0.05 # learning rate exponent
c=1 # learning rate constant for c epochs
w=0 # lower margin for curriculum learning
u=100 # upper margin for curriculum learning
p=0 # number of negative pairs (0 defaults to 5* dataset size)
b=32 # batchsize
# W='weightsfile' # initial weights file
# t rotate
# d distributed

python $script_path -f $f -n $n -e $e -z $z -y $y -r $r -R $R -c $c -w $w -u $u -p $p -b $b -t 