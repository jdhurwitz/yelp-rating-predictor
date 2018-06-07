import os
gpu = 1
task = 'tweet'
batch_size = 256
num_class = 2
dropout = 0.1
lr=0.001
pyfile = 'main.py'
pyfile = 'test.py'


run_command = ' python3 '+pyfile+' --gpu '+str(gpu)+' --task '+task +' --batch_size '+str(batch_size)+' --dropout '+str(dropout) \
+ ' --num_class '+str(num_class)+' --lr '+str(lr)
run_command += ' >> ../bcn_output/' + task+'/'+task+'_output.txt'
print (run_command)
os.system(run_command)