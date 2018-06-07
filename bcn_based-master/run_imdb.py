import os


gpu = 0
task = 'IMDB'
batch_sizes = [32]
num_class = 2
dropout_list = [0.2]
max_norms = [5]
num_units_list = [5]
lr=0.001
resume = 0
epoch = 8
write_to_file = 0
test= 1
# max_example = 10000



pyfile = 'main.py'
pyfile = 'test.py'
for batch_size in batch_sizes:
	for max_norm in max_norms:
		for dropout in dropout_list[::-1]:
			for num_units in num_units_list[::-1]:
				temp = ' --task '+task +' --batch_size '+str(batch_size)+' --dropout '+str(dropout) \
				+ ' --num_class '+str(num_class)+' --lr '+str(lr) +' --num_units '+str(num_units) + ' --max_norm '+str(max_norm)
				# + ' --max_example '+str(max_example)


				sp_batch_size = 32
				model_file_name = 'model_task_'+task +'_batch_size_'+str(batch_size)+'_dropout_'+str(dropout) \
				+ '_num_class_'+str(num_class)+'_lr_'+str(lr) \
				+ '_max_norm_'+str(max_norm)\
				+'_num_units_'+str(num_units)+'_best.pth.tar'

				# model_file_name = 'model_task_'+task +'_batch_size_'+str(batch_size)+'_dropout_'+str(dropout) \
				# + '_num_class_'+str(num_class)+'_lr_'+str(lr) \
				# + '_max_norm_'+str(max_norm)\
				# +'_epochs_'+str(epochs)\
				# self.config.save_path +'Models/' + self.config.save_model\
				# +'_num_units_'+str(num_units)+'_best.pth.tar'


				run_command = ' python3 '+pyfile+' --gpu '+str(gpu)+temp
				if test==1: 
					model_file_name = 'epoch_'+str(epoch)+'_'+model_file_name
					run_command += ' --no_train '
				
				run_command+= ' --save_model '+model_file_name
				if resume==1: run_command+=' --resume '+model_file_name
				
				if pyfile != 'test.py' and  write_to_file ==1:run_command += ' >> ../bcn_output/' + task+'/'+model_file_name+'_output.txt'
				print (run_command)
				os.system(run_command)
			exit()

#python3 main.py --gpu 2 --no_train  --task IMDB --batch_size 32 --dropout 0.3 --num_class 2 --lr 0.001 --num_units 5 --max_norm 3.0 --save_model model_task_IMDB_batch_size_32_dropout_0.3_num_class_2_lr_0.001_num_units_5_best.pth.tar