# Bi-Attentive Classification Network


1) Train ori full classifier: python3 main.py --gpu 2 --model_file_name 'full_ori.pth.tar' (on server margo 2, 91.04%, for rt: first commented paragraph in run_RT.py)
2) Train slectors initialized with full_ori.pth.tar classifier: python3 imdb_main.py --gpu 2 --model_file_name model_sparsity_0.00075_coherent_2.0.pth.tar --load_model 1 --classifier_file_name full_ori.pth.tar

 3) For sparsity_list  =  [ 0.5,  0.015, 0.01, 0.00075, 0.0005,], coherent_list = [2.0] but for [0.00075, 0.0005], [1.0, 2.0], do: save selector outputs: python3 imdb_main.py --gpu 2 --model_file_name dummy_model.pth.tar --load_model 2 --classifier_file_name modle_sparsity_0.0005_coherent_1.0.pth.tar --sparsity 0.0005 --coherent 2.0
 --selector_file_name modle_sparsity_0.0005_coherent_1.0.pth.tar --save_selection 1 --batch_size 32 (server tested: nlp 2, margo 2, use run_imdb.py to run it, shuffle should be True) 

 4) Train WAG full classifier run_imdb_WAG_full.py (NLP 2, 90.74%)  python3 imdb_main_WAG_full_classifier.py --gpu 2 --WAG  --model_file_name full_WAG_classifier_nlp_dp_0.2.pth.tar --load_model 1 --sparsity 0 --coherent 0 --classifier_file_name ful
l_ori.pth.tar
 
 5) python3 imdb_main.py --gpu 2 --classifier_file_name full_WAG_classifier_nlp.pth.tar --WAG  --model_file_name WAG_model_sparsity_0.0005_coherent_2.0.pth.tar --load_model 1 --sparsit
y 0.0005 --coherent 2.0 --classifier_file_name full_WAG_classifier_nlp.pth.tar (91.19%)

 
 
 Sample code for spped up test; python3 imdb_test.py --gpu 2 --load_model 2 --classifier_file_name modle_sparsity_0.00075_coherent_1.0.pth.tar --selector_file_name modle_sparsity_0.00075_coherent_1.0.pth.tar






