import numpy as np
import subprocess
import os
import time
import sys
import shutil

# kaldi, htk, srilm, caffe, mallet, matlab

default_zrc_dir = '/home/c2tao/matdnn/'
default_mat_dir = os.path.join(default_zrc_dir, 'mat')
default_mdnn_dir = os.path.join(default_zrc_dir, 'mdnn')
default_ivector_dir = os.path.join(default_zrc_dir, 'ivector')
default_temp_dir = os.path.join(default_zrc_dir, 'temp')
default_zrst_dir = os.path.join(default_mat_dir, 'zrst')
default_matlab_dir = os.path.join(default_zrst_dir, 'matlab')
sys.path.insert(1, default_zrst_dir)

def lsdir_full(path):
    return [os.path.join(path, f) for f in sorted(os.listdir(path))]

def mkdir_for_file(path):
    # makes parent dirs leading to file
    path_to_file = os.path.split(path)[0]
    if not os.path.exists(path_to_file):
        os.makedirs(path_to_file)
    
def mkdir_for_dir(path):
    # overwrite dir if it exists and make parent dirs to dir
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def run_parallel(func, arg_list): 
    from joblib import Parallel, delayed
    import multiprocessing

    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(func)(*args) for args in arg_list)
    return results

def get_arg_from_dir(path):
    # get parameters from name of directory
    args = []
    for d in os.listdir(path):
        args.append(map(int, d.split('.')[0].split('_')))
    params = zip(*args)
    return map(lambda p: sorted(list(set(p))), params)

def extract_mfcc(input_wav_dir, output_feature_file, output_feature_dir):
    # secure destination
    mkdir_for_file(output_feature_file)
    mkdir_for_dir(output_feature_dir)
    
    # extract mfcc in single file format and dir format
    import util
    mkdir_for_dir(default_temp_dir)
    util.make_feature(input_wav_dir, default_temp_dir)

    def write_feat(feat, feat_id, outfile):
        outfile.write(feat_id+'.wav\n')
        for i in range(feat.shape[0]):
            fline = '{:04d} {:04d} #' + ' {:f}'*feat.shape[1] +'\n'
            outfile.write(fline.format(i, i+1, *feat[i]))
        outfile.write('\n')
    
    with open(output_feature_file, 'w') as outfile:
        for f in sorted(os.listdir(default_temp_dir)):
            feat = util.read_feature(os.path.join(default_temp_dir, f))
            write_feat(feat, f[:-4], outfile)

    for f in sorted(os.listdir(default_temp_dir)):
        feat = util.read_feature(os.path.join(default_temp_dir, f))
        util.write_feature(feat, os.path.join(output_feature_dir, f))

    # cleanup, remove large files
    mkdir_for_dir(default_temp_dir)

def extract_ivector(input_wav_dir, output_ivector_file):
    # secure destination
    mkdir_for_file(output_ivector_file)

    # setup required files
    def make_scp(path_wav, path_scp):
        wav_list = []
        for f in os.listdir(path_wav):
            if '.wav' == f[-4:]:
                wav_list.append(f)
                    
        wav_list = sorted(wav_list)
        with open(path_scp, 'w') as fscp:
            for wav in wav_list:
                fscp.write(wav[:-4]+' '+os.path.join(path_wav, wav)+'\n')
    scp_file = os.path.join(default_ivector_dir, 'material', 'wav.scp')
    make_scp(input_wav_dir, scp_file)

    # extract ivector file
    p = subprocess.Popen('./run.sh', cwd = default_ivector_dir, shell=True).wait()

    # copy ivector to destination
    ivector_file = os.path.join(default_ivector_dir, 'ivector', 'ivector.ark')
    shutil.copyfile(ivector_file, output_ivector_file)

    # cleanup, remove large files
    mkdir_for_dir(os.path.join(default_ivector_dir, 'ivector'))
    mkdir_for_dir(os.path.join(default_ivector_dir, 'log'))
    mkdir_for_dir(os.path.join(default_ivector_dir, 'feat'))


def extract_init(input_wav_dir, cluster_number, output_init_file):
    # secure destination
    mkdir_for_file(output_init_file)
    import util
    init_word_file = output_init_file + '_word'

    # extract init file
    cmd = 'matlab -nosplash -nodesktop -nojvm -r "clusterDetection_function {0} {1} {2}"'.format(input_wav_dir, cluster_number, init_word_file)
    p = subprocess.Popen(cmd, cwd = default_matlab_dir, shell=True)
    p.wait() 
    util.flat_dumpfile(init_word_file, output_init_file)
    
    # cleanup, remove large files
    p = subprocess.Popen('rm *temp*', cwd = default_matlab_dir, shell=True)
    p.wait() 

def train_tokenizer(input_init_file, input_feature_dir, state_number, output_model_dir):
    # secure destination
    mkdir_for_file(output_model_dir)
    
    # train tokenizer
    import asr
    A = asr.ASR(corpus='UNUSED', 
        target=output_model_dir, 
        nState=state_number, 
        nFeature=39, 
        features = lsdir_full(input_feature_dir),
        dump = input_init_file, 
        pad_silence=False)

    A.initialization('a')
    A.iteration('a')
    A.iteration('a_keep')
    A.iteration('a_keep')
    A.iteration('a_keep')

    A.iteration('a_mix')
    A.iteration('a_keep')
    A.iteration('a_keep')
    A.iteration('a_keep')

    A.iteration('a_mix')
    A.iteration('a_keep')
    A.iteration('a_keep')
    A.iteration('a_keep')
    A.iteration('a_keep')
    A.iteration('a_keep')

        
def reinforce_label(input_tokenizer_dir,  output_rein_dir, cluster_list = None, state_list = None):
    # secure output dir
    mkdir_for_dir(output_rein_dir)

    #get cluster_list and state_list from input dir
    if not cluster_list or not state_list:
        cluster_list, state_list = get_arg_from_dir(input_tokenizer_dir)
    print cluster_list, state_list

    # setup working dir for reinforcement and copy result files over 
    mkdir_for_dir(os.path.join(default_mat_dir, 'pattern'))
    mkdir_for_dir(os.path.join(default_mat_dir, 'init'))
    mkdir_for_dir(os.path.join(default_mat_dir, 'exp'))
            
    #for d in os.listdir(input_tokenizer_dir):
    for state in state_list:
        for cluster in cluster_list:
            d = str(cluster)+'_'+str(state)
            target_mlf = os.path.join(default_mat_dir, 'pattern', 'merge_'+d, 'result', 'result.mlf')
            source_mlf = os.path.join(input_tokenizer_dir, d, 'result', 'result.mlf')
            mkdir_for_file(target_mlf)
            shutil.copy(source_mlf, target_mlf)
    
    arg1 = ' '.join(map(str, cluster_list))
    arg2 = ' '.join(map(str, state_list))
    p = subprocess.Popen('./MR_commandline.sh "{}" "{}"'.format(arg1, arg2), cwd = default_mat_dir, shell=True).wait()

    # copy files to output dir
    subprocess.Popen('mv * {}'.format(output_rein_dir), cwd = os.path.join(default_mat_dir, 'init'), shell=True).wait()
     
    # cleanup large files
    mkdir_for_dir(os.path.join(default_mat_dir, 'pattern'))
    mkdir_for_dir(os.path.join(default_mat_dir, 'init'))
    mkdir_for_dir(os.path.join(default_mat_dir, 'exp'))

def train_neuralnet(input_tokenizer):
    pass 
    
class Init(object):
    def __init__(self, init_file = None, wav_dir = None):
        self.init_file = init_file
        self.wav_dir = wav_dir
        self.cluster_number = int(os.path.split()[-1].split('.')[0])

    def extract(self):
        self.cluster_number = cluster_number
        extract_init(self.wav_dir, cluster_number, self.init_file)

class Tokenizer(object):
    def __init__(self,  model_dir = None, init_file = None, feature_dir = None):
        self.init_file = init_file
        self.model_dir = model_dir
        self.feature_dir = feature_dir

    def train(self, state_number):
        self.state_number = state_number
        train_tokenizer(self.feature_dir, self.init_file, state_number, self.model_dir)
         
class MAT(object):
    '''
    def __init__(self, model_dir = None, init_dir = None, feature_dir = None, state_list, cluster_list):
        self.path_mat = path_mat
        self.work_mat = work_mat
        self.num_states = sorted(num_states)
        self.num_tokens = sorted(num_tokens)
    '''
    def extract_labels(self):
        pass 

    def reinforce_label(self):
        self.append_path(self.work_zrst)
        self.mkdir_for_dir(os.path.join(self.work_mat, 'pattern'))

    
def test_api_function():
    root = '/home/c2tao/ZRC_revision/matdnn/files/'
    wav_dir = '/home/c2tao/timit_mini_corpus/' 

    ivector_file = root + 'ivector.ark'
    feature_file = root + 'feat.mfc'
    feature_dir = root + 'mfc/'
    model_dir = root + 'token/'
    init_dir = root +'init/'
    rein_dir = root +'rein/'
    extract_ivector(wav_dir, ivector_file)
    extract_mfcc(wav_dir, feature_file, feature_dir)
    extract_init(wav_dir, 10, init_dir+'10.txt')
    extract_init(wav_dir, 20, init_dir+'20.txt')
    run_parallel(train_tokenizer, [
                (init_dir+'10.txt', feature_dir, 3,  model_dir+'10_3/'),
                (init_dir+'10.txt', feature_dir, 5,  model_dir+'10_5/'),
                (init_dir+'20.txt', feature_dir, 3,  model_dir+'20_3/'),
                (init_dir+'20.txt', feature_dir, 5,  model_dir+'20_5/')])
    #train_tokenizer(init_dir+'10.txt', feature_dir, 3,  model_dir+'10_3/')
    #train_tokenizer(init_dir+'10.txt', feature_dir, 5,  model_dir+'10_5/')
    #train_tokenizer(init_dir+'20.txt', feature_dir, 3,  model_dir+'20_3/')
    #train_tokenizer(init_dir+'20.txt', feature_dir, 5,  model_dir+'20_5/')
    reinforce_label(model_dir, rein_dir, cluster_list = [10], state_list = [3 ,5])
    reinforce_label(model_dir, rein_dir)

if __name__=='__main__':
    root = '/home/c2tao/matdnn_files/'
    wav_dir = '/home/c2tao/timit_mini_corpus/' 

    ivector_file = root + 'ivector.ark'
    feature_file = root + 'feat.mfc'
    feature_dir = root + 'mfc/'
    model_dir = root + 'token/'
    init_dir = root +'init/'
    rein_dir = root +'rein/'
    extract_ivector(wav_dir, ivector_file)
    extract_mfcc(wav_dir, feature_file, feature_dir)
    extract_init(wav_dir, 10, init_dir+'10.txt')
    extract_init(wav_dir, 20, init_dir+'20.txt')
    run_parallel(train_tokenizer, [
                (init_dir+'10.txt', feature_dir, 3,  model_dir+'10_3/'),
                (init_dir+'10.txt', feature_dir, 5,  model_dir+'10_5/'),
                (init_dir+'20.txt', feature_dir, 3,  model_dir+'20_3/'),
                (init_dir+'20.txt', feature_dir, 5,  model_dir+'20_5/')])
    #train_tokenizer(init_dir+'10.txt', feature_dir, 3,  model_dir+'10_3/')
    #train_tokenizer(init_dir+'10.txt', feature_dir, 5,  model_dir+'10_5/')
    #train_tokenizer(init_dir+'20.txt', feature_dir, 3,  model_dir+'20_3/')
    #train_tokenizer(init_dir+'20.txt', feature_dir, 5,  model_dir+'20_5/')
    reinforce_label(model_dir, rein_dir, cluster_list = [10], state_list = [3 ,5])
    reinforce_label(model_dir, rein_dir)


