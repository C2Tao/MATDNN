import numpy as np
import subprocess
import os
import time
import sys
import shutil

default_mat_dir = '/home/c2tao/ZRC_revision/mat/'
default_mdnn_dir = '/home/c2tao/ZRC_revision/mdnn/'
default_ivector_dir = '/home/c2tao/ZRC_revision/ivector/'
default_zrst_dir = os.path.join(default_mat_dir, 'zrst')
default_matlab_dir = os.path.join(default_zrst_dir, 'matlab')
sys.path.insert(1, default_zrst_dir)

class Fobject(object):
    @staticmethod
    def lsdir_full(path):
        return [os.path.join(path, f) for f in sorted(os.listdir(path))]

    @staticmethod
    def mkdir_for_file(path):
        # makes parent dirs leading to file
        path_to_file = os.path.split(path)[0]
        if not os.path.exists(path_to_file):
            os.makedirs(path_to_file)
        
    @staticmethod
    def mkdir_for_dir(path):
        # overwrite dir if it exists and make parent dirs to dir
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
    
    @staticmethod
    def run_parallel(func, arg_list): 
        from joblib import Parallel, delayed
        import multiprocessing

        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores)(delayed(func)(i) for i in arg_list)
        return results

class Init(object):
    def __init__(self, init_file = None, wav_dir = None):
        self.init_file = init_file
        self.wav_dir = wav_dir
    
    @staticmethod
    def extract_init(input_wav_dir, cluster_number, output_init_file):
        Fobject.mkdir_for_file(output_init_file)
        import util
        init_word_file = output_init_file + '_word'

        cmd = 'matlab -nosplash -nodesktop -nojvm -r "clusterDetection_function {0} {1} {2}"'.format(input_wav_dir, cluster_number, init_word_file)
        p = subprocess.Popen(cmd, cwd = default_matlab_dir, shell=True)
        p.wait() 
        util.flat_dumpfile(init_word_file, output_init_file)
    
    def extract(self, cluster_number):
        self.cluster_number = cluster_number
        self.extract_init(self.wav_dir, cluster_number, self.init_file)
    

class Tokenizer(object):
    def __init__(self,  model_dir = None, init_file = None, feature_dir = None):
        self.init_file = init_file
        self.model_dir = model_dir
        self.feature_dir = feature_dir
         
    @staticmethod 
    def train_tokenizer(input_feature_dir, input_init_file, state_number, output_model_dir):
        Fobject.mkdir_for_file(output_model_dir)
        import asr

        A = asr.ASR(corpus='UNUSED', 
            target=output_model_dir, 
            nState=state_number, 
            nFeature=39, 
            features = Fobject.lsdir_full(input_feature_dir),
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

    def train(self, state_number):
        self.state_number = state_number
        self.train_tokenizer(self.feature_dir, self.init_file, state_number, self.model_dir)
         
class MAT(object):
    def __init__(self, model_dir = None, state_list, cluster_list):
    def __init__(self,  model_dir = None, init_file = None, feature_dir = None):
        self.path_mat = path_mat
        self.work_mat = work_mat
        self.num_states = sorted(num_states)
        self.num_tokens = sorted(num_tokens)

    def extract_label(self):
        self 
    

    def reinforce_label(self):
        self.append_path(self.work_zrst)
        self.mkdir_for_dir(os.path.join(self.work_mat, 'pattern'))


class API(object):

    def append_path(self, path):
        sys.path.insert(1, path)

    def lsdir_full(self, path):
        return [os.path.join(path, f) for f in sorted(os.listdir(path))]

    def mkdir_for_file(self, path):
        # makes parent dirs leading to file
        path_to_file = os.path.split(path)[0]
        if not os.path.exists(path_to_file):
            os.makedirs(path_to_file)
        
    def mkdir_for_dir(self, path):
        # overwrite dir if it exists and make parent dirs to dir
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    def extract_ivector(self):
        self.mkdir_for_file(self.path_ivector)
        def make_scp(path_wav, path_scp):
            wav_list = []
            for f in os.listdir(path_wav):
                if '.wav' == f[-4:]:
                    wav_list.append(f)
                        
            wav_list = sorted(wav_list)
            with open(path_scp, 'w') as fscp:
                for wav in wav_list:
                    fscp.write(wav[:-4]+' '+os.path.join(path_wav, wav)+'\n')

        path_scp = os.path.join(self.work_ivector, 'material', 'wav.scp')
        make_scp(self.path_wav, path_scp)
        
        p = subprocess.Popen('./run.sh', cwd = self.work_ivector, shell=True)
        p.wait()
        ori_ivector = os.path.join(self.work_ivector, 'ivector', 'ivector.ark')
        shutil.copyfile(ori_ivector, self.path_ivector)

    def extract_mfcc(self):
        self.append_path(self.work_zrst)
        self.mkdir_for_file(self.path_dnn_mfc)
        self.mkdir_for_dir(self.path_hmm_mfc)
        
        import util
        path_temp_mfc = os.path.join(self.work_zrst, 'temp')

        self.mkdir_for_dir(path_temp_mfc)
        util.make_feature(self.path_wav, path_temp_mfc)

        def write_feat(feat, feat_id, outfile):
            outfile.write(feat_id+'.wav\n')
            for i in range(feat.shape[0]):
                fline = '{:04d} {:04d} #' + ' {:f}'*feat.shape[1] +'\n'
                outfile.write(fline.format(i, i+1, *feat[i]))
            outfile.write('\n')
        
        with open(self.path_dnn_mfc, 'w') as outfile:
            for f in sorted(os.listdir(path_temp_mfc)):
                feat = util.read_feature(os.path.join(path_temp_mfc, f))
                write_feat(feat, f[:-4], outfile)
       
 
        for f in sorted(os.listdir(path_temp_mfc)):
            feat = util.read_feature(os.path.join(path_temp_mfc, f))
            util.write_feature(feat, os.path.join(self.path_hmm_mfc, f))
        self.mkdir_for_dir(path_temp_mfc)

    def extract_label(self, cluster_number):
        self.append_path(self.work_zrst)
        self.mkdir_for_file(self.path_init)

        import util
        path_matlab = os.path.join(self.work_zrst, 'matlab')
        init_word_file = self.path_init + '_word'

        cmd = 'matlab -nosplash -nodesktop -nojvm -r "clusterDetection_function {0} {1} {2}"'.format(self.path_wav, cluster_number, init_word_file)
        p = subprocess.Popen(cmd, cwd = path_matlab, shell=True)
        p.wait() 
        util.flat_dumpfile(init_word_file, self.path_init)
        
         
    def train_hmm(self, state_number):
        self.append_path(self.work_zrst)
        self.mkdir_for_file(self.path_hmm)
        import asr

        A = asr.ASR(corpus='UNUSED', 
            target=self.path_hmm, 
            nState=state_number, 
            nFeature=39, 
            features = self.lsdir_full(self.path_hmm_feature),
            dump = self.path_init, 
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

    def reinforce_label(self):
        self.append_path(self.work_zrst)
        self.mkdir_for_dir(os.path.join(self.work_mat, 'pattern'))

    def train_mat(self, nState_list, nHMM_list):
        for i in nState_list:
            for j in nHMM_list:
                pass

    def decode_mat():
        pass

    def train_dnn():
        pass
    
    def decode_dnn():
        pass
    



if __name__=='__main__':
    '''
    api = API()
    api.work_zrst = '/home/c2tao/ZRC_revision/mat/zrst/'
    api.path_wav = '/home/c2tao/timit_mini_corpus/'
    api.path_dnn_mfc = '/home/c2tao/ZRC_revision/matdnn/featdnn/timit_mini_corpus.mfc'
    api.path_hmm_mfc = '/home/c2tao/ZRC_revision/matdnn/feathmm/mfc/'
    api.extract_mfcc()

    api.work_zrst = api.work_zrst
    api.path_wav = api.path_wav
    api.path_init = '/home/c2tao/ZRC_revision/matdnn/init/timit_mini_corpus_10.txt'
    api.extract_label(10)
    
    api.work_zrst = api.work_zrst
    api.path_init = api.path_init
    api.path_hmm_feature = api.path_hmm_mfc
    api.path_hmm = '/home/c2tao/ZRC_revision/matdnn/token/mfc_10_3/'
    api.train_hmm(3)
    api.path_hmm = '/home/c2tao/ZRC_revision/matdnn/token/mfc_10_5/'
    api.train_hmm(5)
    '''

    wav_dir = '/home/c2tao/timit_mini_corpus/' 
    init_file = '/home/c2tao/ZRC_revision/init.txt'
    model_dir = '/home/c2tao/ZRC_revision/matdnn/token/mfc_10_3/'
    feature_dir = '/home/c2tao/ZRC_revision/matdnn/feathmm/mfc/'


    init = Init(init_file = init_file, wav_dir = wav_dir)
    init.extract(3)
    tok = Tokenizer(init_file = init_file, model_dir = model_dir, feature_dir = feature_dir)
    tok.train(3)

    '''
    api = Fobject()

    def test_fun(x):
        return x*x*x
    print run_parallel(test_fun, [1,2,3])
    api.run_parallel_example()
    '''
    ''' 
    api.work_zrst = '/home/c2tao/ZRC_revision/zrst/'
    api.work_ivector = '/home/c2tao/ZRC_revision/ivector/'
    api.path_wav = '/home/c2tao/ZRC_revision/surprise/wavs/'
    api.path_ivector = 'ivector_xit.ark'
    api.extract_ivector()

    api.path_wav = '/home/c2tao/ZRC_revision/eng/wavs/'
    api.path_ivector = 'ivector_eng.ark'
    api.extract_ivector()

    api.path_wav = '/home/c2tao/timit_train_corpus'
    api.path_ivector = 'ivector_timit_train_corpus.ark'
    api.extract_ivector()
    
    api.path_wav = '/home/c2tao/timit_test_corpus'
    api.path_ivector = 'ivector_timit_test_corpus.ark'
    api.extract_ivector()
    
    api.path_wav = '/home/c2tao/timit_train_corpus'
    api.path_dnn_mfc = 'timit_train_corpus.mfc'
    api.extract_mfcc() 
    
    api.path_wav = '/home/c2tao/timit_test_corpus'
    api.path_dnn_mfc = 'timit_test_corpus.mfc'
    api.extract_mfcc() 
    '''
