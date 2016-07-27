import numpy as np
import subprocess
import os
import time
import sys
import shutil

class Fobject(object):
    def __init__(self):
        pass

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
    

class MAT(Fobject):
    def __init__(self, work_mat, path_mat):
        self.path_mat = path_mat
        self.work_mat = work_mat



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
        path_init_word = self.path_init + '_word'

        cmd = 'matlab -nosplash -nodesktop -nojvm -r "clusterDetection_function {0} {1} {2}"'.format(self.path_wav, cluster_number, path_init_word)
        p = subprocess.Popen(cmd, cwd = path_matlab, shell=True)
        p.wait() 
        util.flat_dumpfile(path_init_word, self.path_init)
        
         
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


    api.work_mat = ''


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
