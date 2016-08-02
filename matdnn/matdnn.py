import numpy as np
import subprocess
import os
import time
import sys
import shutil
import proto
import cPickle

# kaldi, htk, srilm, caffe, mallet, matlab

# matdnn path
default_zrc_dir = '/home/c2tao/matdnn/'

# ivector path
default_ivector_dir = os.path.join(default_zrc_dir, 'ivector')

# mat path
default_mat_dir = os.path.join(default_zrc_dir, 'mat')
default_temp_dir = os.path.join(default_zrc_dir, 'temp')
default_zrst_dir = os.path.join(default_mat_dir, 'zrst')
default_matlab_dir = os.path.join(default_zrst_dir, 'matlab')
sys.path.insert(1, default_zrst_dir)

# mdnn path
import proto
default_mdnn_dir = os.path.join(default_zrc_dir, 'mdnn')
default_proto_dir = os.path.join(default_mdnn_dir, 'proto')

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
        if 'pkl' not in d:
            args.append(map(int, d.split('.')[0].split('_')))
    params = zip(*args)
    return map(lambda p: sorted(list(set(p))), params)

def extract_mfcc(input_wav_dir, output_feature_file, output_htk_feature_dir):
    # secure destination
    mkdir_for_file(output_feature_file)
    mkdir_for_dir(output_htk_feature_dir)
    
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
        util.write_feature(feat, os.path.join(output_htk_feature_dir, f))

    # cleanup, remove large files
    mkdir_for_dir(default_temp_dir)

def extract_ivector(input_wav_dir, output_ivector_file):
    # secure destination
    mkdir_for_file(output_ivector_file)

    # setup required files
    #mkdir_for_dir(os.path.join(default_ivector_dir, 'ivector'))
    #mkdir_for_dir(os.path.join(default_ivector_dir, 'ubm'))
    #mkdir_for_dir(os.path.join(default_ivector_dir, 'log'))
    #mkdir_for_dir(os.path.join(default_ivector_dir, 'feat'))

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
    subprocess.Popen('./run.sh', cwd = default_ivector_dir, shell=True).wait()

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
    subprocess.Popen(cmd, cwd = default_matlab_dir, shell=True).wait() 
    util.flat_dumpfile(init_word_file, output_init_file)
    
    # cleanup, remove large files
    subprocess.Popen('rm *temp*', cwd = default_matlab_dir, shell=True).wait() 
    

def train_tokenizer(input_init_file, input_htk_feature_dir, state_number, output_model_dir, pad_silence=False):
    # secure destination
    mkdir_for_file(output_model_dir)
    
    # train tokenizer
    import asr
    A = asr.ASR(corpus='UNUSED', 
        target=output_model_dir, 
        nState=state_number, 
        nFeature=39, 
        features = lsdir_full(input_htk_feature_dir),
        dump = input_init_file, 
        pad_silence=pad_silence)

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

        
def reinforce_label(input_tokenizer_dir, output_rein_dir, cluster_list = None, state_list = None):
    # secure output dir
    mkdir_for_file(os.path.join(output_rein_dir, 'dummy.txt'))
    
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
    cmd = './MR_commandline.sh "{}" "{}"'.format(arg1, arg2)
    p = subprocess.Popen(cmd, cwd = default_mat_dir, shell=True).wait()

    # copy files to output dir
    subprocess.Popen('mv * {}'.format(output_rein_dir), cwd = os.path.join(default_mat_dir, 'init'), shell=True).wait()
     
    # cleanup large files
    mkdir_for_dir(os.path.join(default_mat_dir, 'pattern'))
    mkdir_for_dir(os.path.join(default_mat_dir, 'init'))
    mkdir_for_dir(os.path.join(default_mat_dir, 'exp'))




class Fobj(object):
    def __init__(self, path, default = None, overwrite = False, **kwargs):
        '''
        if path does not exist, 
            it will be created
        if parameter pickle file already exists,
            kwargs has no effect and it will be discarded
        if parameter pickle file does not exist, or overwrite = True
            the entire directory will be deleted and recreated
        if overparameter pickle file does not exist, 
            the entire directory will be deleted and recreated
        calling build constructs the object with param
        '''
        self.path = path
        self.param_file = os.path.join(path, 'param.pkl')
        print 'initializing', self.__class__.__name__, 'object at', path
        
        if os.path.exists(self.param_file) and not overwrite:
            self.load()
            print '    WARNING: parameter file already exists, set "overwrite = True" to overwrite'
            #print self.param_file
        else:
            self.__dict__.update(kwargs)
            mkdir_for_dir(path)
            print '    updated object parameters'
            #print self

        if overwrite:
            self.build()
        self.save()

            
    def __repr__(self):
        return 'printing: '+self.param_file+'\n'+'\n'.join(map(lambda kv: ' '*4+str(kv[0])+': '+str(kv[1]),
            sorted(self.__dict__.items(), key = lambda x: x[0]) ))

    def load(self):
        with open(self.param_file, 'rb') as f:
            self.__dict__.update(cPickle.load(f))
    
    def save(self):
        with open(self.param_file, 'wb') as f:
            cPickle.dump(self.__dict__, f, protocol=cPickle.HIGHEST_PROTOCOL)

    def build(self):
        pass

class Test_Fobj(Fobj):
    def __init__(self, path, overwrite, a = None, b = None):
        super(self.__class__, self).__init__(path, overwrite, a = a, b = b )

class Test_Fobj2(Fobj):
    def __init__(self, **kwargs):
        super(self.__class__, self).__init__(**kwargs)

class Archive(Fobj):
    # requires (wav_dir)
    def build(self):
        assert hasattr(self, 'wav_dir'), 'ERROR: please specify "wav_dir"'
        self.ivector_file = os.path.join(self.path, 'ivector.ark')
        self.feature_file = os.path.join(self.path, 'mfc.feat')
        self.htk_feature_dir = os.path.join(self.path, 'htk_feature/')
        extract_ivector(self.wav_dir, self.ivector_file)
        extract_mfcc(self.wav_dir, self.feature_file, self.htk_feature_dir)

class Init(Fobj):
    # requires (archive and cluster_list) or (tokenizer)
    def build(self):
        if hasattr(self, 'archive'):
            assert hasattr(self, 'cluster_list'), 'ERROR: please specify "cluster_list"'
            for cluster_number in self.cluster_list:
                extract_init(self.archive.wav_dir, cluster_number, os.path.join(self.path, str(cluster_number) + '.txt'))
        elif hasattr(self, 'tokenizer'):
            self.cluster_list = self.tokenizer.cluster_list
            reinforce_label(self.tokenizer.path, self.path, self.tokenizer.cluster_list, self.tokenizer.state_list)

class Tokenizer(Fobj):
    # requires (init and state_list)
    def build(self):
        assert hasattr(self, 'init'), 'ERROR: please specify "init" object'
        assert hasattr(self, 'state_list'), 'ERROR: please specify "state_list"'
        if not hasattr(self, 'pad_silence'):
            self.pad_silence = False
        self.cluster_list = self.init.cluster_list
        run_parallel(train_tokenizer,[(
                    self.init.path+str(cluster_number)+'.txt', 
                    self.archive.htk_feature_dir, 
                    state_number, 
                    os.path.join(self.path, '{}_{}/'.format(cluster_number, state_number)),
                    self.pad_silence)
            for cluster_number in self.cluster_list for state_number in self.state_list])

class NeuralNet(Fobj):
    def build(self):
        assert hasattr(self, 'feature'), 'ERROR: please specify "feature" object'
        assert hasattr(self, 'hidden_list'), 'ERROR: please specify "hidden_list"'
        assert hasattr(self, 'output_list'), 'ERROR: please specify "output_list"'
        self.model_dir = os.path.join(self.path, 'model')
        mkdir_for_dir(self.model_dir)
        self.input_dim = self.feature.feature_dim

        self.build_proto()
        self.train()
        self.set_model_file()

    def train(self):
        cmd = "caffe train -solver {} ".format(self.solver_file)
        p = subprocess.Popen(cmd, cwd = default_mdnn_dir, shell=True).wait()
    
    def set_model_file(self):
        model_file = sorted(os.listdir(self.model_dir), key=lambda x: int(x.split('_')[2].split('.')[0]), reverse = True)[0]
        self.model_file = os.path.join(self.model_dir, model_file)

    def build_proto(self):
        self.model_name = 'mdnn'
        self.proto_path = os.path.join(self.path, 'proto')
        self.solver_file = os.path.join(self.proto_path, self.model_name+'_solver.prototxt')
        self.deploy_file = os.path.join(self.proto_path, self.model_name+'_deploy.prototxt')
        self.train_file = os.path.join(self.proto_path, self.model_name+'_train.prototxt')
        mkdir_for_file(self.solver_file)
        mkdir_for_file(self.deploy_file)
        mkdir_for_file(self.train_file)
        with open(self.solver_file, 'w') as f:
            f.write(proto.proto_solver(self.train_file, self.model_dir))
        with open(self.deploy_file, 'w') as f:
            f.write(proto.proto_deploy(self.model_name, self.feature.feature_dim, self.hidden_list))
        with open(self.train_file, 'w') as f:
            f.write(proto.proto_train(self.model_name, self.feature.train_list_file, self.hidden_list, self.output_list))

    def decode(self, feature_file, bnf_file):
        bnf_id = len(self.hidden_list)
        cmd = "python work/deploy.py --gpu --output-layer ip{} --input-dim {} --output-dim {} {} {} {} {}".format(
            bnf_id, 
            self.feature.feature_dim, 
            self.hidden_list[bnf_id-1], 
            feature_file, 
            bnf_file, 
            self.deploy_file, 
            self.model_file)
        p = subprocess.Popen(cmd, cwd = default_mdnn_dir, shell=True).wait()


class Feature(Fobj):
    def build(self):
        assert hasattr(self, 'archive'), 'ERROR: please specify "archive" object'
        assert hasattr(self, 'tokenizer'), 'ERROR: please specify "tokenizer" object'
        self.feature_file = os.path.join(self.path, 'full.feat') 
        self.dnn_feature_dir = os.path.join(self.path, 'dnn_feature')
        mkdir_for_dir(self.dnn_feature_dir)
        if not hasattr(self, 'left_context'): self.left_context = 4
        if not hasattr(self, 'right_context'): self.right_context = 4
        if not hasattr(self, 'input_dim'): self.input_dim = 39
        self.feature_dim = self.input_dim*(self.left_context+1+self.right_context) + 400
        self.splice_feat()
        self.make_token_list()
        self.feat_to_h5()
        self.make_train_list()
    
    def splice_feat(self):
        ivector_opt = '--ivector-ark ' + self.archive.ivector_file if os.path.isfile(self.archive.ivector_file) else ''
        cmd = 'python work/splice_feat.py --left-context {} --right-context {} --input-dim {} {} {} {}'.format(
            self.left_context, self.right_context, self.feature_dim, 
            ivector_opt, self.archive.feature_file, self.feature_file)
        p = subprocess.Popen(cmd, cwd = default_mdnn_dir, shell=True).wait()

    def make_token_list(self):
        self.token_list = []
        self.token_list_file = os.path.join(self.path, 'pattern.list')

        c_list, s_list = get_arg_from_dir(self.tokenizer.path)
        for c in c_list:
            for s in s_list:
                self.token_list.append(str(c)+'_'+str(s))
        
        with open(self.token_list_file, 'w') as f:
            for mlf in self.token_list:
                f.write(os.path.join(self.tokenizer.path, mlf, 'result', 'result.mlf')+ '\n')

    def make_train_list(self):
        self.train_list = sorted(os.listdir(self.dnn_feature_dir))
        self.train_list_file = os.path.join(self.path, 'train.list')
        
        with open(self.train_list_file, 'w') as f:
            for feat in self.train_list:
                f.write(os.path.join(self.dnn_feature_dir, feat)+ '\n')

    def feat_to_h5(self):
        silent_opt = '--silent' if self.tokenizer.pad_silence else ''
        cmd = 'python work/feat_pattern_to_h5.py --input-dim {} {} {} {} {}'.format(
            self.feature_dim, self.token_list_file, self.feature_file, self.dnn_feature_dir, silent_opt)
        p = subprocess.Popen(cmd, cwd = default_mdnn_dir, shell=True).wait()

    
def test_fobject():
    fobj = Fobj('tmp', x = 1, y = 2, z = 3)
    fobj = Fobj('tmp', x = 0, y = 0, z = 0)
    fobj = Fobj('tmp', x = 0, y = 0, z = 0, overwrite = True)
    fobj = Test_Fobj('tmp', overwrite = True, a = 1, b = 1)
    fobj = Test_Fobj2(path = 'tmp', overwrite = True, a = 1, b = 1)
    print fobj

def test_parallel_example_function(x, y):
    return x*y

def test_parallel():
    print run_parallel(test_parallel_example_function, [(1, 2), (3, 4), (5, 6)])

def test_mat():
    root = '/home/c2tao/matdnn_files/'
    wav_dir = '/home/c2tao/timit_mini_corpus/' 

    ivector_file = root + 'ivector.ark'
    feature_file = root + 'mfc.feat'
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

def test_api():
    #if __name__=='__main__':
    root = '/home/c2tao/matdnn_files/'
    path = Fobj(path = root + 'path/',
        wav = '/home/c2tao/timit_mini_corpus/',
        archive = root + 'feature/',
        init = root + 'init/',
        model0 = root + 'token/',
        reinforce = root + 'rein/',
        model1 = root + 'token2/',
        proto = root + 'proto/',
        overwrite = True)
   
    param = Fobj(path = root + 'param/',
        cluster_list = [10, 20],
        state_list = [3, 5],
        overwrite = True)

    A = Archive(path = path.archive,
        wav_dir = path.wav,
        overwrite = False)
    

if __name__=='__main__':
    #def test_api():
    #test_mat()
    #test_fobject()
    #test_parallel()
    root = '/home/c2tao/matdnn_files/'
    execute_list = [7]

    path = Fobj(path = root + 'path/',
        wav = '/home/c2tao/timit_mini_corpus/',
        archive = root + 'feature/',
        init = root + 'init/',
        model0 = root + 'token/',
        reinforce = root + 'rein/',
        model1 = root + 'token2/',
        dnn = root + 'neuralnet/',
        dnn_feature = root + 'dnn_feature/',
        overwrite = True)
   
    param = Fobj(path = root + 'param/',
        cluster_list = [10, 20],
        state_list = [3, 5],
        overwrite = True)

    A = Archive(path = path.archive,
        wav_dir = path.wav,
        overwrite = 0 in execute_list)
 
    I = Init(path = path.init, 
        archive = A, 
        cluster_list = param.cluster_list, 
        overwrite = 1 in execute_list)
    
    T = Tokenizer(path = path.model0, 
        archive = A, 
        init = I,
        state_list = param.state_list,
        overwrite = 2 in execute_list)

    R = Init(path = path.reinforce, 
        tokenizer = T,
        overwrite = 3 in execute_list)

    S = Tokenizer(path = path.model1, 
        archive = A, 
        init = R,
        state_list = param.state_list,
        overwrite = 4 in execute_list)

    F = Feature(path = path.dnn_feature,
        archive = A,
        tokenizer = S,
        overwrite = 5 in execute_list)
    
    P = NeuralNet(path = path.dnn,
        feature = F,
        model_name = 'mini',
        hidden_list = [256, 256, 39],
        output_list = [10, 10, 20, 20],
        overwrite = 6 in execute_list)

    P.set_model_file() 
    P.decode(P.feature.feature_file, 'tmp.feat')

    #G = Archive(path = path.dnn, 
    #    neuralnet = P, feature = F,    
    #    )    




