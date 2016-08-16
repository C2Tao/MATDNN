from matdnn import Fobj, Archive, Init, Tokenizer, Feature, NeuralNet
#from matdnn import MAT, MDNN
import os, sys
execute_list = sys.argv[1:]

def run(exe_word):
    return exe_word in execute_list


if __name__=='__main__':
    wav = '/home_local/c2tao/timit_mini_corpus/'
    root = '/home_local/c2tao/timit_mini_matdnn/'

    path = Fobj(path = root + 'path/',
        wav = wav,
        bnf = root + 'bnf{}_feat/',
        hmm = root + 'bnf{}_hmm/',
        dnn = root + 'bnf{}_dnn/',
        overwrite = True)

    hyper = Fobj(path = root + 'hyper/',
        num_mr = 2,#>=1
        num_bnf = 2,
        overwrite = True)
       
    param = Fobj(path = root + 'param/',
        cluster_list = [10, 20],
        state_list = [3, 5],
        hidden_list = [32, 32, 10],
        overwrite = True)

    A, M, N = {}, {}, {}

    # bnf 0
    A[0] = Archive(path = path.bnf.format(0),
        wav_dir = path.wav,
        overwrite = run('a0'))

    M[0] = MAT(path = path.hmm.format(0),
        archive = A[0],
        cluster_list = param.cluster_list, 
        state_list = param.state_list,
        overwrite = run( 'b0' ))

    for j in range(hyper.num_mr+1):
        M[0].train(mr = j, overwrite = run('b0'))


    # bnf >=1
    for i in range(hyper.num_bnf):
        N[i] = MDNN(path = path.dnn.format(i),
            archive = A[i],
            mat = M[i],
            hidden_list = param.hidden_list,
            overwrite = run( 'c'+str(i) ))

        N[i].train(mr = hyper.num_mr, overwrite = run( 'c'+str(i) )) 

        A[i+1] = Archive(path = path.bnf.format(i+1),
            neuralnet = N[i].neuralnet_list[hyper.num_mr],
            feature = N[i].feature_list[hyper.num_mr],
            overwrite = run( 'a'+str(i+1) ))

        M[i+1] = MAT(path = path.hmm.format(i+1),
            archive = A[i+1],
            mat = M[i],
            cluster_list = param.cluster_list, 
            state_list = param.state_list,
            overwrite = run( 'b'+str(i+1) ))

        for j in range(hyper.num_mr+1):
            M[i+1].train(mr = j, overwrite = run('b'+str(i+1)))
