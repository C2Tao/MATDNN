from matdnn import Fobj, Archive, Init, Tokenizer, Feature, NeuralNet
#from matdnn import MAT, MDNN
import os, sys


if __name__=='__main__':
    wav = '/home_local/c2tao/timit_train_corpus_nosa/'
    root = '/home_local/c2tao/timit_train_matdnn_nosa/'
    import sys
    
    #execute_list = sys.argv[1:]
    execute_list  = ['00','01','02','03','04','05','06']

    def run(exe_word):
        return exe_word in execute_list
    


    param = Fobj(path = root + 'param/',
        cluster_list = [50, 100, 300, 500],
        state_list = [3, 5, 7, 9],
        hidden_list = [256, 256, 39],
        overwrite = True)

    #nBF = int(sys.argv[1][0])
    #nMR = int(sys.argv[1][1])
    nBF = 0
    nMR = 6
    A,I,T,F,N  = {},{},{},{},{}
    for nI in range(nBF+1):
        nJ = 0
        if nI==0:
            A[nI] = Archive(path = root + 'archive_bnf{}/'.format(nI),
                wav_dir = wav,
                overwrite = run('{}{}'.format(nI, nJ)))
        else:
            A[nI] = Archive(path = root + 'archive_bnf{}/'.format(nI),
                neuralnet = N[nI-1, nMR],
                feature = F[nI-1, nMR],
                overwrite = run('{}{}'.format(nI, nJ)))
     
        for nJ in range(nMR+1):
            if nI==0 and nJ==0:
                I[nI, nJ] = Init(path = root + 'init_bnf{}_mr{}/'.format(nI, nJ), 
                    archive = A[nI], 
                    cluster_list = param.cluster_list, 
                    overwrite = run('{}{}'.format(nI, nJ)))
            elif nI>1 and nJ==0:
                I[nI, nJ] = Init(path = root + 'init_bnf{}_mr{}/'.format(nI, nJ), 
                    tokenizer = T[nI-1, nMR-1],
                    overwrite = run('{}{}'.format(nI, nJ)))
            else:
                I[nI, nJ] = Init(path = root + 'init_bnf{}_mr{}/'.format(nI, nJ), 
                    tokenizer = T[nI, nJ-1],
                    overwrite = run('{}{}'.format(nI, nJ)))

            T[nI, nJ] = Tokenizer(path = root + 'tokenizer_bnf{}_mr{}/'.format(nI, nJ),
                archive = A[nI], 
                init = I[nI, nJ],
                state_list = param.state_list,
                overwrite = run('{}{}'.format(nI, nJ)))

            #if nJ != nMR: continue

            F[nI, nJ] = Feature(path = root + 'feature_bnf{}_mr{}/'.format(nI, nJ),
                archive = A[nI],
                tokenizer = T[nI, nJ],
                overwrite = run('{}{}'.format(nI, nJ)))
            
            N[nI, nJ] = NeuralNet(path = root +'neuralnet_bnf{}_mr{}/'.format(nI, nJ),
                feature = F[nI, nJ],
                hidden_list = param.hidden_list,
                overwrite = run('{}{}'.format(nI, nJ)))
    

