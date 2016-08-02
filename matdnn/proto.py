def remove_indent(strx, level):
    return '\n'.join(map(lambda x: x[4*level:], strx.split('\n')))

def proto_solver(train_file, model_dir):
    strx = '''\
    net: "{train_file}"
    # test_iter specifies how many forward passes the test should carry out.
    # In the case of MNIST, we have test batch size 100 and 100 test iterations,
    # covering the full 10,000 testing images.
    test_iter: 500
    # Carry out testing every 500 training iterations.
    test_interval: 1000
    # The base learning rate, momentum and the weight decay of the network.
    base_lr: 0.1126
    momentum: 0.5
    weight_decay: 0.0000
    # The learning rate policy
    lr_policy: "inv"
    gamma: 0.0001
    power: 0.75
    # Display every 500 iterations
    display: 500
    # The maximum number of iterations
    max_iter: 1000
    # snapshot intermediate results
    snapshot: 500
    snapshot_prefix: "{model_dir}/"
    # solver mode: CPU or GPU
    solver_mode: GPU'''
    stry = strx.replace('{train_file}', train_file).replace('{model_dir}', model_dir)
    return remove_indent(stry, 1)

def proto_deploy(model_name, input_dim, hidden_list):
    strx = '''\
    name: "{model_name}_deploy"
    input: "data"
    input_dim: 1024
    input_dim: {input_dim}
    input_dim: 1
    input_dim: 1'''

    stry = '''\
    layers {
      name: "ip{hidden_id}"
      type: INNER_PRODUCT
      bottom: "{ip_hidden_id-1}"
      top: "ip{hidden_id}"
      inner_product_param {
        num_output: {hidden_size}
      }
      param: "ip{hidden_id}_w"
      param: "ip{hidden_id}_b"
    }'''

    strz = '''\
    layers {
      name: "sig{hidden_id}"
      type: SIGMOID
      bottom: "ip{hidden_id}"
      top: "ip{hidden_id}"
    }'''
    def mkstryz(hidden_id, hidden_size): 
        stra = stry.replace('{hidden_id}', str(hidden_id))
        if hidden_id!=1:
            strb = stra.replace('{ip_hidden_id-1}', 'ip'+str(hidden_id-1))
        else:
            strb = stra.replace('{ip_hidden_id-1}', 'data')
        strc = strb.replace('{hidden_size}', str(hidden_size))
        strd = strc.replace('{input_dim}', str(input_dim))
        if hidden_id != len(hidden_list):
            stre = strz.replace('{hidden_id}', str(hidden_id))
        else:
            stre = ''
        return '\n'.join([strd, stre])
    stra = strx.replace('{model_name}', model_name).replace('{input_dim}', str(input_dim))

    str_list = []
    for hidden_id, hidden_size in enumerate(hidden_list):
        str_list.append(mkstryz(hidden_id+1, hidden_size))
    strb = '\n'.join(str_list)
    return remove_indent(stra + '\n' + strb, 1)
        

def proto_train(model_name, train_list_file, hidden_list, output_list): 
    def proto_train_0(model_name): 
        str0 = '''\
        name: "{x}_train"
        layers {
          name: "data"
          type: HDF5_DATA
          top: "data"
          top: "label"
          hdf5_data_param {
            source: "{train_list_file}"
            batch_size: 1024
          }
        }
        '''
        return str0.replace('{x}', model_name).replace('{train_list_file}', train_list_file)

    def proto_train_1(output_len):
        stra = '''
        layers {
            name: "slice_label"
            type: SLICE
            bottom: "label"\n'''
        str1 = '''\
            top: "label{}"'''
        strb = '''
            slice_param {
                slice_dim: 1\n'''
        str2 = '''\
                slice_point: {}'''
        strc = '''
            }
        }'''
        str3 = '\n'.join([str1.format(i) for i in range(output_len)])
        str4 = '\n'.join([str2.format(i) for i in range(1, output_len)])
        return stra+str3+strb+str4+strc

    def proto_train_2(hidden_list):
        str2 = '''\
        layers {
          name: "ip{m}"
          type: INNER_PRODUCT
          bottom: "{ipm-1}"
          top: "ip{m}"
          blobs_lr: 1
          blobs_lr: 2
          inner_product_param {
            num_output: {h}
            weight_filler {
              type: "xavier"
            }
            bias_filler {
              type: "constant"
            }
          }
          param: "ip{m}_w"
          param: "ip{m}_b"
        }
        layers {
          name: "sig{m}"
          type: SIGMOID
          bottom: "ip{m}"
          top: "ip{m}"
        }'''
        def mkstr2(m, h): 
            stra = str2.replace('{m}', str(m))
            if m!=1:
                strb = stra.replace('{ipm-1}', 'ip'+str(m-1))
            else:
                strb = stra.replace('{ipm-1}', 'data')
            return strb.replace('{h}', str(h))
        str_list = []
        for m, h in enumerate(hidden_list):
            str_list.append(mkstr2(m+1, h))
        return '\n'.join(str_list)
        
    def proto_train_3(hidden_len, output_list):
        str3='''\
        layers {
          name: "output{x}"
          type: INNER_PRODUCT
          bottom: "ip{z}"
          top: "output{x}"
          blobs_lr: 1
          blobs_lr: 2
          inner_product_param {
            num_output: {y}
            weight_filler {
              type: "xavier"
            }
            bias_filler {
              type: "constant"
            }
          }
        }
        layers {
            name: "loss{x}"
            type: SOFTMAX_LOSS
            bottom: "output{x}"
            bottom: "label{x}"
            top: "loss{x}"
        }
        layers{
            name: "acc{x}"
            type: ACCURACY
            bottom: "output{x}"
            bottom: "label{x}"
            top: "acc{x}"
            include: { phase: TEST }
        }'''
        def mkstr3(i, cardinality): 
            stra = str3.replace('{z}', str(hidden_len))
            strb = stra.replace('{x}', str(i))
            return strb.replace('{y}', str(cardinality))
        str_list = []
        for i, cardinality in enumerate(output_list):
            str_list.append(mkstr3(i, cardinality))
        return '\n'.join(str_list)
    p0 = proto_train_0(model_name)
    p1 = proto_train_1(len(output_list))
    p2 = proto_train_2(hidden_list)
    p3 = proto_train_3(len(hidden_list), output_list)
    return remove_indent('\n'.join([p0, p1, p2, p3]), 2)

if __name__ == '__main__':
    model_name = 'timit'
    hidden_list = [256, 256, 39]
    output_list = [10, 10, 20, 20]
    input_dim = 751

    print '========================'
    print proto_solver(model_name)
    print '========================'
    print proto_deploy(model_name, input_dim, hidden_list)
    print '========================'
    print proto_train(model_name, hidden_list, output_list)
    print '========================'

    '''
    print '========================'
    print proto_train_0(model_name)
    print '========================'
    print proto_train_0(model_name)
    print '========================'
    print proto_train_1(len(output_list))
    print '========================'
    print proto_train_2(hidden_list)
    print '========================'
    print proto_train_3(len(hidden_list), output_list)
    print '========================'
    '''
