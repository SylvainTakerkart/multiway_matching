import os.path as op
import os
import numpy as np
import glob
import pickle
import networkx as nx

from pitskernel.pitskernel_networkx import sxdnewkernel



subjects_list = ['OAS1_0006', 'OAS1_0009', 'OAS1_0025', 'OAS1_0049', 'OAS1_0051', 'OAS1_0054', 'OAS1_0055',
                 'OAS1_0057', 'OAS1_0059', 'OAS1_0061', 'OAS1_0077', 'OAS1_0079', 'OAS1_0080', 'OAS1_0087',
                 'OAS1_0104', 'OAS1_0125', 'OAS1_0136', 'OAS1_0147', 'OAS1_0150', 'OAS1_0151', 'OAS1_0152',
                 'OAS1_0156', 'OAS1_0162', 'OAS1_0191', 'OAS1_0192', 'OAS1_0193', 'OAS1_0202', 'OAS1_0209',
                 'OAS1_0218', 'OAS1_0224', 'OAS1_0227', 'OAS1_0231', 'OAS1_0236', 'OAS1_0239', 'OAS1_0246',
                 'OAS1_0249', 'OAS1_0253', 'OAS1_0258', 'OAS1_0294', 'OAS1_0295', 'OAS1_0296', 'OAS1_0310',
                 'OAS1_0311', 'OAS1_0313', 'OAS1_0325', 'OAS1_0348', 'OAS1_0379', 'OAS1_0386', 'OAS1_0387',
                 'OAS1_0392', 'OAS1_0394', 'OAS1_0395', 'OAS1_0397', 'OAS1_0406', 'OAS1_0408', 'OAS1_0410',
                 'OAS1_0413', 'OAS1_0415', 'OAS1_0416', 'OAS1_0417', 'OAS1_0419', 'OAS1_0420', 'OAS1_0421',
                 'OAS1_0431', 'OAS1_0437', 'OAS1_0442', 'OAS1_0448', 'OAS1_0004', 'OAS1_0005', 'OAS1_0007',
                 'OAS1_0012', 'OAS1_0017', 'OAS1_0029', 'OAS1_0037', 'OAS1_0043', 'OAS1_0045', 'OAS1_0069',
                 'OAS1_0090', 'OAS1_0092', 'OAS1_0095', 'OAS1_0097', 'OAS1_0101', 'OAS1_0102', 'OAS1_0105',
                 'OAS1_0107', 'OAS1_0108', 'OAS1_0111', 'OAS1_0117', 'OAS1_0119', 'OAS1_0121', 'OAS1_0126',
                 'OAS1_0127', 'OAS1_0131', 'OAS1_0132', 'OAS1_0141', 'OAS1_0144', 'OAS1_0145', 'OAS1_0148',
                 'OAS1_0153', 'OAS1_0174', 'OAS1_0189', 'OAS1_0211', 'OAS1_0214', 'OAS1_0232', 'OAS1_0250',
                 'OAS1_0261', 'OAS1_0264', 'OAS1_0277', 'OAS1_0281', 'OAS1_0285', 'OAS1_0302', 'OAS1_0314',
                 'OAS1_0318', 'OAS1_0319', 'OAS1_0321', 'OAS1_0328', 'OAS1_0333', 'OAS1_0340', 'OAS1_0344',
                 'OAS1_0346', 'OAS1_0350', 'OAS1_0359', 'OAS1_0361', 'OAS1_0368', 'OAS1_0370', 'OAS1_0376',
                 'OAS1_0377', 'OAS1_0385', 'OAS1_0396', 'OAS1_0403', 'OAS1_0409', 'OAS1_0435', 'OAS1_0439',
                 'OAS1_0450']

# parameters
hem = 'lh'
graph_type = 'radius'
graph_param = 60

# define directories
db_name = 'OASIS'
root_analysis_dir = '/hpc/meca/users/takerkart/multiway_graph_matching/'+ db_name
experiment = 'oasis_pits02'
analysis_dir = op.join(root_analysis_dir, experiment)
fullgraphs_dir = op.join(analysis_dir, 'full_hemisphere_pitgraphs')
gram_dir = op.join(analysis_dir, 'withinsubj_gram_matrices','{}_{:d}'.format(graph_type,graph_param))
try:
    os.makedirs(gram_dir)
    print('Creating new directory: {}'.format(gram_dir))
except:
    print('Output directory is {}'.format(gram_dir))


for subject in subjects_list:

    localgraphs_dir = op.join(analysis_dir, 'local_graphs', '{}_{:d}'.format(graph_type, graph_param), subject)
    paths_list = sorted(glob.glob(op.join(localgraphs_dir,'localgraph_{}_pit*.gpk.gz'.format(hem))))

    graphs_list = [nx.read_gpickle(path) for path in paths_list]
    n_pits = len(graphs_list)

    g1 = graphs_list[2]
    g2 = graphs_list[28]

    # note that we use x_sigma=50 here because the coordinates of the freesurfer sphere are between 0 and 100
    kernel = sxdnewkernel(x_sigma = 50, d_sigma = 0.5, subkernels=True)

    K = np.zeros([n_pits, n_pits,7])
    for i in range(n_pits):
        for j in range(i,n_pits):
            K[i,j,:] = kernel.evaluate(graphs_list[i], graphs_list[j])
            K[j,i,:] = np.copy(K[i,j,:])

    gram_path = op.join(gram_dir,'K_{}_{}.pck'.format(subject,hem))
    f = open(gram_path,'wb')
    pickle.dump(K,f)
    f.close()

'''
def main():
    args = sys.argv[1:]
    
    if len(args) < 2:
	    print('Wrong number of arguments, run it as: %run 02_compute_local_graphs_pitcentered.py lh radius 50')
	    sys.exit(2)
    else:
        hem = args[0]
        graph_type = args[1]
        graph_param = int(args[2])

    for subject in subjects_list:
        compute_localgraphs(subject,hem,graph_type,graph_param)


if __name__ == "__main__":
    main()
'''

