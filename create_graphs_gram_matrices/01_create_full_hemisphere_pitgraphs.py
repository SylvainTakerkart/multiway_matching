# this is a copy of the file create_full_hemisphere_pitsgraph.py, writte by Yaroslvav Mavliutov in July 2019
# the github repository where this file can be found is located at https://github.com/gauzias/pytorch_graph_pits
# note that we should remove the dependency on the graph class, which is a leftover of the old sgbm_lib package
# used in the original SGBM code

from nibabel import load
import os
import os.path as op
import sys
import numpy as np
from sklearn.neighbors import kneighbors_graph
import networkx as nx

#from preprocessing.graph import pitsgraph
class pitsgraph:
    def __init__(self, A, X, D, Y, S=None, T=None, other_coords=None):
        # adjacancy matrix
        self.A = A

        # coordinates of each pit
        self.X = X

        # attributes of the pits: depth
        self.D = D

        # labels of the pits: 0 or 1
        self.Y = Y

        # attributes of the basins: area (surface)
        self.S = S

        # attributes of the basins: mean thickness
        self.T = T

        # other coordinates just in case (here, the spherical coordinates, rho & theta)
        self.other_coords = other_coords


db_name = 'OASIS'
fs_db_path = '/hpc/meca/data/OASIS/FS_OASIS'
input_data_dir = '/hpc/meca/data/OASIS/SulcalPits/OASIS_database/neuroImage_paper/OASIS_pits/subjects'

root_analysis_dir = '/hpc/meca/users/auzias/pits_graph_clustering/'+ db_name
root_analysis_dir = '/hpc/meca/users/takerkart/multiway_graph_matching/'+ db_name
experiment = 'oasis_pits02'

analysis_dir = op.join(root_analysis_dir, experiment)

param_string = 'dpfMap/alpha_0.03/an0_dn20_r1.5/alpha0.03_an0_dn20_r1.5'

def compute_fullgraphs(hem):

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

    if hem=='lh':
        BV_hem = 'L'
    else:
        BV_hem = 'R'

    fullgraphs_dir = op.join(analysis_dir, 'full_hemisphere_pitgraphs')
    try:
        os.makedirs(fullgraphs_dir)
        print('Creating new directory: %s' % fullgraphs_dir)
    except:
        print('Output directory is %s' % fullgraphs_dir)

    for subject in subjects_list:
        #try:
        pitgraphs_path = op.join(fullgraphs_dir, 'full_%s_%s_pitgraph.gpickle' % (hem, subject))
        print(pitgraphs_path)
        exists = op.isfile(pitgraphs_path)
        if exists:
            continue
        else:
            # get basins texture (-1 in poles; everything 0 or above is a real basin with one pit)
            basins_path = op.join(input_data_dir, subject, '%s_%s_area50FilteredTexture.gii' % (param_string, BV_hem))
            basins_tex_gii = load(basins_path)
            basins_tex = basins_tex_gii.darrays[0].data

            # get pits texture (0 everywhere except single vertex with one where the pits are)
            pits_path = op.join(input_data_dir, subject, '%s_%s_area50FilteredTexturePits.gii' % (param_string, BV_hem))
            pits_tex_gii = load(pits_path)
            pits_tex = pits_tex_gii.darrays[0].data
            pits_inds = np.where(pits_tex == 1)[0]

            # read depth of the pits
            depth_path = op.join(input_data_dir, subject,
                                 'dpfMap', '%s_dpf_0.03.gii' % (BV_hem))
            depth_gii = load(depth_path)
            depth_tex = depth_gii.darrays[0].data

            # get area of basins
            area_tex = basins_tex_gii.darrays[2].data
            basins_area = area_tex[pits_inds]
            # convert vector to matrix
            basins_area = np.atleast_2d(basins_area).T

            # read triangulated spherical mesh and get coordinates of all vertices
            mesh_path = os.path.join(fs_db_path, subject, 'surf', '%s.sphere.reg.gii' % hem)
            mesh_gii = load(mesh_path)
            mesh_coords = mesh_gii.darrays[0].data

            # read thickness
            thickness_path = op.join(fs_db_path, subject, 'surf', '%s.thickness.gii' % hem)
            thickness_gii = load(thickness_path)
            thickness_tex = thickness_gii.darrays[0].data

            # read labels of nodes
            labels_path = op.join(fs_db_path, 'ML_deep_shallow', '%s_%s_deep_shallow_from_destrieux.gii' % (subject, hem))
            labels_tex_gii = load(labels_path)
            labels_tex = labels_tex_gii.darrays[0].data

            g = graphs_construction(pits_inds, basins_tex, basins_area, depth_tex,
                                    mesh_coords, thickness_tex, pits_path, basins_path, labels_tex)

            # write graph
            G = transform_to_networks(g)
            nx.write_gpickle(G, pitgraphs_path)
        #except:
        #    print('cannot load data for subject '+subject)


def graphs_construction(pits_inds, basins_tex, basins_area, depth_tex, mesh_coords,
                        thickness_tex, pits_path, basins_path, labels_tex):

    # convert cartesian coordinates into spherical coordinates
    ro = np.sqrt(np.sum(mesh_coords * mesh_coords, 1))
    phi = np.arctan(mesh_coords[:, 1] / mesh_coords[:, 0])
    theta = np.arccos(mesh_coords[:, 2] / ro)
    spherical_coords = np.vstack([phi, theta]).T

    # compute connectivity of the mesh
    mesh_connectivity = kneighbors_graph(mesh_coords, n_neighbors=6, include_self=False)

    # get coordinates of the pits on the sphere and in 3d space
    pits_spherecoords = spherical_coords[pits_inds, :]
    pits_3dcoords = mesh_coords[pits_inds, :]
    n_pits = len(pits_inds)

    # get basins labels for each pit, and put them in the same order as the pits!
    basins_labels = []
    for pit_ind in range(n_pits):
        basins_labels.append(basins_tex[pits_inds[pit_ind]])

    # sanity check on basins and pits textures (one pit per basin; same number of pits and basins etc.)
    basins_tmp1_labels = np.unique(basins_tex)

    # get rid of the nodes which have negative labels (-1 labels for the poles)
    basins_tmp1_labels = basins_tmp1_labels[np.where(basins_tmp1_labels >= 0)[0]]
    basins_tmp2_labels = basins_labels[:]
    basins_tmp2_labels.sort()
    if ((len(basins_tmp1_labels) != len(basins_tmp2_labels)) or np.max(
            np.abs(np.array(basins_tmp1_labels) - np.array(basins_tmp2_labels)))):
        print("Error: there's something weird with the pits and/or basins textures: %s and %s " % (pits_path, basins_path))
    # build connectivity matrix of the basin-based region adjacancy graph
    basins_submask = []
    basins_size = []
    n_basins = np.size(basins_labels)
    for basin_ind, basin_label in enumerate(basins_labels):
        basins_submask.append(np.array(np.nonzero(basins_tex == basin_label))[0])
        basins_size.append(len(basins_submask[-1]))

    adjacency = np.zeros([n_basins, n_basins])
    for i in range(n_basins):
        for j in range(i):
            adjacency[i, j] = mesh_connectivity[basins_submask[i], :][:, basins_submask[j]].sum()
            adjacency[j, i] = adjacency[i, j]
        adjacency[adjacency != 0] = 1

    # add ones on the diagonal (every node is connected to itself)
    np.fill_diagonal(adjacency, 1.)

    pits_depth = depth_tex[pits_inds]

    # convert vector to matrix
    pits_depth = np.atleast_2d(pits_depth).T

    pits_label = labels_tex[pits_inds]
    pits_label = np.atleast_2d(pits_label).T

    # compute mean thickness in basin
    basins_thickness = np.zeros(n_pits)
    for basin_ind, basin_label in enumerate(basins_labels):
        basin_inds = np.where(basins_tex == basin_label)[0]
        basins_thickness[basin_ind] = np.mean(thickness_tex[basin_inds])

    # convert vector to matrix
    basins_thickness = np.atleast_2d(basins_thickness).T

    g = pitsgraph(adjacency, pits_3dcoords, pits_depth, pits_label, basins_area, basins_thickness, pits_spherecoords)
    return g


def transform_to_networks(value):
    # create networkx object
    G = nx.from_numpy_matrix(value.A)
    nx.set_node_attributes(G, array_to_dict(value.D), 'depth')
    nx.set_node_attributes(G, array_to_dict(value.Y), 'labels')
    nx.set_node_attributes(G, array_to_dict(value.X), 'coord')
    nx.set_node_attributes(G, array_to_dict(value.S), 'area')
    nx.set_node_attributes(G, array_to_dict(value.T), 'thickness')
    nx.set_node_attributes(G, array_to_dict(value.other_coords), 'sph_coord')
    return G

def array_to_dict(array):
    D = {}
    for i in range(array.shape[0]):
        try:
            D[i] = [array[i][0], array[i][1], array[i][2]]
        except:
            D[i] = array[i][0]
    return D


def main():
    args = sys.argv[1:]

    # for both hemispheres, just run it as 'python 01_create_full_hemisphere_pitgraphs.py'
    # for just one hemisphere, run it as 'python 01_create_full_hemisphere_pitgraphs.py lh'

    if len(args) < 1:
        hemispheres_list = ['lh', 'rh']
    else:
        hem = args[0]
        hemispheres_list = [hem]

    for hem in hemispheres_list:
        compute_fullgraphs(hem)

if __name__ == "__main__":
    main()