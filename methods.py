from Bio.PDB.PDBParser import PDBParser
import matplotlib.pyplot as plt
import pyvista as pv
import numpy as np
import gudhi

def PDB_prot_reader(name, file_name):

    parser = PDBParser(PERMISSIVE=1)
    structure = parser.get_structure(name, file_name)
    id = []
    coord = []
    i = 0

    for model in structure.get_list():
        for chain in model.get_list():
            for residue in chain.get_list():
                if residue.has_id("C"):
                    coord.append(list(residue["C"].get_coord()))
                    id.append(i)
                    i+=1 
    
    return [np.array(id), np.array(coord)]


def PDB_mol_reader(name, file_name):
    parser = PDBParser(PERMISSIVE=1)
    structure = parser.get_structure(name, file_name)
    id = []
    coord = []
    i = 0

    atoms = structure.get_atoms()

    for atom in atoms:
        coord.append(atom.get_coord())
        id.append(i)
        i+=1

    return [id, np.array(coord)]

def plot_3d(points_id, points):
    plot = pv.Plotter(off_screen=True)
    plot.add_mesh(points, color = 'k', render_points_as_spheres=True, point_size =5, show_scalar_bar=True)
    plot.background_color = 'w'
    path = plot.generate_orbital_path(n_points=100, shift = 100, factor=3.0)
    plot.open_gif("RESULTS//" + points_id + ".gif")
    plot.orbit_on_path(path, write_frames=True)
    plt.close()
    return

def generate_rips(points, epsilon):
    rips_complex = gudhi.RipsComplex(points = points, max_edge_length = epsilon)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    print('Rips complex is of dimension '  + \
        repr(simplex_tree.dimension()) + ' - ' + \
        repr(simplex_tree.num_simplices()) + ' simplices - ' + \
        repr(simplex_tree.num_vertices()) + ' vertices.') 

    return simplex_tree

def plot_rips(points, rips, edges, triangles, structure_id, epsilon, local):
    plot = pv.Plotter(off_screen=True)
    plot.add_mesh(points, color = 'k', render_points_as_spheres=True, point_size=2, show_scalar_bar=True)
    boring_cmap = plt.cm.get_cmap("viridis", 5)
    for idx, edge in enumerate(edges):
        a = np.array([points[edge[0]][0], points[edge[0]][1], points[edge[0]][2]])
        b = np.array([points[edge[1]][0], points[edge[1]][1], points[edge[1]][2]])
        line = pv.Line(a, b)
        plot.add_mesh(line, scalars = local[idx], cmap=boring_cmap)

    for idx, triangle in enumerate(triangles):
        a = np.array([points[triangle[0]][0], points[triangle[0]][1], points[triangle[0]][2]])
        b = np.array([points[triangle[1]][0], points[triangle[1]][1], points[triangle[1]][2]])
        c = np.array([points[triangle[2]][0], points[triangle[2]][1], points[triangle[2]][2]])
        tri = pv.Triangle([a, b, c])

        for idx2, edge in enumerate(edges):
            if triangle[0] == edge[0] and triangle[1] == edge[1]:
                index0 = idx2
            elif triangle[1] == edge[0] and triangle[2] == edge[1]:
                index1 = idx2
            elif triangle[0] == edge[0] and triangle[2] == edge[1]:
                index2 = idx2

        mean_color = (local[index0] + local[index1] + local[index2])/3
        plot.add_mesh(tri, scalars = mean_color, cmap=boring_cmap, style='surface', show_edges=False, opacity = 0.1, line_width=1)

    plot.background_color = 'w'
    path = plot.generate_orbital_path(n_points=100, shift = 100, factor=3.0)
    plot.open_gif("RESULTS//Rips_" + "_" + structure_id + "_" + str(epsilon) + ".gif")
    plot.orbit_on_path(path, write_frames=True)

    return

def inconsistency(structure, rips, edges, triangles):
    no_vertex = len(structure[0])
    no_edge = len(edges)
    no_triangle = len(triangles)

    #Computation of coboundary map
    d0 = np.zeros((no_edge,no_vertex))
    d1 = np.zeros((no_triangle,no_edge))
    Y_vec = np.zeros(no_edge)
    Y_g = np.zeros(no_edge)
    Y_curv = np.zeros(no_edge)

    for idx, edge in enumerate(edges):
        dist = np.linalg.norm(structure[1][edge[0]] - structure[1][edge[1]])
        Y_vec[idx] = dist
        d0[idx, edge[0]] = -1
        d0[idx, edge[1]] = 1

    for i in range(no_triangle):
        for idx, edge in enumerate(edges):
            if triangles[i][0] == edge[0] and triangles[i][1] == edge[1]:
                index0 = idx
            elif triangles[i][1] == edge[0] and triangles[i][2] == edge[1]:
                index1 = idx
            elif triangles[i][0] == edge[0] and triangles[i][2] == edge[1]:
                index2 = idx

        d1[i, index0] = 1
        d1[i, index1] = 1
        d1[i, index2] = -1

    L0 = np.dot(np.transpose(d0), d0)
    div = np.dot(np.transpose(-d0), Y_vec)
    s = np.linalg.lstsq(L0, -div, 1e-10)
    Y_g = np.dot(d0, s[0])
    Y_curv = Y_vec- Y_g

    local_inconsistency = np.abs(Y_curv/Y_vec)
    inconsistency1 = np.sum(np.abs(Y_curv/Y_vec))
    inconsistency2 = np.sqrt(np.sum(np.power((Y_curv/Y_vec), 2)))

    return local_inconsistency, inconsistency1, inconsistency2