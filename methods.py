from Bio.PDB.PDBParser import PDBParser
import matplotlib.pyplot as plt
import pyvista as pv
import numpy as np

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
                    for atom in residue.get_list():
                        coord.append(atom.get_coord())
                        id.append(i)
                        i+=1 
    
    return [id, np.array(coord)]


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

def plot_3d(structure_id, structure):
    plot = pv.Plotter(off_screen=True)
    plot.add_mesh(structure[1], color = 'k', render_points_as_spheres=True, point_size =5, show_scalar_bar=True)
    plot.background_color = 'w'
    path = plot.generate_orbital_path(n_points=100, shift = 100, factor=3.0)
    plot.open_gif("RESULTS//" + structure_id + ".gif")
    plot.orbit_on_path(path, write_frames=True)
    plt.close()
    return