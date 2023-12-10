from functools import wraps
from pathlib import Path
from typing import Union, List
import os
import json
from multiprocessing import Pool
from pprint import pprint
import h5py
import numpy as np
import openfoamparser_mai as Ofpp
import pyvista




def save_json(filename, data, save_path) -> None:
    """Cохраняет json"""
    file_path = save_path / Path(filename)
    with open(file_path, 'w', encoding="utf8") as f:
        json.dump(data, f)


def save_json_in_chunks(filename, data, save_path, chunk_size=1000):
    full_path = os.path.join(save_path, filename)
    with open(full_path, 'w') as file:
        file.write('[')
        for i, item in enumerate(data):
            json_str = json.dumps(item)
            file.write(json_str)
            if i < len(data) - 1:
                file.write(',\n')
            if i % chunk_size == 0 and i != 0:
                file.flush()  # Flush data to disk periodically
        file.write(']')


# The wrapper function for multiprocessing
def save_json_in_chunks_wrapper(args):
    save_json_in_chunks(*args)


def json_streaming_writer(filepath, data_func, data_args):
    """Write JSON data to a file using a generator to minimize memory usage."""
    data_gen = data_func(*data_args)
    try:
        with open(filepath, 'w') as file:
            print(f"writing {filepath}")
            file.write('[')
            for i, item in enumerate(data_gen):
                if i != 0:  # Add a comma before all but the first item
                    file.write(',')
                json.dump(item, file)
            file.write(']')
        print(f"Finished writing {filepath}")
    except Exception as e:
        print(f"Failed to write {filepath}: {str(e)}") 


def create_nodes_gen(mesh_bin):
    """Generator for nodes."""
    for point in mesh_bin.points:
        yield {
            'X': point[0],
            'Y': point[1],
            'Z': point[2]
        }


def create_faces_gen(mesh_bin):
    """Generator for faces."""
    for face in mesh_bin.faces:
        yield list(face)


def create_elements_gen(mesh_bin, p, u, c):
    """Generator for elements."""
    for i, cell in enumerate(mesh_bin.cell_faces):
        yield {
            'Faces': cell,
            'Pressure': p[i],
            'Velocity': {
                'X': u[i][0],
                'Y': u[i][1],
                'Z': u[i][2]
            },
            'VelocityModule': np.linalg.norm(u[i]),
            'Position': {
                'X': c[i][0],
                'Y': c[i][1],
                'Z': c[i][2]
            }
        }


def create_surfaces_gen(surfaces):
    """Generator for surfaces."""
    for surface in surfaces:
        yield surface


def _face_center_position(points: list, mesh: Ofpp.FoamMesh) -> list:
    vertecis = [mesh.points[p] for p in points]
    vertecis = np.array(vertecis)
    return list(vertecis.mean(axis=0))




def process_computational_domain(solver_path: Union[str, os.PathLike, Path],
                                 save_path: Union[str, os.PathLike, Path],
                                 p: np.ndarray,
                                 u: np.ndarray,
                                 c: np.ndarray,
                                 surface_name: str) -> None:
    """Сохранение геометрии расчетной области в виде json файла с полями:
    'Nodes' - List[x: float, y: float, z:float], 
    'Faces' - List [List[int]], 
    'Elements' - List [Dict{Faces: List[int],
                            Pressure: float,
                            Velocity: List[float],
                            VelocityModule: float,
                            Position: List[float]}
                            ], 
    'Surfaces' - List[
                    Tuple[Surface_name: str, 
                    List[Dict{ParentElementID: int,
                              ParentFaceId: int,
                              Position: List[float]}]
                    ]

    Args:
        solver_path (Union[str, os.PathLike, Path]): Путь до папки с расчетом.
        save_path (Union[str, os.PathLike, Path]): Путь для сохранения итогового json.
        p (np.ndarray): Поле давления.
        u (np.ndarray): Поле скоростей.
        c (np.ndarray): Центры ячеек.
        surface_name (str): Имя для поверхности.
    """
    
    # Step 0: parse mesh and scale vertices
    mesh_bin = Ofpp.FoamMesh(solver_path )

    # Step I: compute TFemFace_Surface
    domain_names = ["motorBike_0".encode('ascii')]
    surfaces = []

    for i, domain_name in enumerate(domain_names):
        bound_cells = list(mesh_bin.boundary_cells(domain_name))

        boundary_faces = []
        boundary_faces_cell_ids = []
        for bc_id in bound_cells:
            faces = mesh_bin.cell_faces[bc_id]
            for f in faces:
                if mesh_bin.is_face_on_boundary(f, domain_name):
                    boundary_faces.append(f)
                    boundary_faces_cell_ids.append(bc_id)

        f_b_set = set(zip(boundary_faces, boundary_faces_cell_ids))

        body_faces = []
        for f, b in f_b_set:
            try:
                position = _face_center_position(mesh_bin.faces[f], mesh_bin)
                d = {'ParentElementID': b,
                    'ParentFaceId': f,
                    'Position': {'X': position[0], 'Y': position[1], 'Z': position[2]}
                    }
                body_faces.append(d)
            except IndexError:
                print(f'Indexes for points: {f} is not valid!')

        surfaces.append({'Item1': surface_name,
                'Item2': body_faces}) 
    
    # Define file paths
    nodes_path = os.path.join(save_path, 'Nodes.json')
    faces_path = os.path.join(save_path, 'Faces.json')
    elements_path = os.path.join(save_path, 'Elements.json')
    surfaces_path = os.path.join(save_path, 'Surfaces.json')

    # Prepare arguments for the multiprocessing function
    
    tasks = [
    (nodes_path, create_nodes_gen, (mesh_bin,)),
    (faces_path, create_faces_gen, (mesh_bin,)),
    (elements_path, create_elements_gen, (mesh_bin, p, u, c)),
    (surfaces_path, create_surfaces_gen, (surfaces,))
        ]

    # Use multiprocessing pool
    with Pool() as pool:
        pool.starmap(json_streaming_writer, tasks)



def pressure_field_on_surface(solver_path: Union[str, os.PathLike, Path],
                                 p: np.ndarray,
                                 surface_name: str = 'Surface') -> None:
    """Поле давлений на поверхности тела:
    'Nodes' - List[x: float, y: float, z:float], 
    'Faces' - List [List[int]], 
    'Elements' - List [Dict{Faces: List[int],
                            Pressure: float,
                            Velocity: List[float],
                            VelocityModule: float,
                            Position: List[float]}
                            ], 
    'Surfaces' - List[
                    Tuple[Surface_name: str, 
                    List[Dict{ParentElementID: int,
                              ParentFaceId: int,
                              Position: List[float]}]
                    ]

    Args:
        solver_path (Union[str, os.PathLike, Path]): Путь до папки с расчетом.
        p (np.ndarray): Поле давления.
        surface_name (str): Имя для поверхности.
    """
    
    # Step 0: parse mesh and scale vertices
    mesh_bin = Ofpp.FoamMesh(solver_path )

    # Step I: compute TFemFace_Surface
    domain_names = ["motorBike_0".encode('ascii')]
    surfaces = []

    for i, domain_name in enumerate(domain_names):
        bound_cells = list(mesh_bin.boundary_cells(domain_name))

        boundary_faces = []
        boundary_faces_cell_ids = []
        for bc_id in bound_cells:
            faces = mesh_bin.cell_faces[bc_id]
            for f in faces:
                if mesh_bin.is_face_on_boundary(f, domain_name):
                    boundary_faces.append(f)
                    boundary_faces_cell_ids.append(bc_id)

        f_b_set = set(zip(boundary_faces, boundary_faces_cell_ids))

        body_faces = []
        for f, b in f_b_set:
            try:
                position = _face_center_position(mesh_bin.faces[f], mesh_bin)
                d = {'ParentElementID': b,
                    'ParentFaceId': f,
                    'CentrePosition': {'X': position[0], 'Y': position[1], 'Z': position[2]},
                    'PressureValue': p[b]
                    }
                body_faces.append(d)
            except IndexError:
                print(f'Indexes for points: {f} is not valid!')

        surfaces.append({'Item1': surface_name,
                'Item2': body_faces}) 
        

        return surfaces


PATH_TO_CASE = input('PATH (example: "Sphere_stationary/0.3M"): ')
END_TIME = '150'


base_path = Path(PATH_TO_CASE)
time_path = base_path / Path(END_TIME)
p_path = time_path / Path('p')
p = Ofpp.parse_internal_field(p_path)

surface = pressure_field_on_surface(base_path, p)


all_dt = []
for s in surface[0]['Item2']:
    all_dt.append(s)
    #pprint(s)
    #break

pli = input("output file name (example.h5): ")
with h5py.File(pli, 'w') as hf:
    for i, element in enumerate(all_dt):
        group = hf.create_group(f'element_{i}')
        group.create_dataset('CentrePosition_X', data=element['CentrePosition']['X'])
        group.create_dataset('CentrePosition_Y', data=element['CentrePosition']['Y'])
        group.create_dataset('CentrePosition_Z', data=element['CentrePosition']['Z'])
        group.create_dataset('ParentElementID', data=element['ParentElementID'])
        group.create_dataset('ParentFaceId', data=element['ParentFaceId'])
        group.create_dataset('PressureValue', data=element['PressureValue'])


def convert_hdf5_to_elements(input_hdf5_file):
    elements = []
    with h5py.File(input_hdf5_file, 'r') as hf:
        for group_name in hf:
            group = hf[group_name]
            centre_position = {
                'X': group['CentrePosition_X'][()],
                'Y': group['CentrePosition_Y'][()],
                'Z': group['CentrePosition_Z'][()]
            }
            parent_element_id = group['ParentElementID'][()]
            parent_face_id = group['ParentFaceId'][()]
            pressure_value = group['PressureValue'][()]
            element = {
                'CentrePosition': centre_position,
                'ParentElementID': parent_element_id,
                'ParentFaceId': parent_face_id,
                'PressureValue': pressure_value
            }
            elements.append(element)
    return elements

elements = convert_hdf5_to_elements(pli)
with open(pli[:-3] + '_dec',"w+") as f:
    f.write(str(elements))
print(elements)