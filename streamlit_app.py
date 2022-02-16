# from streamlit.state.session_state import WidgetArgs
import torch
import streamlit as st
from streamlit import components
from streamlit import file_util
# import streamlit.report_thread as ReportThread
# from streamlit.server.server import Server
# from streamlit import caching
from streamlit import script_runner
from streamlit.server.server import StaticFileHandler
from contextlib import contextmanager
from io import StringIO
from streamlit.script_run_context import SCRIPT_RUN_CONTEXT_ATTR_NAME
from threading import current_thread

from utils.visualizers import Visualizers

import os, sys
import PIL
import urllib.request
import numpy as np

import open3d as o3d


@classmethod
def _get_cached_version(cls, abs_path: str):
    #taken from https://discuss.streamlit.io/t/html-file-cached-by-streamlit/9289
    with cls._lock:
        return cls.get_content_version(abs_path)


@contextmanager
def st_redirect(src, dst):
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), SCRIPT_RUN_CONTEXT_ATTR_NAME, None):
                buffer.write(b)
                output_func(buffer.getvalue())
            else:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write


@contextmanager
def st_stdout(dst):
    with st_redirect(sys.stdout, dst):
        yield


@contextmanager
def st_stderr(dst):
    with st_redirect(sys.stderr, dst):
        yield


def header_properties(field_list, field_names):

    # List of lines to write
    lines = []

    # First line describing element vertex
    lines.append('element vertex %d' % field_list[0].shape[0])

    # Properties lines
    i = 0
    for fields in field_list:
        for field in fields.T:
            lines.append('property %s %s' % (field.dtype.name, field_names[i]))
            i += 1

    return lines


def write_ply(filename, field_list, field_names, triangular_faces=None):
    """
    Write ".ply" files

    Parameters
    ----------
    filename : string
        the name of the file to which the data is saved. A '.ply' extension will be appended to the 
        file name if it does no already have one.

    field_list : list, tuple, numpy array
        the fields to be saved in the ply file. Either a numpy array, a list of numpy arrays or a 
        tuple of numpy arrays. Each 1D numpy array and each column of 2D numpy arrays are considered 
        as one field. 

    field_names : list
        the name of each fields as a list of strings. Has to be the same length as the number of 
        fields.

    Examples
    --------
    >>> points = np.random.rand(10, 3)
    >>> write_ply('example1.ply', points, ['x', 'y', 'z'])

    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example2.ply', [points, values], ['x', 'y', 'z', 'values'])

    >>> colors = np.random.randint(255, size=(10,3), dtype=np.uint8)
    >>> field_names = ['x', 'y', 'z', 'red', 'green', 'blue', values']
    >>> write_ply('example3.ply', [points, colors, values], field_names)

    """

    # Format list input to the right form
    field_list = list(field_list) if (type(field_list) == list
                                      or type(field_list) == tuple) else list(
                                          (field_list, ))
    for i, field in enumerate(field_list):
        if field.ndim < 2:
            field_list[i] = field.reshape(-1, 1)
        if field.ndim > 2:
            print('fields have more than 2 dimensions')
            return False

    # check all fields have the same number of data
    n_points = [field.shape[0] for field in field_list]
    if not np.all(np.equal(n_points, n_points[0])):
        print('wrong field dimensions')
        return False

    # Check if field_names and field_list have same nb of column
    n_fields = np.sum([field.shape[1] for field in field_list])
    if (n_fields != len(field_names)):
        print('wrong number of field names')
        return False

    # Add extension if not there
    if not filename.endswith('.ply'):
        filename += '.ply'

    # open in text mode to write the header
    with open(filename, 'w') as plyfile:

        # First magical word
        header = ['ply']

        # Encoding format
        header.append('format binary_' + sys.byteorder + '_endian 1.0')

        # Points properties description
        header.extend(header_properties(field_list, field_names))

        # Add faces if needded
        if triangular_faces is not None:
            header.append('element face {:d}'.format(
                triangular_faces.shape[0]))
            header.append('property list uchar int vertex_indices')

        # End of header
        header.append('end_header')

        # Write all lines
        for line in header:
            plyfile.write("%s\n" % line)

    # open in binary/append to use tofile
    with open(filename, 'ab') as plyfile:

        # Create a structured array
        i = 0
        type_list = []
        for fields in field_list:
            for field in fields.T:
                type_list += [(field_names[i], field.dtype.str)]
                i += 1
        data = np.empty(field_list[0].shape[0], dtype=type_list)
        i = 0
        for fields in field_list:
            for field in fields.T:
                data[field_names[i]] = field
                i += 1

        data.tofile(plyfile)

        if triangular_faces is not None:
            triangular_faces = triangular_faces.astype(np.int32)
            type_list = [('k', 'uint8')] + [(str(ind), 'int32')
                                            for ind in range(3)]
            data = np.empty(triangular_faces.shape[0], dtype=type_list)
            data['k'] = np.full((triangular_faces.shape[0], ),
                                3,
                                dtype=np.uint8)
            data['0'] = triangular_faces[:, 0]
            data['1'] = triangular_faces[:, 1]
            data['2'] = triangular_faces[:, 2]
            data.tofile(plyfile)

    return True


def export_mesh(viz, pred_xyz, colors, static_path):
    isMesh = False
    while not isMesh:
        mes_warn = st.warning("Reconstructing mesh...")
        mes_warn_2 = st.warning("This might take a while...")
        mesh = viz.export_mesh(pred_xyz, None)
        #save mesh
        mesh_filename = os.path.join(static_path, "pred_mesh.obj")
        if os.path.isfile(mesh_filename):
            os.remove(mesh_filename)
        o3d.io.write_triangle_mesh(mesh_filename,
                                   mesh,
                                   write_triangle_uvs=True)
        if mesh:
            isMesh = True
            mes_warn.empty()
            mes_warn_2.empty()


def main():
    #Do not cache files by filename.
    StaticFileHandler._get_cached_version = _get_cached_version
    st.set_page_config(layout="wide")
    static_path = file_util.get_static_dir()
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/anonymous-eccv2022-submission/eccv2022-4d/main/images/pipeline.png',
        os.path.join(static_path, "banner.png"))
    banner = PIL.Image.open(os.path.join(static_path, "banner.png"))
    st.image(banner, use_column_width=True)
    st.markdown(
        "<h1 style='text-align: center; color: white;'>Joint Learning for 4D Reconstruction and Flow Estimation of Point Clouds</h1>",
        unsafe_allow_html=True)
    st.markdown("## Demo:", unsafe_allow_html=True)
    st.markdown(
        "This is a live demo showcasing the models developed within 4D Reconstruction.",
        unsafe_allow_html=True)
    st.markdown(
        "You can infer your's human or object geometry and donwnload the predicted mesh by simply uploading point clouds.",
        unsafe_allow_html=True)
    st.markdown("## Note that: ", unsafe_allow_html=True)
    st.markdown(
        "Please be aware that the mesh reconstruction could take some time as all of the processing is conducted on a CPU. Before uploading a new data please refresh your browser.",
        unsafe_allow_html=True)
    st.markdown("## Upload point clouds: ", unsafe_allow_html=True)

    input_PC = st.file_uploader('Upload your point clouds here', type='npz')

    viz = Visualizers()

    if input_PC is not None:

        input_pc_filename = os.path.join(static_path, 'input_pointcloud.ply')
        if os.path.isfile(input_pc_filename):
            os.remove(input_pc_filename)
        pointcloud_dict = np.load(input_PC)
        # with st_stdout("code"):
        #     print(pointcloud_dict)
        #     print(pointcloud_dict.shape)
        points = pointcloud_dict['points'].astype(np.float32)
        # with st_stdout("info"):
        #     print(points.shape)

        index = np.random.choice(points.shape[0], 3000, replace=False)
        points = points[index]
        # with st_stdout("info"):
        #     print(points.shape)
        write_ply(input_pc_filename, points, ['x', 'y', 'z'])

        text_file = open("./html/ply.html", "r")
        #read whole file to a string
        html_string = text_file.read()
        st.markdown("## Input Point Clouds", unsafe_allow_html=True)
        st.markdown(
            "Inspect the input point clouds through the interactive 3D Model Viewer",
            unsafe_allow_html=True)
        h1 = components.v1.html(html_string, height=600)
        st.success("Point clouds has been created!")
        linko = f'<a href="input_pointcloud.ply" download="input_pointcloud.ply"><button kind="primary" class="css-15r570u edgvbvh1">Download Point Cloud!</button></a>'
        st.markdown(linko, unsafe_allow_html=True)




        if st.button('Reconstruct mesh'):
            # index = np.random.choice(points.shape[0], 3000, replace=False)
            # points = points[index]
            export_mesh(viz, points, None, static_path)
            text_file = open("./html/mesh.html", "r")
            #read whole file to a string
            html_string = text_file.read()
            text_file.close()
            st.markdown(
                "## Reconstructed Mesh using Poisson Surface Reconstruction",
                unsafe_allow_html=True)
            st.markdown(
                "Inspect the reconstructed mesh through the interactive 3D Model Viewer",
                unsafe_allow_html=True)
            h2 = components.v1.html(html_string, height=600)
            st.success("Mesh has been created!")
            linko = f'<a href="pred_mesh.obj" download="pred_mesh.obj"><button kind="primary" class="css-15r570u edgvbvh1">Download Reconstructed Mesh!</button></a>'
            st.markdown(linko, unsafe_allow_html=True)
            #Acknow
            st.markdown(
                "<h2 style='text-align: center; color: white;'>Thank you !!!</h2>",
                unsafe_allow_html=True)


if __name__ == '__main__':
    main()