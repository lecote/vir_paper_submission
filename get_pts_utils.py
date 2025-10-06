import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from magicgui.widgets import ComboBox, Container
from bioio import BioImage
import napari

COLOR_CYCLE = [
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf'
]


def get_ball_coords(radius: int, center: Tuple[int]) -> Tuple[np.ndarray]:
    """
    Use radius and center to return the coordinates within that 3d region
    as a 'slice'.
    """

    coords = np.nonzero(mo.ball(radius))
    # 'coords' is a tuple of 1d arrays - to move center using pure numpy, 
    # first convert to a 2d array
    coords_array = np.array(coords)
    center_array = np.array([center]).T

    # transform coordinates to be centered at 'center'
    coords_array = coords_array - radius + center_array
    # convert coordinates back to tuple of 1d arrays, which can be used
    # directly as a slice specification
    coords_tuple = (
        coords_array[0,:],
        coords_array[1,:],
        coords_array[2,:]
    )

    return coords_tuple


def create_label_menu(points_layer, labels):
    """Create a label menu widget that can be added to the napari viewer dock
    
    Requires manually changing the first point clicked to first label, then it will cycle appropriately.

    Parameters
    ----------
    points_layer : napari.layers.Points
        a napari points layer
    labels : List[str]
        list of the labels for each keypoint to be annotated (e.g., the body parts to be labeled).

    Returns
    -------
    label_menu : Container
        the magicgui Container with our dropdown menu widget
    """
    # Create the label selection menu
    label_menu = ComboBox(label='feature_label', choices=labels)
    label_widget = Container(widgets=[label_menu])


    def update_label_menu(event):
        """Update the label menu when the point selection changes"""
        new_label = str(points_layer.current_properties['label'][0])
        if new_label != label_menu.value:
            label_menu.value = new_label

    points_layer.events.current_properties.connect(update_label_menu)

    def label_changed(selected_label):
        """Update the Points layer when the label menu selection changes"""
        current_properties = points_layer.current_properties
        current_properties['label'] = np.asarray([selected_label])
        points_layer.current_properties = current_properties
        points_layer.refresh_colors()

    label_menu.changed.connect(label_changed)

    return label_widget


def point_annotator(
        im_path: str,
        im_prop: dict,
        csv_path: str,
        labels: List[str],
        scale_by_px=False,
        channel=0,
        radius=3
):
    """Create a GUI for annotating points in a series of images.

    Parameters
    ----------
    im_path : str
        glob-like string for the images to be labeled.
    im_prop : dict
        dict of **kwargs to pass to view_image.
    labels : List[str]
        list of the labels for each keypoint to be annotated (e.g., the body parts to be labeled).
    csv_path : str
        glob-like string for saving the points csv files.
    radius : int
        radius for pt size. Default 3.
    """
    #stack = imread(im_path)
    stack = BioImage(im_path)
    if scale_by_px == True:
        scale_bypx = [(d) for d in stack.physical_pixel_sizes]
    else:
        scale_bypx = None
    
    dirname, filename = os.path.split(im_path)
    img_name = os.path.join(os.path.split(dirname)[1],filename)

    #viewer = napari.view_image(stack)
    viewer = napari.view_image(stack.data, scale = scale_bypx, **im_prop)
    #viewer = napari.view_image(stack.data, **im_prop)
    
    
    points_layer = viewer.add_points(
        ndim=3,
        scale = scale_bypx,
        property_choices={'label': labels, 'img': [img_name]}, #'sigma': [pt_size]},
        edge_color='label',
        edge_color_cycle=COLOR_CYCLE,
        symbol='o',
        face_color='transparent',
        out_of_slice_display=True,
        edge_width=0.5,  # fraction of point size
        size=radius
    )
    points_layer.edge_color_mode = 'cycle'

    # add the label menu widget to the viewer
    label_widget = create_label_menu(points_layer, labels)
    viewer.window.add_dock_widget(label_widget)
        
    @viewer.bind_key('.')
    def next_label(event=None):
        """Keybinding to advance to the next label with wraparound"""
        current_properties = points_layer.current_properties
        current_label = current_properties['label'][0]
        ind = list(labels).index(current_label)
        new_ind = (ind + 1) % len(labels)
        new_label = labels[new_ind]
        current_properties['label'] = np.array([new_label])
        points_layer.current_properties = current_properties
        points_layer.refresh_colors()

    def next_on_click(layer, event):
        """Mouse click binding to advance the label when a point is added"""
        if layer.mode == 'add':
            # By default, napari selects the point that was just added.
            # Disable that behavior, as the highlight gets in the way
            # and also causes next_label to change the color of the
            # point that was just added.
            layer.selected_data = set()
            next_label()

    points_layer.mode = 'add'
    points_layer.mouse_drag_callbacks.append(next_on_click)

    @viewer.bind_key(',')
    def prev_label(event):
        """Keybinding to decrement to the previous label with wraparound"""
        current_properties = points_layer.current_properties
        current_label = current_properties['label'][0]
        ind = list(labels).index(current_label)
        n_labels = len(labels)
        new_ind = ((ind - 1) + n_labels) % n_labels
        new_label = labels[new_ind]
        current_properties['label'] = np.array([new_label])
        points_layer.current_properties = current_properties
        points_layer.refresh_colors()
    
    @viewer.bind_key('s')
    def save_csv(event):
        """Keybinding to save point layer with image name metadata"""
        channel_name = im_prop['name'][channel]
        num_pts = len(points_layer.data)
        stats_intensities = np.zeros((5, num_pts))
        for p in range(num_pts):
            imdata = stack.data[0][channel]
            pt_coords = points_layer.data[p,:].round().astype(int)
            region_coords = get_ball_coords(radius, pt_coords)
            stats_intensities[:,p] = [imdata[region_coords].mean(), imdata[region_coords].sum(), imdata[region_coords].max(), imdata[region_coords].min(), imdata[region_coords].std()]
        points_layer.features = pd.concat([points_layer.features, pd.DataFrame({'mean_intensity_'+channel_name : stats_intensities[0],
                              'sum_intensity_'+channel_name : stats_intensities[1],
                              'max_intensity_'+channel_name : stats_intensities[2],
                              'min_intensity_'+channel_name : stats_intensities[3],
                              'std_intensity_'+channel_name : stats_intensities[4]})], axis = 1)
        points_layer.save(csv_path)
        

    napari.run()
