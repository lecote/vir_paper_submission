import os, time, itertools
from typing import List, Tuple

import numpy as np
import pandas as pd

from skimage import morphology as mo
from collections import defaultdict

from magicgui.widgets import ComboBox, Container, TextEdit, LineEdit, PushButton, CheckBox
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

def keep_values_to_metadata(points_layer, keep_values):
    """Create row based on keep values structure of 'vir_A-3_HMR-1' and 'mean_intensity_1_HMR-1'"""
    #this should probably be a class
    
    def turn_str_into_dict(v_str):
        v_string = v_str.split("--")
        v_string = [v_string[0], v_string[1].split("_")]
        #TODO add check for correct format and raiseError if incorrect
        v_dict = {'pt' : v_string[0], 'r' : v_string[1][0], 'ch' : v_string[1][1]}
        return v_dict
    
    def get_bkgd_sub_value(v_str):
        v_dict = turn_str_into_dict(v_str)
        col_to_use = 'mean_intensity_'+v_dict['r']+'_'+v_dict['ch']
        df_use = points_layer.features.filter(items=['label',col_to_use])
        df_use = df_use.set_index('label')
        bkgd = df_use.filter(like="bkgd", axis=0).mean()
        bkgdsub_value = df_use.loc[v_dict['pt']].item() - bkgd.item()
        return bkgdsub_value
    
    for v in keep_values:
        points_layer.metadata[v] = get_bkgd_sub_value(v)
            
def measure_intensities(stack, points_layer, channels, radius, im_prop, img_name):
    """Get intensity statistics at each point and add to features"""
    print_flag=0
    for channel in channels:
        channel_name = im_prop['name'][channel]
        num_pts = len(points_layer.data)
        stats_intensities = np.full([5, num_pts, len(radius)], np.nan)
        dict_to_update = defaultdict(List)
        for r in range(len(radius)):
            for p in range(num_pts):
                imdata = stack.data[0][channel]
                pt_coords = points_layer.data[p,:].round().astype(int)
                region_coords = get_ball_coords(radius[r], pt_coords)
                try:
                    stats_intensities[:,p,r] = [imdata[region_coords].mean(), 
                                                imdata[region_coords].sum(), 
                                                imdata[region_coords].max(), 
                                                imdata[region_coords].min(), 
                                                imdata[region_coords].std()]

                except IndexError:
                    if print_flag == 0:
                        print('IndexError in ' + str(radius[r]) + 'px radius in '+img_name)
                        print_flag = 1

            dict_to_update.update(
                    {'mean_intensity_'+str(radius[r])+'_'+channel_name : stats_intensities[0,:,r],
                     'sum_intensity_'+str(radius[r])+'_'+channel_name  : stats_intensities[1,:,r],
                     'max_intensity_'+str(radius[r])+'_'+channel_name  : stats_intensities[2,:,r],
                     'min_intensity_'+str(radius[r])+'_'+channel_name  : stats_intensities[3,:,r],
                     'std_intensity_'+str(radius[r])+'_'+channel_name  : stats_intensities[4,:,r]
                    })
        points_layer.features = pd.concat([points_layer.features, pd.DataFrame(dict_to_update)], axis = 1)
            
def append_csv(points_layer, metadata_file):
        df_metadata = pd.DataFrame([points_layer.metadata]);
        if not os.path.exists(metadata_file):
            df_metadata.to_csv(metadata_file, mode='w', index=False, header=True)
        else:
            
            df_metadata.to_csv(metadata_file, mode='a', index=False, header=False)
            
def create_notes_menu(points_layer, 
                      default_notes_values, 
                      channels, 
                      radius, 
                      keep_values, 
                      metadata_file, 
                      im_prop, 
                      csv_path,
                     img_name):
    """Create a notes  widget that can be added to the napari viewer dock
    
    Parameters
    ----------
    points_layer : napari.layers.Points
        a napari points layer
    default_notes_values : Dict
        default notes value.

    Returns
    -------
    label_menu : Container
        the magicgui Container with our notes widget
    """
    
    #create widget
    worm_box = LineEdit(label='worm', value = default_notes_values['worm'])
    use_box = ComboBox(label='use', choices = default_notes_values['use'])
    stage_box = LineEdit(label='stage', value = default_notes_values['stage'])
    red_box = ComboBox(label='mCherry', choices = default_notes_values['red'])
    notes_box = LineEdit(label='notes', value = default_notes_values['notes'])
    save_button = PushButton(text='update metadata', value = False)
    notes_widget = Container(widgets=[worm_box, use_box, stage_box, red_box, notes_box, save_button])
    
    def update_metadata():
        current_properties = points_layer.metadata
        current_properties = {'img': [img_name],
                              'worm' : worm_box.value, 
                                          'use' : use_box.value,
                                          'stage' : stage_box.value,
                                          'red' : red_box.value,
                                          'notes' : notes_box.value.rstrip('\n\r')}
        points_layer.metadata = current_properties
        #print(points_layer.metadata)
        points_layer.events.properties.emitted = True
        worm_name = points_layer.metadata['worm']
        points_layer.features['worm'] = worm_name


    save_button.changed.connect(update_metadata)
    
    return notes_widget


def point_annotator(
        im_path: str,
        im_prop: dict,
        csv_path: str,
        metadata_file: str,
        labels: List[str],
        keep_values: List[str],
        default_notes_values={'worm' : 'middle'},
        scale_by_px=False,
        channels=[0],
        radius=[3]
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
    keep : List[str]
        list of values to bkgd subtract and keep
    default_notes_value : dict
        dict of choices for notes
    csv_path : str
        glob-like string for saving the points csv files.
    channels : List[int]
        channels other than quantify. Default is 0.
    radius : List[int]
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
        property_choices={'label': labels, 'img': [img_name], 'worm': default_notes_values['worm']},
        metadata={'img': [img_name], 
                'worm': default_notes_values['worm'][0],
                'use': default_notes_values['use'][0],
                'stage' : default_notes_values['stage'][0], 
                'red' : default_notes_values['red'][0], 
                'notes' : default_notes_values['notes'][0] },
        edge_color='label',
        edge_color_cycle=COLOR_CYCLE,
        symbol='o',
        face_color='transparent',
        out_of_slice_display=True,
        edge_width=0.5,  # fraction of point size
        size=radius[0]
    )
    points_layer.edge_color_mode = 'cycle'

    # add the label menu widget to the viewer
    label_widget = create_label_menu(points_layer, labels)
    notes_widget = create_notes_menu(points_layer, 
                                     default_notes_values, 
                                     channels, 
                                     radius, 
                                     keep_values, 
                                     metadata_file,
                                    im_prop,
                                    csv_path,
                                    img_name)
    viewer.window.add_dock_widget(notes_widget)
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
        """Keybinding to save point layer with image name"""
        measure_intensities(stack, points_layer, channels, radius, im_prop, img_name)
        points_layer.save(str(csv_path))
        keep_values_to_metadata(points_layer, keep_values)
        print(points_layer.metadata)
        append_csv(points_layer, metadata_file)
        
    viewer.show(block=True)
    napari.run()
