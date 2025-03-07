#!/usr/bin/python
# -- coding: utf-8 --

import os, sys
import argparse
import numpy as np
import multiprocessing
import logging

import matplotlib.path as mpltPath
import scipy.ndimage
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull

from tifffile import tifffile

# https://github.com/mrirecon/dl-segmentation-realtime-cmr/blob/main/scripts/assess_dl_seg_utils.py
def mask_from_polygon_mplt(coordinates:list, img_shape:list):
	"""
	Transfer a polygon into a binary mask by using matplotlib.Path

	:param list coordinates: List of coordinates in format [[x1,y1], [x2,y2], ...]
	:param list img_shape: Shape of the output mask in format [xdim,ydim]
	:returns: Binary 2D mask
	:rtype: np.array
	"""
	path = mpltPath.Path(coordinates)
	points = [[x+0.5,y+0.5] for y in range(img_shape[1]) for x in range(img_shape[0])]
	inside2 = path.contains_points(points)
	new_mask = np.zeros(img_shape, dtype=np.int32)
	count=0
	for y in range(img_shape[1]):
		for x in range(img_shape[0]):
			new_mask[x,y] = int(inside2[count])
			count += 1
	return new_mask

def transform_polygon_mp(args:tuple):
	"""
	Multiprocessing the transformation of a polygon to a binary mask.

	:param tuple args: tuple(coordinates, img_shape)
		WHERE
		list coordinates is list of coordinates in format [[x1,y1], [x2,y2], ...]
		list img_shape is list of dimensions of output mask [xdim, ydim]
	:returns: binary mask
	:rtype:	np.array
	"""
	(coordinates, label_value, img_shape) = args
	mask = mask_from_polygon_mplt(coordinates, img_shape)
	mask[mask > 0] = label_value
	return mask

def combined_mask_from_list(mask_list, label_values):
	"""
	Create a combined mask for an input list of binary masks and label values.
	The binary masks are assigned the corresponding label value.
	Overlap is removed, higher label values take precedence.
	"""
	new_mask = np.zeros((mask_list[0].shape[0], mask_list[0].shape[1]), dtype=np.int32)
	for (mask, label) in zip(mask_list, label_values):
		new_mask += mask
		new_mask[new_mask > label] = label
	return new_mask

def make_labels_convex_mp(label_arr, mp_number = 8):
	"""
	Multi-processing for creating convex labels.

	:param np.ndarray label_arr: Array containing input labels with format [x, y, z]
	:param int mp_number: Number of multi-processes
	:return: convex labels
	:rtype: np.ndarray
	"""
	label_list = [label_arr[:,:,i] for i in range(label_arr.shape[2])]
	pool = multiprocessing.Pool(processes=mp_number)
	mask_list = pool.map(make_labels_convex, label_list)
	mask_stack = np.stack(mask_list, axis=2)
	return mask_stack

def make_labels_convex(arr, min_pixels_register = 10, min_pixels_new = 250):
	"""
	Create convex labels.
	Input labels with less than "min_pixels_register" are ignored and removed.
	Convex labels with less than "min_pixels_new" are removed.

	:param np.ndarray arr: 2D array in format [x, y]
	:param int min_pixels_register: Minimum size for input labels.
	:param int min_pixels_new: Minimum size for output labels.
	"""
	label_values = []
	mask_list = []
	if 0 != np.max(arr):
		for label_idx in range(1, np.max(arr)+1):
			if np.count_nonzero(arr == label_idx) > min_pixels_register:
				mask = arr.copy()
				mask[mask != label_idx] = 0
				coordinates = []

				for x in range(mask.shape[0]):
					for y in range(mask.shape[1]):
						if mask[x,y] != 0:
							coordinates.append([x,y])

				hull = ConvexHull(coordinates)
				edge_coords = []

				for idx in hull.vertices:
					edge_coords.append(coordinates[idx])

				new_mask = mask_from_polygon_mplt(edge_coords, arr.shape)
				new_mask[new_mask > 0] = label_idx

				if np.count_nonzero(arr == label_idx) > min_pixels_new:
					mask_list.append(new_mask)
					label_values.append(label_idx)

			# remove small labels
			else:
				arr[arr == label_idx] = 0

		if len(mask_list) > 0:
			return combined_mask_from_list(mask_list, label_values)
		else:
			return arr

	else:
		return arr


def read_tif_stack(file):
	"""
	Read stack of TIF files.
	"""
	images = tifffile.imread(file)
	images = np.transpose(images, (1,2,0))
	return images

def affine_transform_euler(data, euler_angles, label_flag = False):
	"""
	Affine transform using Euler angles as input.
	"""
	# Euler angles [degree]
	# https://quaternions.online/ for visualization
	(ex, ey, ez) = euler_angles

	rot_obj = R.from_euler('xyz', [ex, ey, ez], degrees=True)

	# calculate offset to have center of input at center of output
	(xdim, ydim, zdim) = data.shape
	x_vec = np.array([[xdim // 2, ydim // 2, zdim // 2]])
	rot_matrix = rot_obj.as_matrix()
	y_vec = np.dot(rot_matrix, x_vec.T)
	t_vec = x_vec.T - y_vec
	offset = [t_vec[0][0], t_vec[1][0], t_vec[2][0]]

	if label_flag:
		result = scipy.ndimage.affine_transform(data, rot_matrix, order=0, offset=offset, prefilter=False)
	else:
		result = scipy.ndimage.affine_transform(data, rot_matrix, offset=offset)
	return result

def pad_scaled_output(arr, target_shape, pad_type = 'zero'):
	"""
	Pad input array, either with constant value 'zero'
	or the mean value of corner sections of the volume with the smallest standard deviation.

	:param np.ndarray arr: Input array in format [x, y, z]
	:param tuple target_shape: Shape of the padded volume in format (x, y, z)
	:param str pad_type: Either 'zero' or 'mean'
	:returns: Padded input
	:rtype: np.ndarray
	"""

	if "mean" == pad_type:
		corner_arrays = [arr[0:arr.shape[0]//10, 0:arr.shape[1]//10, :], arr[0:arr.shape[0]//10, -arr.shape[1]//10:], arr[-arr.shape[0]//10:, 0:arr.shape[1]//10], arr[-arr.shape[0]//10:, -arr.shape[1]//10:]]
		stdv = [np.std(a) for a in corner_arrays]
		min_std_index = stdv.index(min(stdv))
		pad_value = np.mean(corner_arrays[min_std_index])
	elif "zero" == pad_type:
		pad_value = 0
	else:
		sys.exit("Choose either 'zero' or 'mean' for padding.")

	logging.info("Using padding with pad_value " + str(pad_value))

	pad_before_x = (target_shape[0] - arr.shape[0]) // 2
	pad_after_x = target_shape[0] - pad_before_x - arr.shape[0]

	pad_before_y = (target_shape[1] - arr.shape[1]) // 2
	pad_after_y = target_shape[1] - pad_before_y - arr.shape[1]

	return np.pad(arr, ((pad_before_x, pad_after_x), (pad_before_y, pad_after_y), (0,0)), constant_values = pad_value)

def scale_by_factor(array, scale, label_flag = False):
	"""
	Scaling an array by a given factor.
	"""
	# scaling by factor 1 / s
	s = 1 / scale
	matrix = np.asarray([
		[s, 0, 0, 0],
		[0, s, 0, 0],
		[0, 0, 1, 0],
		[0, 0, 0, 1],
	])
	output_shape = (int(array.shape[0] / s), int(array.shape[1] / s), array.shape[2])

	if label_flag:
		scaled = np.ndarray(output_shape, dtype=np.int32)
		result = scipy.ndimage.affine_transform(array, matrix, order=0, output=scaled, output_shape=output_shape, prefilter=False)
	else:
		scaled = np.ndarray(output_shape, dtype=np.uint16)
		result = scipy.ndimage.affine_transform(array, matrix, output=scaled, output_shape=output_shape)

	return result

def main(input_file, dir_out, scale, ex, ey, ez, make_convex):
	# check file format
	if not os.path.isfile(input_file):
		sys.exit("Input file does not exist.")

	if input_file.split(".")[-1] not in ["TIFF", "TIF", "tiff", "tif"]:
		sys.exit("Input file must be in tif format.")

	basename = input_file.split("/")[-1].split(".tif")[0]

	if scale != 1 and not (ex == 0 and ey == 0 and ez == 0):
		sys.exit("Either scaling or rotation. A combination has not been implemented yet.")

	# check for corresponding annotations
	data_dir = input_file.split(basename)[0]
	label_path = data_dir + basename + "_annotations.tif"
	if not os.path.isfile(label_path):
		logging.debug("No corresponding label was found.")
		label_path = ""

	if "" == dir_out:
		logging.debug("The output is stored in the directory containing the input, since no output directory has been given.")
		dir_out = data_dir

	image_file = os.path.join(data_dir, basename + ".tif")
	images = read_tif_stack(image_file)

	#---Images---
	if scale != 1:
		images_aff = scale_by_factor(images, scale)
		images_aff = pad_scaled_output(images_aff, images.shape, pad_type = "zero")
		save_images = os.path.join(dir_out, basename + "_aff_scaled_" + str(scale) + ".tif")

	else:
		images_aff = affine_transform_euler(images, (ex, ey, ez))
		save_images = os.path.join(dir_out, basename + "_affExyz" + str(int(ex)).zfill(2) + str(int(ey)).zfill(2) + str(int(ez)).zfill(2) + ".tif")

	array_out = np.transpose(images_aff, (2,0,1))
	tifffile.imwrite(save_images, array_out)

	#---Labels---
	if label_path != "":
		label_file = os.path.join(data_dir, basename + "_annotations.tif")
		labels = read_tif_stack(label_file)

		if scale != 1:
			labels_aff = scale_by_factor(labels, scale)
			labels_aff = pad_scaled_output(labels_aff, labels.shape, pad_type = "zero")
			save_labels = os.path.join(dir_out, basename + "_aff_scaled_" + str(scale) + "_annotations.tif")

		else:
			labels_aff = affine_transform_euler(labels, (ex, ey, ez), label_flag = True)
			if make_convex:
				labels_aff = make_labels_convex_mp(labels_aff, mp_number = 16)

			save_labels = os.path.join(dir_out, basename + "_affExyz" + str(int(ex)).zfill(2) + str(int(ey)).zfill(2) + str(int(ez)).zfill(2) + "_annotations.tif")

		array_out = np.transpose(labels_aff, (2,0,1))
		tifffile.imwrite(save_labels, array_out)

if __name__ == "__main__":

	parser = argparse.ArgumentParser(
		description="Script to augment LSM data in tif format using rotation or scaling.")

	parser.add_argument('input', type=str, help="Input image file")

	parser.add_argument('-o', "--output", type=str, default="", help="Output directory")
	parser.add_argument('-c', "--convex", action='store_true', help="Flag for making affine transformed output labels convex.")

	parser.add_argument('-s', "--scale", type=float, default=1, help="Factor to scale input with affine transformation. Only supports s<=1.")
	parser.add_argument('--ex', type=float, default=0, help="Euler angle x")
	parser.add_argument('--ey', type=float, default=0, help="Euler angle y")
	parser.add_argument('--ez', type=float, default=0, help="Euler angle z")

	args = parser.parse_args()

	main(args.input, args.output, args.scale, args.ex, args.ey, args.ez, args.convex)