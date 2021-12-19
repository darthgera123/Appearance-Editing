import torch, os, sys, cv2, json, argparse, random, glob, struct, math, time
import pyredner

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_file', type=str, default='')
	parser.add_argument('--output_file', type=str, default='./', help='')

	args = parser.parse_args()

	obj = pyredner.load_obj(args.input_file, return_objects=True)[0]
	uv_vertices, uv_indices = pyredner.compute_uvs(obj.vertices, obj.indices, print_progress=True)

	obj.uvs = uv_vertices
	obj.uv_indices = uv_indices

	pyredner.save_obj(obj, args.output_file)
