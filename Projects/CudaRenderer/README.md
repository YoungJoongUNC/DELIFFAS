# Quick demo for the forward pass of the CudaRenderer
To run a forward rendering demo, perform the following steps:
1. Navigate to the `Projects/CudaRenderer` directory
2. Create this directory `mkdir output_test_render`
3. Run `srun -p gpu20 --gres gpu:1 -t 00:05:00 --pty python test_render.py`
4. In `Projects/CudaRenderer/output_test_render` you should find images showing a cone. Each image is a different combination of shading modes and albedo modes.

**Some notes**:
- The depth rendering and vertex positions are in global scale /coordinate system. Thus, the rendered images might look "wrong".
- The normal rendering mode provides the normal orientations of the normals in global space mapped onto the 2D texture space.

# Quick demo for the backward pass of the CudaRenderer

## Optimizing light
This demo illustrates how the scene lighting (in terms of spherical harmonics) can be optimized using a rendering loss under the assumption the geometry and albedo (texture or vertex colors) are known.
1. Navigate to the `Projects/CudaRenderer` directory
2. Create this directory `mkdir output_test_gradient_light`
3. Run `srun -p gpu20 --gres gpu:1 -t 00:15:00 --pty python test_gradient_light.py`
4. In `Projects/CudaRenderer/output_test_gradient_light` you should find the images of the intermediate iterations of the optimization as well as the final result

## Optimizing texture
This demo illustrates how the texture of an object can be optimized using a dense rendering loss under the assumption the geometry and the lighting are known.
1. Navigate to the `Projects/CudaRenderer` directory
2. Create this directory `mkdir output_test_gradient_texture`
3. Run `srun -p gpu20 --gres gpu:1 -t 00:15:00 --pty python test_gradient_texture.py`
4. In `Projects/CudaRenderer/output_test_gradient_texture` you should find the images of the intermediate iterations of the optimization as well as the final result

## Optimizing vertex colors
This demo illustrates how the vertex colors of an object can be optimized using a dense rendering loss under the assumption the geometry and the lighting are known.

1. Navigate to the `Projects/CudaRenderer` directory
2. Create this directory `mkdir output_test_gradient_vertex_colors`
3. Run `srun -p gpu20 --gres gpu:1 -t 00:15:00 --pty python test_gradient_vertex_color.py`
4. In `Projects/CudaRenderer/output_test_gradient_vertex_colors` you should find the images of the intermediate iterations of the optimization as well as the final result

## Optimizing vertex deformations
This demo illustrates how the vertex positions of an object can be optimized using a dense rendering loss under the assumption the appearance (texture or vertex colors) and the lighting are known.

1. Navigate to the `Projects/CudaRenderer` directory
2. Create this directory `mkdir output_test_gradient_vertex_positions`
3. Run `srun -p gpu20 --gres gpu:1 -t 00:15:00 --pty python test_gradient_vertex_positions.py`
4. In `Projects/CudaRenderer/output_test_gradient_vertex_positions` you should find the images of the intermediate iterations of the optimization as well as the final result
