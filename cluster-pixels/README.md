# Clustering Image Pixels

`main.py` clusters pixels in an image based on both their colors and positions,
chooses a uniform color for each cluster, and then smooths the result out.

Example:
`python main.py example_source.jpg --dest_filename example_dest.jpg`

<div style="text-align:center">
<img src="./example_source.jpg" width="45%" alt="source image">
<img src="./example_dest.jpg" width="45%" alt="dest image">
</div>

It also outputs a "color palette" showing the position, color, and size of each centroid:
<img src="./example_color_palette.png" alt="color palette">
