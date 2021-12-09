# polynomial_surface_fitting


## Sample deployment

singularity build test.simg docker:phytooracle/polynomial_surface_fitting_s10
singularity run test.simg -i ./Wintercrop_3/

## Output structure

### If -c for cropping
where_you_ran_it/segmentation_pointclouds/plant_names/poly_crop.ply

where_you_ran_it/plant_reports/plant_names/poly_crop.csv

(figures to be added)

### In not -c, only counting
where_you_ran_it/plant_reports/plant_names/poly_n.csv

