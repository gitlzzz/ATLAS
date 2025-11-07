## Database Generation

Generate a database generation template file using `mdb_gen_configuration_file -t database_generation`.

:::{attention}
All keys are mandatory unless stated otherwise.
:::


### Database - `[database]`

General settings and file paths for the database.


- :alt:`database_name`: (str) Name of the database to be used for internal reference and as the filename.
  - Example: `'my_material_db'`.

- :alt:`min_num_atoms`: (int) Minimum number of atoms allowed in the generated structures.
  - Default is `64`.

- :alt:`max_num_atoms`: (int) Maximum number of atoms allowed in the generated structures.
  - Default is `128`.

- :alt:`min_cell_size`: (float) Minimum cell size in Angstrom.
  - Default is `5.0`.

- :alt:`relax_struct_path`: (optional, str) Path to a folder containing DFT optimized structures.
  - Default is `''`.

- :alt:`database_path`: (str, PosixPath) Path where the final database will be saved.
  - Default is `''`.

- :alt:`rng_seed`: (optional, int) Numerical value used to fix the RNG seed. If not specified, it will be chosen randomly each run.
  - Example: `42`.

- :alt:`overwrite_db`: (optional, bool) Allow database overwrite. If false, and the database exists, the new database name will include a timestamp.
  - Default is `False`.

#### Composition - `[database.composition]`

Settings for the composition of the database.


- :alt:`size`: (int) Maximum number of structures to generate for the database.
  - Default is `7500`.

##### Ratios - `[database.composition.ratios]`

Fraction of different structure types. The sum of the fractions must be equal to 1.0.


- :alt:`bulk`: (float) Fraction of structures that will be bulk.
  - Default is `0.4`.

- :alt:`surface`: (float) Fraction of structures that will be surfaces.
  - Default is `0.6`.

- :alt:`cluster`: (optional, float) Fraction of structures that will be clusters.
  - Default is `0.0`.

#### Plot_Db - `[database.plot_db]`

Display and Export Options for the phase diagram plot.


- :alt:`show`: (optional, bool) Whether to display the database with a phase diagram after creation.
  - Default is `True`.

- :alt:`format`: (optional, str) Format for the figure.
  - Default is `'png'`.
  - Possible values are: `png`, `svg`.

##### Rc_Params - `[database.plot_db.rc_params]`

Matplotlib rcParams for the plot.


- :alt:`font.family`: (str) Font family for the phase diagram plot.
  - Default is `'monospace'`.

- :alt:`font.size`: (int) Font size for the phase diagram plot.
  - Default is `14`.

#### Show_Db_Ase - `[database.show_db_ase]`

ASE GUI display options.

:::{attention}
This section is optional.
:::


- :alt:`show`: (optional, bool) Whether to display the database using ASE GUI after creation.
  - Default is `False`.

#### Export - `[database.export]`

Export options for the database.


- :alt:`export`: (bool) Whether to export the database.
  - Default is `True`.

- :alt:`format`: (str) Export format supported by ASE (e.g., 'extxyz').
  - Default is `'extxyz'`.

- :alt:`file_path`: (str, PosixPath) Path where the exported file will be saved.
  - Default is `''`.

- :alt:`file_name`: (str) Name of the exported file.
  - Default is `'export_db_filename'`.

### Phase_Diagram - `[phase_diagram]`

Description of the phase diagram.


- :alt:`material_name`: (str) Internal name for the material in the phase diagram.
  - Default is `'default_material_name'`.

- :alt:`element_list`: (list[str]) List of elements to include in the phase diagram.
  - Example:

```python
['Cu', 'O']
```

- :alt:`base_element`: (str) Symbol of the most abundant element in the phase.
  - Example: `'Cu'`.

#### Defines a specific phase within the phase diagram. Multiple phases can be added. - `[phase_diagram.phase.XXXXX]`

This key describes settings for dynamic entries. Several entries can be added by using different key names.

The key name (`XXXXX`) is used as the reference name. **Replace XXXXX with a name of your choice.**

Example parameters for each entry:


- :alt:`name`: (str) Name to be used as reference for the phase.
  - Example: `'alpha'`.

- :alt:`cluster_element`: (optional, str) Symbol of the element defining the cluster.

- :alt:`prototype`: (str) Materials Project ID of the prototypical structure.
  - Example: `'mp-30'`.

- :alt:`offset`: (float) Fraction of composition allowed over and under the phase limits.
  - Default is `0.1`.

- :alt:`limit_max_num_structures`: (optional, int) Maximum number of structures to generate for this phase.
  - Default is `100`.

- :alt:`allow_modifications`: (optional, bool) Allow modifications (supercells, replacements, etc.) to the base structure.
  - Default is `True`.

- :alt:`use_cache`: (optional, bool) Store structures in cache to speed up generation. Can consume a lot of disk space.
  - Default is `False`.

Parameters using `composition.` prefix:

- :alt:`composition.min`: (float) Minimum composition as a fraction of the current phase element.
  - Example: `0.1`.
- :alt:`composition.max`: (float) Maximum composition as a fraction of the current phase element.
  - Example: `0.25`.

Parameters using `replacements.` prefix:

- :alt:`replacements.replace`: (optional, bool) Whether to replace specific elements. Elements in element_list will be considered for replacement and replaced by a single element species.
  - Default is `False`.
- :alt:`replacements.element_list`: (optional, list[str]) List of elements to be replaced.
  - Example:

```python
['Ti']
```
- :alt:`replacements.replace_with`: (optional, str) Element to replace with.
  - Example: `'Ir'`.

### Generation - `[generation]`

Structure generation settings.


- :alt:`generate_type`: (list[str]) Types of structures to generate.
  - Default is `['bulk', 'surface', 'cluster']`.
  - Possible values are: `bulk`, `surface`, `cluster`.

#### Bulk - `[generation.bulk]`

Bulk structure generation settings.


- :alt:`num_struct`: (int) Number of structures to generate.
  - Default is `25`.

- :alt:`num_repeat`: (int) Number of repeats for each structure.
  - Default is `5`.

- :alt:`supercell_max_idx`: (int) Maximum Miller index for the bulk supercells.
  - Default is `2`.

#### Surface - `[generation.surface]`

Surface structure generation settings.


- :alt:`min_miller_index`: (int) Minimum Miller index used to generate surface structures.
  - Default is `1`.

- :alt:`max_miller_index`: (int) Maximum Miller index used to generate surface structures.
  - Default is `3`.

- :alt:`min_slab_size_ang`: (optional, float) Minimum slab thickness in Angstrom.
  - Default is `7.0`.

- :alt:`min_vacuum_size_ang`: (float) Minimum size of the vacuum layer in Angstroms.
  - Default is `12.0`.

- :alt:`get_supercells`: (bool) Whether to generate supercells for surface structures.
  - Default is `True`.

- :alt:`fixed_layers`: (int) Number of fixed layers in the surface slab.
  - Default is `3`.

- :alt:`max_number_supercells`: (int) Maximum number of surface supercells to generate.
  - Default is `200`.

- :alt:`save_in_db`: (bool) Whether to save generated surfaces in the database.
  - Default is `True`.

- :alt:`num_replacements`: (int) Number of replacement percentages to generate for each structure.
  - Default is `20`.

- :alt:`num_repeat_replace`: (int) Number of repeats for each replacement.
  - Default is `2`.

- :alt:`frac_slabs_save`: (optional, float) Fraction of slabs to save after generation.
  - Default is `0.1`.

- :alt:`frac_supercells_save`: (optional, float) Fraction of unreplaced supercells to save after generation.
  - Default is `0.1`.

- :alt:`max_slab_num`: (int) Maximum number of slabs to gather from the slab generation.
  - Default is `15`.

- :alt:`n_workers`: (optional, int) Maximum number of workers for parallel processing.

### Deformation - `[deformation]`

Lattice deformation settings.

:::{attention}
This section is optional.
:::


- :alt:`lattice_frac_deform_max`: (float) Maximum deformation value as a percentage of the lattice side length.
  - Default is `0.05`.

- :alt:`lattice_frac_deform_min`: (float) Minimum deformation value as a percentage of the lattice side length.
  - Default is `0.01`.

- :alt:`num_repeats`: (int) Number of repeats for each structure with random deformations.
  - Default is `5`.

- :alt:`limit_max_num_deformations`: (int) Maximum number of lattice deformations to generate.
  - Default is `100`.

### Perturbation - `[perturbation]`

Perturbation settings.

:::{attention}
This section is optional.
:::


- :alt:`filter_struct_types`: (list[str]) Types of structures to which the perturbation will be applied.
  - Default is `['bulk', 'surface']`.
  - Possible values are: `bulk`, `surface`, `cluster`.

- :alt:`limit_max_num_perturbs`: (int) Maximum number of perturbations to generate.
  - Default is `100`.

- :alt:`num_repeats`: (int) Number of repeats for each structure with random perturbations.
  - Default is `1`.

- :alt:`perturbation_ang`: (optional, float) Perturbation magnitude in Angstrom.
  - Default is `0.04`.

### Adsorbates - `[adsorbates]`

Adsorbate placement settings.

:::{attention}
This section is optional.
:::


- :alt:`filter_struct_types`: (optional, list[str]) Types of structures to which adsorbates will be added.
  - Default is `['surface']`.
  - Possible values are: `bulk`, `surface`, `cluster`.

- :alt:`limit_max_num_perturbs`: (optional, int) Maximum number of structures with adsorbates to generate.
  - Default is `100`.

- :alt:`num_repeats`: (int) Number of repeats for each structure.
  - Default is `1`.

- :alt:`adsorbate_species`: (list[str]) List of adsorbate species to consider.
  - Example:

```python
['H', 'H2O']
```

### Struct_Filters - `[struct_filters]`

Settings for filtering out incorrect structures.

:::{attention}
This section is optional.
:::


#### No_Neighbors - `[struct_filters.no_neighbors]`

Filter for structures with atoms that have no neighbors.


- :alt:`cov_rad_multiplier`: (optional, float) Multiplier applied to the covalent radii to be used as cutoff radius for the neighbor check.
  - Default is `1.2`.

#### Layer_Distance - `[struct_filters.layer_distance]`

Filter for layer distances in surface slabs.

:::{attention}
This section is optional.
:::


- :alt:`max_layer_distance_ang`: (optional, float) Maximum accepted distance between layers in Angstrom.
  - Default is `4.0`.

#### Duplicate_Slabs - `[struct_filters.duplicate_slabs]`

Filter for duplicate slabs.

:::{attention}
This section is optional.
:::


- :alt:`tolerance`: (optional, float) Tolerance for the duplicate slabs filter.
  - Default is `0.2`.

### Vacancies - `[vacancies]`

Settings for vacancy generation.


- :alt:`filter_struct_types`: (list[str]) Types of structures to which vacancies will be applied.
  - Default is `['bulk', 'surface']`.
  - Possible values are: `bulk`, `surface`, `cluster`.

- :alt:`limit_max_num_vacancies`: (optional, int) Maximum number of structures with vacancies to generate.
  - Default is `400`.

- :alt:`num_repeats`: (int) Number of repeats for each structure with different random vacancies.
  - Default is `3`.

- :alt:`max_vacancy_percentage`: (float) Maximum vacancies to generate as a percentage of the total number of atoms.
  - Default is `0.75`.

- :alt:`min_vacancy_percentage`: (float) Minimum vacancies to generate as a percentage of the total number of atoms.
  - Default is `0.025`.

- :alt:`element_list`: (list[str]) List of elements to consider for the vacancies.
  - Example:

```python
['O']
```

### Targeted_Modification - `[targeted_modification]`

Settings for targeted structural modifications.

:::{attention}
This section is optional.
:::


#### Central_Atom_Octahedral - `[targeted_modification.central_atom_octahedral]`

Apply perturbations to the central atom in octahedral sites.

:::{attention}
This section is optional.
:::


- :alt:`filter_phases`: (optional, list[str]) Only apply the modification to the following phases.
  - Example:

```python
['rutile', 'original_IrO2']
```

- :alt:`filter_struct_types`: (optional, list[str]) Types of structures to which the modification will be applied.
  - Default is `['bulk', 'surface']`.
  - Possible values are: `bulk`, `surface`, `cluster`.

- :alt:`central_element`: (optional, str) Symbol of the central element of the octahedral site.
  - Example: `'Ir'`.

- :alt:`num_repeats`: (optional, int) Number of repeats for each structure with different perturbations.
  - Default is `3`.

- :alt:`limit_max_num_modifications`: (optional, int) Maximum number of modified structures to generate.
  - Default is `200`.

- :alt:`max_perturbation_ang`: (optional, float) Maximum perturbation movement of the central atom in Angstrom.
  - Default is `0.2`.

### Concave_Hull - `[concave_hull]`

Settings for descriptors and concave hull generation.

:::{attention}
This section is optional.
:::


- :alt:`gen_concave_hull`: (optional, bool) Whether to generate the concave hull of the descriptors for all structures in the database.
  - Default is `False`.

- :alt:`descriptor`: (optional, str) Descriptor to use for the concave hull generation.
  - Default is `'SOAP'`.
  - Possible values are: `SOAP`, `MACE`.

- :alt:`dim_reduction`: (optional, str) Dimensionality reduction method for the concave hull generation.
  - Default is `'autoencoder'`.
  - Possible values are: `PCA`, `autoencoder`.

- :alt:`plot_filename`: (optional, str) Filename for the figure displaying the concave hull.
  - Default is `'descriptors_concave_hull.png'`.
