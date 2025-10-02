# Input specification

The input format of MDB is [TOML](https://toml.io/en/). The syntax from TOML is unchanged. The available parameters are different depending on the selected tool.

Users are advised to use the `mdb_gen_configuration_file` utility to generate a template file which can be customized. However, the configuration files can be created from scratch using the sections from below and the appropiate TOML syntax as in the following example:

```toml
[database]
database_name = 'test'
...


[database.plot_db]
show = true

...

[generation]
generate_type = ['bulk', 'surface', 'cluster']

```

Please, check the tool's corresponding section below to learn more about all the available options:

- [Database Generation](./input_database_generation.md)
- [DFT Batch](./input_dft.md)
- [Active Learning](./input_active_learning.md)
