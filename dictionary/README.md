# Data dictionaries

What every table and column of the benchmark databases means, in business terms: the
semantic layer the analyst reads before it writes SQL.

- `financial.yaml`: the Czech bank database, written by hand. English names, meanings,
  units, translations of every Czech code present in the data, how values were encoded,
  known quirks and join paths.
- `<database>.yaml` for the other ten BIRD databases: converted as-is from the description
  files BIRD ships (`scripts/14_convert_descriptions.py`). Columns BIRD leaves undescribed are
  marked `status: missing`.
- `_snapshot/<database>.json`: each database's tables, columns, types and row counts, and for
  the Czech bank also every value of its low-cardinality text columns
  (`scripts/13_snapshot_schema.py`). The tests check the dictionaries against these files, and
  the files against the database.

## Format

```yaml
database: financial
origin: hand-written            # or bird-description
name: ...
description: ...
quirks: [...]                   # database-wide
tables:
  <table>:
    name: ...                   # English name
    description: ...
    grain: one row per ...
    primary_key: [...]
    joins:
      - {to: <table>, on: <condition>, cardinality: many-to-one, meaning: ...}
    columns:
      <column>:
        name: ...               # English name
        kind: identifier | code | label | quantity | date | text
        description: ...
        unit: CZK               # quantities
        codes: {VALUE: meaning} # code columns: every value present in the data
        null_meaning: ...       # when the column has NULLs
        encoding: ...           # how the value was or is encoded
        references: <table>.<column>
        quirks: [...]
```

A `code` holds values with a meaning to translate (`PRIJEM`: credit); a `label` is a name or an
opaque code that is its own meaning (a district's name, an anonymized bank code). Converted
dictionaries keep BIRD's fields: `bird_name`, `name`, `description`, `data_format`,
`value_description`.

## Licence and sources

The files in this folder describe the BIRD databases and are derived from BIRD's database
description files and data: Li, J., et al. (2023), *Can LLM Already Serve as a Database
Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs*, NeurIPS Datasets and
Benchmarks. They are licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), as BIRD is. The Czech bank
dictionary also draws on Berka, P. (1999), *Guide to the Financial Data Set*, PKDD'99 Discovery
Challenge. The code in this repository is under the MIT License.
