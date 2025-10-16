# AITW stiffness surrogate model

**NOTES** Requires python3.10 for the current pinned version of tensorflow.


## Installaton

### OPTIONAL Create and initialize a clean virtual environment

```bash
python3.10 -m venv "surrogate-virtual-env"
source surrogate-virtual-env/bin/activate
```

### Install dependencies

```bash
cd <PATH to folder with pyproject.toml>
pip install .[plotting]
```

## Usage

### CLI

The code makes avaialble a CLI utility to run the inference given an input JSON file.

```bash
aitw-stiffness example-input/input.json output.json 0.05 0.95
```

Addjust the parameters as needed (Use `--help` for more information).

### Parameters

Parameter | Description | Type
--- | --- | ---
`<input-path>` | File path to the input file containing a properly formatted JSON file. See fromat below. | String
`<output-path>` | File path for the stiffness estimate JSON file. Example output shown below. | String
`<quantile-lower-bound>` | Lower bound percentage of the uncertainty estimate. | Float [ 0, 1]
`<quantile-upper-bound>` | Upper bound percentage of the uncertainty estimate | Float [ 0, 1]

## Input parameter values and ranges

The surrogate has been trained with physical parameters ranging between the values below. Fiber orientation distribution parameter takes one of the two values below. \
An error will be thrown if the input parameter values do not coincide with the limits.

| Parameter   | Possible value |
| ------------------------------- | ----- |
| fiberOrientationDistribution    | "vonMises", "vonMisesFisher" |

| Parameter   | Minimum     | Maximum        |
| ------------------------------- | ----- | -------- |
| fiberAspectRatio                |   1.2 |    20000 |
| orientationConcentration        | 1e-10 |      100 |
| microfibrilAngle                |     0 |       45 |
| lumenPorosity                   |     0 |     0.99 |
| fiberVolumeFraction             |     0 |        1 |
| celluloseContent                |  0.05 |      0.9 |
| hemicelluloseContent            |  0.01 |      0.5 |
| ligninContent                   |  0.01 |      0.6 |
| pectinContent                   |     0 |     0.15 |
| extractivesContent              |     0 |     0.15 |
| ashContent                      |     0 |      0.2 |
| celluloseCrystallinity          |   0.2 |      1.0 |
| matrixYoungsModulus             |     0 |      100 |
| matrixPoissonRatio              |     0 |      0.5 |
| airPorosity                     |     0 |      0.5 |
| tangentialInterfaceCompliance   | 1e-10 | 100000.0 |
| longitudinalInterfaceCompliance | 1e-10 | 100000.0 |

## Output file example

This results in the following output file format in "./output.json":

```json
{
  "E_T": {
    "median": 3.550992727279663,
    "lowerBound": 3.2585065364837646,
    "upperBound": 3.869732141494751
  },
  "E_L": {
    "median": 1.9295220375061035,
    "lowerBound": 1.6106904745101929,
    "upperBound": 2.3114655017852783
  },
  "nu_T": {
    "median": 0.2790219783782959,
    "lowerBound": 0.19306382536888123,
    "upperBound": 0.36498013138771057
  },
  "nu_LT": {
    "median": 0.19985055923461914,
    "lowerBound": 0.15456369519233704,
    "upperBound": 0.24513742327690125
  },
  "mu_LT": {
    "median": 1.482556939125061,
    "lowerBound": 1.3604427576065063,
    "upperBound": 1.61563241481781
  },
  "quantiles": [
    0.05,
    0.95
  ]
}
```

