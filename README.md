# radarwow-verify

**radar-wow: Verification.** Gauge matching, scores (bias, RMSE, CSI, FSS), stratified by band, range and rain type.

Part of [radar-wow](https://github.com/radar-wow), a set of repos that together build a radar QPE system that works for S, C and X band radars.
Every algorithm is one class implementing one paper, with the same interface: it takes an xradar sweep or volume with canonical moment names, declares the bands and moments it needs, and records provenance.

## Sub-components

| sub-component | algorithm | source | DOI | status |
|---|---|---|---|---|
| gauge matching | `match_gauges` | radar-gauge pairs in space and time |  | planned |
| continuous + categorical scores | `scores` | standard verification metrics |  | planned |

## Install

```bash
pip install git+https://github.com/radar-wow/radarwow-core git+https://github.com/radar-wow/radarwow-verify
```

## Usage

```python
from radarwow.core import load_sample
from radarwow.verify import *  # see table above

vol = load_sample('arm_csapr2_c')        # or open_volume('my_file')
# out = SomeAlgorithm()(vol)            # band read from the file, or band='S'|'C'|'X'
```

## Status legend

*implemented + notebook*: code, tests, executed notebook. *planned*: interface fixed, code to write. *stub*: class exists, raises `NotImplementedError`. *to review*: paper to read before deciding.

## License

MIT. Data used in the notebooks comes from public archives (NEXRAD on AWS, ARM, IMD via open-radar-data) under their own terms.
