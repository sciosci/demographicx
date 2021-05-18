# `demographicx`: A Python package for estimating gender and ethnicity using deep learning transformers

## Summary

This package was used in the following publication to estimate gender 
and ethnicity of scientists

**Acuna, Daniel E.**, **Liang, Lizhen** (2021), *Are AI ethics conferences different and more diverse compared to traditional
computer science conferences?* In Proceedings of the AAAI/ACM Conference on AI, Ethics, and
Society https://doi.org/10.1145/3461702.3462616

## Installation

```bash
$ pip install git+https://github.com/sciosci/demographicx
```

## Example

```python
from demographicx import GenderEstimator

gender_estimator = GenderEstimator()
gender_estimator.predict('Daniel')

{'male': 0.9886190672823015,
 'unknown': 0.011367974526753396,
 'female': 1.2958190945360288e-05}
```

```python
from demographicx import EthnicityEstimator
ethnicity_estimator = EthnicityEstimator()
ethnicity_estimator.predict('lizhen liang')

{'black': 2.1461191541442314e-06,
     'hispanic': 4.0070474029127346e-05,
     'white': 0.0002176521167431309,
     'asian': 0.999740131290074}

ethnicity_estimator.predict('daniel wegmann')

{'black': 4.120965729769303e-06,
     'hispanic': 0.0023926903023342287,
     'white': 0.9963380370701861,
     'asian': 0.00126515166175015}
```

## Dependencies

* torch
* transformers
* numpy
* scipy

## License

See `LICENSE`
