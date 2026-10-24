# Data

The experiments use the [CPSC 2018](http://2018.icbeb.org/Challenge.html) 12-lead ECG dataset
(6,877 records, 500 Hz, 6–60 s). The **pre-processed** version (`CPSC_final.zip`, 2.3 GB), including the
expert masks for PAC/PVC, can be downloaded here:

**https://drive.google.com/file/d/1Eaj9kxT0G6OizEOG8bxIx91DEpP_ww97/view?usp=sharing**

Extract it so that the layout matches `configs/default.yaml`:

```
data/CPSC_final/
├── meta_data.csv            # columns: record, label (0–8, see eagle.CLASS_NAMES)
├── <record>.npz             # key "x": float array of shape (12, 2500)
└── mask/
    └── <record>_mask.npz    # key "mask": binary array of shape (2500,)
```

## Pre-processing applied

1. Resample from 500 Hz to 250 Hz.
2. Split each record into non-overlapping 10 s windows (2,500 samples), zero-padding the last window.
3. Smooth with a Savitzky–Golay filter (window 25, order 3).
4. Min–max normalise to [0, 1].

The thesis reports 13,406 windows. The released `CPSC_final.zip` lists 12,797 windows in `meta_data.csv`
(10,251 train / 2,546 validation per fold with the default config) and contains 487 expert masks
(280 PAC, 201 PVC, 2 on windows labelled Normal; 4 masks have no matching row in `meta_data.csv`). The filtering helpers are in
`eagle/data/preprocess.py`.

## Expert masks

Masks mark the region that explains the diagnosis: the abnormal P wave / PR interval before a
premature beat for **PAC**, and the wide QRS complex for **PVC**. They were drawn with
`tools/labelling_app.py`; new masks are written to `data/CPSC_final/mask/`.
