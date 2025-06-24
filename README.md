<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;">
                Temporal Score Analysis for Understanding and Correcting Diffusion Artifacts</h1>
<p align='center' style="text-align:center;font-size:1.25em;">
    <a href="https://yucao16.github.io" target="_blank" style="text-decoration: none;">Yu Cao<sup>*</sup></a>&nbsp;,&nbsp;
    <a href="https://zengqunzhao.github.io" target="_blank" style="text-decoration: none;">Zengqun Zhao</a>&nbsp;,&nbsp;
    <a href="https://www.eecs.qmul.ac.uk/~ioannisp/" target="_blank" style="text-decoration: none;">Ioannis Patras</a>&nbsp;,&nbsp;
    <a href="https://www.eecs.qmul.ac.uk/~sgg/" target="_blank" style="text-decoration: none;">Shaogang Gong<sup>&#8224</sup></a></br>
</p>
<p align='center' style="text-align:center;font-size:1.25em;">
Queen Mary University of London<br/>
</p>

## Overview

ASCED provides method for detecting and correcting artifacts in diffusion-generated images through temporal score analysis. The repository includes two main demonstration notebooks:
- `notebooks/detection_demo.ipynb`: Demonstrates artifact detection in generated images
- `notebooks/correction_demo.ipynb`: Demonstrates artifact correction in diffusion models

## Requirements

Before running the notebooks, you need to download:

1. **Model weights**: Download the pre-trained diffusion model weights from [yandex-research/ddpm-segmentation](https://github.com/yandex-research/ddpm-segmentation) and place them in the `checkpoints/ddpm/` directory
2. **Pickle files and seed data**: Download the following from [HERE](https://drive.google.com/drive/folders/15Ns41G_GXnsrMOq9C6F30_VqdvRjEf2v?usp=sharing):
   - `normalized_score_dict.pkl` and place it in `experiments/`
   - Seed files (`noise_*.npy`) and place them in `datasets/noise/`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YuCao16/ASCED.git
cd ASCED
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Artifact Detection
Open and run `notebooks/detection_demo.ipynb` to see demonstrations of:
- Temporal difference analysis
- Artifact mask generation
- Acceleration comparison between artifact and non-artifact regions

### Artifact Correction
Open and run `notebooks/correction_demo.ipynb` to see demonstrations of:
- DDIM sampling with artifact correction
- Visual comparison of corrected outputs

## Citation

If you find this work useful, please cite:
```bibtex
@inproceedings{cao2025temporal,
  title={Temporal Score Analysis for Understanding and Correcting Diffusion Artifacts},
  author={Cao, Yu and Zhao, Zengqun and Patras, Ioannis and Gong, Shaogang},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={7707--7716},
  year={2025}
}
```

## Contributing / Issues

Please feel free to open an issue on GitHub if you encounter problems or have suggestions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Parts of this project page were adopted from the [Nerfies](https://nerfies.github.io/) page.
