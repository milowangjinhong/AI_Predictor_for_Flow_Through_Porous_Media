# AI Predictor for Flow Through Porous Media

This is a repository created for data-driven prediction of flow through porous media. 

Three architectures are implemented and tested for the application:
- Convolutional Auto-Encoder
- U-NET
- Fourier Neural Operator

---

## Installation

### Clone the repo
```
git clone https://github.com/milowangjinhong/AI_Predictor_for_Flow_Through_Porous_Media.git 
```

### Install dependencies
```
cd docs
pip install -r requirements.txt
```
---

## Usage

### Data Structure
Input with dimension (1, dim1, dim2) with 
- input[0, :, :] as gamma

Output with dimension (3, dim1, dim2) with 
- out[0, :, :] as pressure
- out[1, :, :] as u
- out[2, :, :] as v

### Training

The training code with [submission script](training_scripts/submit_all.pbs) is included in [**/training_script/**](./training_script/).

The customised training code is included in [**/training_scripts/training.py**](./training_scripts/training.py)

### Post-Processing

The example post-processing and visualisation is included in this notebook: [**post-processing.ipynb**](./post-processing.ipynb).
