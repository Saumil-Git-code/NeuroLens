# NeuroLens

NeuroLens is a Django application for managing brain MRI scans and reviewing
model-generated tumor classifications. It includes a PyTorch inference
pipeline, Grad-CAM heatmaps, patient and scan workflows, and a model-training
notebook. This project was made to assist the doctors in making more accurate decisions while evaluating Brain MRIs of patients.

> **Research/demo software only.** This project is not a medical device and
> must not be used to diagnose, treat, or make decisions about patients.
> Never use real patient information in a demo or development environment.

## Requirements

- Python 3.10 or newer
- pip
- A supported PyTorch installation (CPU works; CUDA is optional)

The dependencies in `requirements.txt` include PyTorch and may take
considerable time and disk space to install. If you need CUDA, follow the
installation instructions on the official [PyTorch website](https://pytorch.org/get-started/locally/)
for your operating system and GPU before installing the remaining requirements.

## Local development (Windows PowerShell)

From the project directory:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

Open `http://127.0.0.1:8000/`. Django uses a temporary development secret key
when `DJANGO_SECRET_KEY` is unset and `DJANGO_DEBUG` is enabled. For a stable
local key, set it in your PowerShell session before running Django:

```powershell
$env:DJANGO_SECRET_KEY = python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

The development defaults use SQLite. The database, uploaded scans, generated
heatmaps, Python environments, and local datasets are excluded from Git.

## Model files and datasets

Model checkpoints are intentionally not included: the VGG16 checkpoint is
approximately 512 MiB, beyond GitHub's regular per-file limit. Put a compatible
checkpoint at `ml/weights/best_model.pth` (or `ml/weights/vgg16.pth`) to enable
trained-model inference. Without one, the application uses its deterministic
mock predictor.

The training notebook expects an MRI dataset with `Training/` and `Testing/`
folders, and the demo seeding command can use local sample images. Dataset
files are not included. Obtain data from a source whose terms permit your
intended use, and do not commit identifiable or otherwise sensitive medical
images.

## Security and deployment

The checked-in Django settings are development-only: they enable debug mode,
allow every host, and contain a public development secret key. Do not deploy
this configuration or use it with real patient information. Before deployment,
move the secret key and host list into environment-based configuration, disable
debug mode, and complete Django's deployment checklist, including storage,
database, HTTPS, and access controls.

## Tests and checks

```powershell
python manage.py check
python manage.py test
```
## Future Scops
1. This project is limited to brain only. Its scope can be expanded to cater a wide variety of diseases in other body parts also.
2. This project uses VGG16 and also compares it with ResNet-18 which can be upgraded to ViT to capture more complex MRI images

