# NumericEssentials

Solving real problems with classic numerical methods in Python

## Setup Instructions

### Requirements

* Python 3.8+
* pip
* VS Code or Jupyter Notebook

Follow the steps below to run the project locally using Python and Jupyter Notebook.

### 1. Clone the repository

```bash
git clone https://github.com/lfelipediniz/NumericEssentials.git
cd NumericEssentials
````

### 2. Create and activate a virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate.bat    # for CMD
venv\Scripts\Activate.ps1    # for PowerShell
```

> If activation fails on PowerShell, run:
>
> ```powershell
> Set-ExecutionPolicy RemoteSigned
> ```

#### Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook

You can open the notebook using the VS Code interface or via terminal:

```bash
jupyter notebook
```

Then, open the file:

```
NumericEssentials.ipynb
```

## Topics Covered

* LU Decomposition (banded and full)
* Root finding (Bisection & Newton methods)
* SVD and image compression
* Interpolation (Lagrange, Newton, spline)
* Least squares curve fitting





