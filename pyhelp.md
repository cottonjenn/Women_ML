How to create a virtual environment:
   to create venv with latest version: `python3 -m venv .venv` <-- .venv or venv are standard, but you can name it other things
   to create vene with old version: `python3.12 -m venv .venv` <-- to get something like tensorflow

Mac -->
   - to activate: `source .venv/bin/activate` --> when done run `deactivate` or close IDE
   - upgrade pip: `pip install --upgrade pip`
   - to download packages: `pip install <library_name>`
      - packages: `pip install numpy selenium pandas matplotlib seaborn scikit-learn statsmodels openpyxl pyarrow jupyter ipykernel plotly scipy streamlit tensorflow tensorflow.keras` <-- data science / ML packages (NOT comma-delimited)

Windows -->
   - To activate `.\.venv\Scripts\Activate` --> when done run `deactivate` or close IDE
   - upgrade pip: `python.exe -m pip install --upgrade pip`
   - packages: `pip install numpy, selenium, pandas, matplotlib, seaborn, scikit-learn, statsmodels, openpyxl, pyarrow, jupyter, ipykernel, plotly, scipy, streamlit` <-- (comma-delimited)
   - to see full list of sub-libraries etc go to ".venv\Lib\site-packages" or requirements.txt

`python --version` == Python 3.14.2
`pip --version` == pip 26.0.1

to see all current notebooks run: `pip freeze > requirements.txt` <-- creates a requirements.txt file
to install same versions somewhere else: `pip install -r requirements.txt`
