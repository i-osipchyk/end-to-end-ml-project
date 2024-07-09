# end-to-end-ml-project

## Workflows

1. Update config.yaml
2. Update schema.yaml
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update components
7. Update the pipeline
8. Update the main.py
9. Update the app.py

## Steps to run the project

### Step 1. Clone the repository

```bash
git clone https://github.com/i-osipchyk/end-to-end-ml-project
```

### Step 2. Create virtual environment

```bash
python3 -m venv venv for MacOS/Linux or python -m venv venv for Windows
```

### Step 3. Activate virtual environment

```bash
source venv/bin/activate for MacOS/Linux or venv/Scripts/activate for Windows
```

### Step 4. Install requirements

```bash
pip install -r requirements.txt
```

### Step 5. Execute the code

```bash
python3 app.py or python app.py for Windows
```


## Mlflow

[Documentation](https://mlflow.org/docs/latest/index.html)

## DagsHub

[dagshub](https://dagshub.com/)

To run Mlflow on Dagshub server, import these environmental variables:
os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/ivan.osipchyk.work/end-to-end-ml-project.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]=YOUR_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"]=YOUR_TOKEN

Note that this Mlflow dashboard will be accessible only with owner credentials.

After exporting variables, visit this link to see the dashboard:
https://dagshub.com/ivan.osipchyk.work/end-to-end-ml-project.mlflow


