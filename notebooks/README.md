# 📓 Jupyter Notebooks

## Available Notebooks

### 1. `01_EDA_Telecom_Churn.ipynb`
**Purpose**: Exploratory Data Analysis

**Contents**:
- Load Telecom dataset
- Data overview and statistics
- Churn analysis
- Demographics analysis
- Service analysis
- Tenure and charges analysis
- Correlation analysis
- Key insights

**When to use**: Before running the pipeline, to understand the data

---

### 2. `02_Model_Experimentation.ipynb`
**Purpose**: Model testing and comparison

**Contents**:
- Load processed features
- Train-test split
- Logistic Regression
- Random Forest
- XGBoost
- Model comparison
- Feature importance

**When to use**: After preprocessing, to experiment with models

---

## How to Use

### 1. Install Jupyter
```bash
pip install jupyter
```

### 2. Launch Jupyter
```bash
jupyter notebook
```

### 3. Open Notebooks
Navigate to `notebooks/` folder and open any `.ipynb` file

---

## Prerequisites

### For EDA Notebook:
- Download Telecom dataset
- Place in `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

### For Model Experimentation:
- Run preprocessing first:
```bash
python src/data/data_loader.py
python src/data/preprocessing.py
```

---

## Notebook Workflow

```
1. EDA Notebook → Understand data
2. Run preprocessing → Create features
3. Model Experimentation → Test models
4. Run main.py → Complete pipeline
5. Launch dashboard → View results
```

---

## Tips

- Run cells sequentially (Shift+Enter)
- Restart kernel if needed (Kernel → Restart)
- Save frequently (Ctrl+S)
- Export as HTML for sharing (File → Download as → HTML)

---

## Why Use Notebooks?

✓ **Interactive exploration** - See results immediately
✓ **Visualization** - Charts and graphs inline
✓ **Documentation** - Mix code and markdown
✓ **Experimentation** - Try different approaches
✓ **Portfolio** - Show your analysis process

---

**Happy exploring! 📊**
