# 地方财政多指标时序预测模型（TabNet Pipeline）  
**Local Fiscal Multi-Target Time-Series Forecasting (TabNet Pipeline)**  

---

## 项目简介 / Project Overview 

**特别注意** ：由于涉及地方政府的具体财政情况以及使用者的不同需要，为不影响结果的准确性，模型的训练和测试数据文件（train_data.xlsx、test_data.xlsx）**默认无内容**，需使用者根据自身需求填充数据后才能确保模型结果准确，文档内无数据则程序报错。  
Important Note: Due to the involvement of local governments' specific fiscal situations and the different needs of users, to avoid affecting the accuracy of results, the model's training and test data files (train_data.xlsx, test_data.xlsx) are empty by default. Users must fill in the data according to their own needs to ensure the accuracy of the model results. The program will report an error if there is no data in the files.

本项目基于 **PyTorch TabNet** 框架，构建了一个用于预测地方政府财政多维指标的时序预测模型。  
The project implements a **multi-target time-series forecasting model** based on **PyTorch TabNet**, designed to predict multiple fiscal indicators for local governments.  

模型可自动完成数据清洗、缺失值处理、特征编码、模型训练与验证、未来三年预测及结果输出。  
It automatically performs data cleaning, missing value imputation, categorical encoding, model training & validation, and 3-year ahead forecasting with structured Excel output.

---

## 项目结构 / Project Structure  

```
├── TabNet_Government_Fiscal_Time_Series_Forecasting_Model.py   # 主程序 / Main script
├── train_data.xlsx                            # 训练数据 / Training data
├── test_data.xlsx                             # 预测输入数据 / Input data for prediction
└── output_data.xlsx                           # 预测结果输出 / Model output file
```

---

## 模型原理 / Model Concept  

**TabNet** 是一种基于注意力机制的深度学习模型，能够自动学习特征重要性、选择性关注输入特征，并保持较强的可解释性。  
**TabNet** is an attention-based deep learning model for tabular data that dynamically selects features and provides interpretability while maintaining high predictive performance.

模型的核心特点 / Key features:
- 多目标预测（同时预测多个财政指标）  
  Multi-output regression for fiscal indicators  
- 自动特征选择与编码  
  Automatic feature selection and categorical encoding  
- 按年份划分验证集（最近一年为验证集）  
  Validation split by year (latest year reserved for validation)  
- 自动生成未来三年的逐步预测  
  Auto-iterative forecasting for the next three years  

---

## 主要预测指标 / Key Target Indicators  

| 类别 Category | 指标名称 Indicator |
|---------------|--------------------|
| 财政规模 Fiscal Scale | 政府年财政收入 / Government Annual Revenue<br>政府年财政支出 / Government Annual Expenditure |
| 收入结构 Revenue Structure | 土地出让收入占比 / Land Transfer Revenue Share<br>第一、二、三产业财政收入占比 / Primary, Secondary, Tertiary Industry Share |
| 支出结构 Expenditure Structure | 公共预算支出占比 / General Budget Share<br>社保基金支出占比 / Social Security Fund Share<br>政府性基金支出占比 / Government Fund Share<br>国有资本经营支出占比 / State-Owned Capital Share |

---

## 程序流程 / Workflow  

1️⃣ **加载数据 / Load Data**  
从 Excel 文件加载训练与预测数据。  
Load training and prediction datasets from Excel files.  

2️⃣ **数据预处理 / Preprocessing**  
- 自动识别数值与类别列
  Automatically identify numerical and categorical columns  
- 丢弃缺失值超过 50% 的列
  Drop columns with missing values exceeding 50%  
- 缺失值填补（数值→中位数，类别→众数） 
  Missing value imputation (numerical → median, categorical → mode) 

3️⃣ **模型训练与验证 / Model Training & Validation**  
- 使用最近一年作为验证集
  Use the most recent year as the validation set  
- 输出 MAE、RMSE、R² 指标
  Output metrics of MAE, RMSE, and R²  

4️⃣ **预测未来三年 / Three-Year Forecasting**  
- 逐城市滚动预测未来 3 年财政指标
  Conduct rolling forecast of fiscal indicators for the next 3 years on a city-by-city basis  
- 结果自动生成结构化表格  
  Automatically generate structured tables for the results
---


## 数据格式要求 / Data Requirements  

### 训练数据 / `train_data.xlsx`  
必须包含以下列：
The following columns must be included:  
- 城市编号   
  City ID  
- 年份   
  Year  
- 城市类型、规模、是否省会、是否沿海等分类属性  
  Categorical attributes such as city type, size, whether it is a provincial capital, and whether it is coastal
- 所有财政目标列（参考上表）
  All fiscal target columns (refer to the above table)  

### 预测输入 / `test_data.xlsx`  
-包含需预测的城市与年份信息。   
 It contains information of the cities and years to be predicted. 
-特征可缺失，程序将自动填补。    
 Features may be missing, and the program will automatically impute them.

---

## 作者与用途 / Author & Purpose  

该脚本由研究者开发，用于构建地方政府财政健康与可持续性分析模型，可作为 **地方财政健康指数 (Local Fiscal Health Index)** 的预测核心模块。  
Developed for research on **Local Fiscal Health Index**, this script serves as the forecasting engine for assessing fiscal sustainability and financial balance of local governments.  

---
