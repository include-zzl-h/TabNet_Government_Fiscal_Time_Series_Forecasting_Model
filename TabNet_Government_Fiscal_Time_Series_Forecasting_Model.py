import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from pytorch_tabnet.tab_model import TabNetRegressor

TRAIN_FILE = "train_data.xlsx"
PRED_INPUT_FILE = "test_data.xlsx"
OUTPUT_FILE = "output_data.xlsx"

CITY_COL = "城市编号"
YEAR_COL = "年份"

# 目标列
TARGET_COLS = [
    "政府年财政收入",
    "土地出让收入占地方一般公共预算收入的比例",
    "政府年财政支出",
    "第一产业所占财政总收入比例",
    "第二产业所占财政总收入比例",
    "第三产业所占财政总收入比例",
    "公共预算支出占财政总支出比例",
    "社保基金支出占财政总支出比例",
    "政府性基金支出占财政总支出比例",
    "国有资本经营支出占财政总支出比例",
]

# 典型分类列
CATEGORICAL_COLS = [
    "城市规模",
    "城市类型",
    "功能属性划分",
    "是否为省会地区",
    "是否为沿海城市",
    "是否有自贸区",
]

ID_LIKE_COLS = ["城市名称"]

# 最近一年作为验证集
N_VALID_YEARS = 1

# TabNet参数（默认起点，按需要可进行调整）
TABNET_PARAMS = dict(
    n_d=16,
    n_a=16,
    n_steps=5,
    gamma=1.5,
    n_independent=2,
    n_shared=2,
    lambda_sparse=1e-4,
    seed=42,
    verbose=10,
)

# 训练超参数（适用于样本量较小的情况，可根据样本量进行调整）
FIT_PARAMS = dict(max_epochs=100, patience=20, batch_size=128, virtual_batch_size=16)


# 浮点型转换
def cols_convert_float(df, exclude_cols):
    for c in df.columns:
        if c in exclude_cols:
            continue
        if df[c].dtype == object:
            try:
                df[c] = pd.to_numeric(df[c], errors="ignore")
            except Exception:
                pass
    return df


# 丢弃缺失值占比大于50%的列
def drop_high_missing_features(df, feature_cols, threshold=0.5):
    to_drop = []
    for c in feature_cols:
        miss_ratio = df[c].isna().mean()
        if miss_ratio > threshold:
            to_drop.append(c)
    if to_drop:
        print(f"丢弃缺失>50%的特征列：{to_drop}")
        feature_cols = [c for c in feature_cols if c not in to_drop]
    return feature_cols


# 划分验证集
def split_val(df, year_col, n_valid_years):
    years_sorted = sorted(df[year_col].dropna().unique().tolist())
    split_boundary = years_sorted[-n_valid_years]
    train_idx = df[year_col] < split_boundary
    valid_idx = df[year_col] >= split_boundary
    return train_idx, valid_idx


def mae_rmse_r2(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred, multioutput="uniform_average")
    rmse = mean_squared_error(y_true, y_pred, multioutput="uniform_average") ** 0.5
    r2 = r2_score(y_true, y_pred, multioutput="variance_weighted")
    return mae, rmse, r2


# 加载数据
def load_data():
    train_df = pd.read_excel(TRAIN_FILE)
    pred_df = pd.read_excel(PRED_INPUT_FILE)
    # 类型清洗
    train_df = cols_convert_float(
        train_df, exclude_cols=[CITY_COL, YEAR_COL] + CATEGORICAL_COLS + ID_LIKE_COLS
    )
    pred_df = cols_convert_float(
        pred_df, exclude_cols=[CITY_COL, YEAR_COL] + CATEGORICAL_COLS + ID_LIKE_COLS
    )
    # 关键列存在性检查
    for col in [CITY_COL, YEAR_COL]:
        if col not in train_df.columns:
            raise KeyError(f"训练数据缺少关键列：{col}")
        if col not in pred_df.columns:
            raise KeyError(f"输入预测值缺少关键列：{col}")
    missing_targets = [t for t in TARGET_COLS if t not in train_df.columns]
    if missing_targets:
        raise KeyError(f"训练数据缺少目标列：{missing_targets}")
    return train_df, pred_df


def prepare_features(train_df, pred_df):
    feature_cols = [c for c in train_df.columns if c not in TARGET_COLS]
    for c in ID_LIKE_COLS:
        if c in feature_cols:
            feature_cols.remove(c)
    feature_cols= drop_high_missing_features(train_df, feature_cols, threshold=0.5)
    X_train_full = train_df[feature_cols].copy()
    y_train_full = train_df[TARGET_COLS].copy()
    num_cols = []
    cat_cols = []
    for c in feature_cols:
        if c in [CITY_COL, YEAR_COL]:
            num_cols.append(c)
        elif c in CATEGORICAL_COLS:
            cat_cols.append(c)
        else:
            if pd.api.types.is_numeric_dtype(X_train_full[c]):
                num_cols.append(c)
            else:
                cat_cols.append(c)

    # 数据填充，数值用中位数，类别用众数
    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    X_train_full[num_cols] = num_imputer.fit_transform(X_train_full[num_cols])
    if cat_cols:
        X_train_full[cat_cols] = cat_imputer.fit_transform(X_train_full[cat_cols])

    enc = None
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64)
        X_train_full[cat_cols] = enc.fit_transform(X_train_full[cat_cols].astype(str))

    # 补充预测集缺失数据
    for c in feature_cols:
        if c not in pred_df.columns:
            pred_df[c] = np.nan
    X_pred = pred_df[feature_cols].copy()
    X_pred[num_cols] = num_imputer.transform(X_pred[num_cols])
    if cat_cols:
        X_pred[cat_cols] = cat_imputer.transform(X_pred[cat_cols])
        X_pred[cat_cols] = enc.transform(X_pred[cat_cols].astype(str))

    tools = {
        "feature_cols": feature_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "num_imputer": num_imputer,
        "cat_imputer": cat_imputer if cat_cols else None,
        "encoder": enc,
    }
    return X_train_full, y_train_full, X_pred, tools


def train_model(X_train, y_train):
    model = TabNetRegressor(**TABNET_PARAMS)
    model.fit(X_train.values, y_train.values, **FIT_PARAMS)
    return model


# 按年份划分训练集
def train_and_eval(X_all, y_all):
    train_idx, valid_idx = split_val(X_all, YEAR_COL, N_VALID_YEARS)
    X_tr, X_va = X_all.loc[train_idx], X_all.loc[valid_idx]
    y_tr, y_va = y_all.loc[train_idx], y_all.loc[valid_idx]
    print(f"训练集：{X_tr.shape}, 验证集：{X_va.shape}")
    model = train_model(X_tr, y_tr)
    y_pred = model.predict(X_va.values)
    mae, rmse, r2 = mae_rmse_r2(y_va.values, y_pred)
    print(f"MAE={mae:.4f} | RMSE={rmse:.4f} | R^2={r2:.4f}")
    return model


def predict_three_years(model, tools, train_df, pred_df_raw):
    feature_cols = tools["feature_cols"]
    num_cols = tools["num_cols"]
    cat_cols = tools["cat_cols"]
    num_imputer = tools["num_imputer"]
    cat_imputer = tools["cat_imputer"]
    enc = tools["encoder"]
    outputs = []
    for city, g in pred_df_raw.groupby(CITY_COL):
        g = g.sort_values(YEAR_COL)
        years = g[YEAR_COL].tolist()
        last_train_year = (
            int(train_df[train_df[CITY_COL] == city][YEAR_COL].max())
            if city in train_df[CITY_COL].unique()
            else int(train_df[YEAR_COL].max())
        )
        future_years_needed = [
            last_train_year + 1,
            last_train_year + 2,
            last_train_year + 3,
        ]
        has_all_future_rows = all(y in years for y in future_years_needed)
        if has_all_future_rows:
            sub = g[g[YEAR_COL].isin(future_years_needed)].copy()
            for c in feature_cols:
                if c not in sub.columns:
                    sub[c] = np.nan
            subX = sub[feature_cols].copy()
            subX[num_cols] = num_imputer.transform(subX[num_cols])
            if cat_cols:
                subX[cat_cols] = cat_imputer.transform(subX[cat_cols])
                subX[cat_cols] = enc.transform(subX[cat_cols].astype(str))
            pred_vals = model.predict(subX.values)
            pred_df_city = pd.DataFrame(pred_vals, columns=TARGET_COLS)
            pred_df_city.insert(0, YEAR_COL, sub[YEAR_COL].values)
            pred_df_city.insert(0, CITY_COL, sub[CITY_COL].values)
            outputs.append(pred_df_city)
        else:
            latest_year_in_pred = int(g[YEAR_COL].max())
            start_year = max(latest_year_in_pred, last_train_year)
            base = g[g[YEAR_COL] == start_year].copy()
            if base.empty:
                base = g.sort_values(YEAR_COL).head(1).copy()
            else:
                base = base.iloc[[0]]
            for c in feature_cols:
                if c not in base.columns:
                    base[c] = np.nan
            baseX = base[feature_cols].copy()
            baseX[num_cols] = num_imputer.transform(baseX[num_cols])
            if cat_cols:
                baseX[cat_cols] = cat_imputer.transform(baseX[cat_cols])
                baseX[cat_cols] = enc.transform(baseX[cat_cols].astype(str))
            curX = baseX.copy()
            cur_year = int(base[YEAR_COL].values[0])
            for step in range(1, 4):
                cur_year += 1
                curX_loc = curX.copy()
                curX_loc[YEAR_COL] = cur_year
                pred_vals = model.predict(curX_loc.values)
                pred_row = pd.DataFrame([pred_vals[0]], columns=TARGET_COLS)
                pred_row.insert(0, YEAR_COL, cur_year)
                pred_row.insert(0, CITY_COL, base[CITY_COL].values[0])
                outputs.append(pred_row)
    final_pred = (
        pd.concat(outputs, ignore_index=True)
        if outputs
        else pd.DataFrame(columns=[CITY_COL, YEAR_COL] + TARGET_COLS)
    )
    final_pred = final_pred.sort_values([CITY_COL, YEAR_COL]).reset_index(drop=True)
    return final_pred


def main():
    print("正在加载数据...")
    train_df, pred_df_raw = load_data()
    print("正在预处理数据...")
    X_all, y_all, X_pred_dummy, tools = prepare_features(train_df, pred_df_raw)
    print("开始训练")
    model = train_and_eval(X_all, y_all)
    print("开始预测")
    pred_out = predict_three_years(model, tools, train_df, pred_df_raw)
    print(f"生成预测结果:{OUTPUT_FILE}")
    pred_out.to_excel(OUTPUT_FILE, index=False)
    print("完成！")


if __name__ == "__main__":
    main()
