"""Validate restricted input membership and outcomes without printing patient IDs."""
from pathlib import Path
import pandas as pd

def main():
    root=Path(__file__).resolve().parent
    data=pd.read_csv(root/'开腹肝胆道结石手术SSI清洗.csv')
    split=pd.read_csv(root/'splits/split_folds_full.csv')
    schema=pd.read_csv(root/'特征类别/feature_types_summary.csv')
    assert data.ID.is_unique and split.ID.is_unique, 'Duplicate IDs'
    assert len(schema)==64 and schema.feature.is_unique, 'Expected 64 unique predictors'
    assert set(schema.feature).issubset(data.columns), 'Missing predictor columns'
    assert set(split.split)=={'DEV','VAL'}, 'Unexpected split values'
    assert len(split)==348 and (split.split=='DEV').sum()==278 and (split.split=='VAL').sum()==70
    joined=split.merge(data,on='ID',how='left',suffixes=('_split',''),validate='one_to_one',indicator=True)
    assert joined['_merge'].eq('both').all(), 'Missing patient rows'
    assert joined.Infection.isin([0,1]).all()
    assert joined.Infection.eq(joined.Infection_split).all(), 'Outcome mismatch'
    assert joined.Infection.sum()==66 and joined.loc[joined.split=='VAL','Infection'].sum()==11
    assert set(joined.loc[joined.split=='DEV','fold_id'])==set(range(5))
    print('Validated fixed cohort: 348 patients; 64 predictors; DEV 278; VAL 70; SSI 66.')
    print(f"BMI missing in the analytic cohort: {joined.BMI.isna().sum()}/348 ({joined.BMI.isna().mean()*100:.2f}%).")

if __name__=='__main__': main()
