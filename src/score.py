import argparse, joblib, pandas as pd

def score(model_path, input_path, output_path):
    bundle=joblib.load(model_path); df=pd.read_csv(input_path); df['predicted_loss']=bundle['model'].predict(df); df['predicted_loss']=df['predicted_loss'].clip(lower=0); df.to_csv(output_path,index=False); return df
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--model',default='artifacts/model.joblib'); p.add_argument('--input',required=True); p.add_argument('--output',default='predictions.csv'); a=p.parse_args(); score(a.model,a.input,a.output)
