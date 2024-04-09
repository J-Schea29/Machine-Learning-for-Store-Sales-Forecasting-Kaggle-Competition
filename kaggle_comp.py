import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import torch
import gc

def ensemble():

    e_sum=0
    for e in es:
        e_sum =+ e

    return pred_1*e_1/e_sum + pred_2*e_2/e_sum + pred_3*e_3/e_sum + pred_4*e_4/e_sum + pred_5*e_5/e_sum + pred_6*e_6/e_sum
    
def kaggle_submit():

    ensemble().to_csv('submission.csv', index=True)
    
    api = KaggleApi()
    api.authenticate()
    api.competition_submit('submission.csv','1st API Submission','store-sales-time-series-forecasting')

def main():
    
    # Call your functions here
    pause_duration =20
    
    !python3 wrangling.py 
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(pause_duration)
    
    !python3 models/xgboost.py
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(pause_duration)
    
    !python3 models/rforest.py
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(pause_duration)
    
    !python3 models/knearest.py
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(pause_duration)
    
    !python3 models/lstm.py
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(pause_duration)
    
    kaggle_submit()

if __name__ == "__main__":
    main()

    