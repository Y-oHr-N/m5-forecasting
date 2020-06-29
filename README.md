# m5-forecasting

## Usage

```
git clone https://github.com/Y-oHr-N/m5-forecasting.git
cd m5-forecasting
# Install related packages
pip install -r requirements.txt
# Download raw data
mkdir -p data/raw
kaggle competitions download m5-forecasting-accuracy --path data/raw
unzip data/raw/m5-forecasting-accuracy.zip -d data/raw
# Submit submission_accuracy.csv to m5-forecasting-accuracy
python src/main.py --description DESCRIPTION --accuracy
# Submit submission_uncertantinty.csv to m5-forecasting-uncertainty
python src/main.py --description DESCRIPTION --uncertainty
```
