# GSN_rain_predictor
## Pobieranie SEVIR dataset
tylko częscti czyli ir069
```
Data key: ir107
Description: Infrared Satellite imagery (Window)
Spatial Resolution: 2 km
Patch Size: 192 x 192
Time step: 5 minutes
```
w folderze projektowym wykonać komendę:

aws s3 cp --no-sign-request s3://sevir/CATALOG.csv CATALOG.csv

aws s3 sync --no-sign-request s3://sevir/data/ir069 data/
