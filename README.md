# GSN_rain_predictor
## Pobieranie SEVIR dataset
tylko częscti czyli ir069
```
Sensor 	Data key 	Description 	Spatial Resolution 	Patch Size 	Time step
ir107 	Infrared Satellite imagery (Window) 	2 km 	192 x 192 	5 minutes
```
w /data wykonać komendę:
aws s3 cp --no-sign-request s3://sevir/CATALOG.csv CATALOG.csv
aws s3 sync --no-sign-request s3://sevir/data/ir107 .
