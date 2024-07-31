Battery SOC data

Training : Cycle 1; Cycle 2; Cycle 3; US06; 
Test : Cycle 4
Input : Current; Voltage; Battery_Temp_degC;
Output : Capacity;

Data was resampled at sample rate of 50 datapoint to reduce data 

for output file 
Y files ending in _Pana = capacity 
Y files ending in _TM = normalized capacity (SOC)
Y files ending in _Class = regression of capacity has been grouped into 10 class (10% - 100%)
