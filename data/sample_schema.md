# Data Schema

Optimum Matrix expects `.xlsx` files with the following columns.

## Required Columns

| Column | Type | Unit | Description |
|---|---|---|---|
| `Timestamp` | datetime | — | Reading timestamp |
| `Chill_Water_Inlet_Flowrate` | float | L/min | Chilled water flow into cooling circuit |
| `Chill_Water_Inlet_Temperature` | float | °C | Chilled water inlet temperature |
| `Chill_Water_Outlet_Temperature` | float | °C | Chilled water outlet temperature |
| `Chill_Water_Tank_Temp` | float | °C | Chilled water tank temperature |
| `Cooling_Roll_Area_Humidity` | float | % RH | Humidity in the cooling roll area |
| `Cooling_Roll_Area_Temperature` | float | °C | Ambient temp in the cooling roll area |
| `Cooling_Roller_1_Surface_Temp` | float | °C | Surface temp of cooling roller 1 |
| `Cooling_Roller_2_Surface_Temp` | float | °C | Surface temp of cooling roller 2 |
| `Cooling_Roller_3_Surface_Temp` | float | °C | Surface temp of cooling roller 3 |
| `Cooling_Roller_4_Surface_Temp` | float | °C | Surface temp of cooling roller 4 |
| `PID_Output_Valve_Status` | float | % open | Cooling valve position (PID output) |
| `PID_Set_Point` | float | °C | Target rewinder web temperature |
| `Rewinder_Web_Temp` | float | °C | Measured rewinder web temperature |
| `Zone_8_Temperature` | float | °C | Dryer zone 8 temperature |
| `Web Speed` | float | m/min | Web (paper/film) travel speed |
| `Web Tension` | float | N | Web tension |
| `Post_Dryer_Web_Temp_OP` | float | °C | Web temperature immediately after dryer |

## Training-only Column

| Column | Type | Description |
|---|---|---|
| `Web_break` | int (0/1) | 1 = web break event occurred |

## Sampling

The model auto-detects the sampling interval and adjusts its prediction horizon accordingly. Typical deployments use 1-minute readings with a 60-minute forecast horizon.
