# Rest-of-season plans for your team (2026-10-12 00:58)

Model: 2026 deploy: mean(extratrees, xgb) tuned on all 17 gamedays (walk-forward, official points) (planning blend on). Start: Hamilton, Antonelli, Colapinto, Lawson, Bearman, Ferrari, Racing Bulls, bank $2.4m, 3 free transfers. Prices: the game's current prices, projected with its price rule. Expected points; re-plan every week.

No Negative is valued with the forecast error (E[max(X,0)] - E[X], sd 14.6 per driver / 20.6 per constructor); Autopilot (not optimised): about 6 points in a typical week - play it in a week with no other chip. Final Fix is not modelled (it needs qualifying results).

| chips available | expected points, rest of season | chips used (race by race) |
|---|---|---|
| no chips | 994 | - |
| 3x_booster | 1036 | São Paulo GP: 3x_booster |
| no_negative | 1006 | US GP: no_negative |
| wildcard | 996 | Las Vegas GP: wildcard |
| limitless | 1050 | Mexico City GP: limitless |
| wildcard, 3x_booster, no_negative, limitless | 1105 | US GP: no_negative, Mexico City GP: limitless, São Paulo GP: wildcard, Las Vegas GP: 3x_booster |

## Chips available: none - expected 994 points over 6 races

| race | drivers | constructors | boost | chip | transfers in | out | exp. pts |
|---|---|---|---|---|---|---|---|
| US GP | Antonelli, Bearman, Hulkenberg, Lawson, Lindblad | Ferrari, Mercedes | Antonelli |  | Hulkenberg, Lindblad, Mercedes | Colapinto, Hamilton, Racing Bulls | 212.4 |
| Mexico City GP | Hadjar, Antonelli, Alonso, Bearman, Hulkenberg | Ferrari, Mercedes | Antonelli |  | Alonso, Hadjar | Lawson, Lindblad | 209.7 |
| São Paulo GP | Hadjar, Antonelli, Hulkenberg, Lindblad, Perez | Ferrari, Mercedes | Antonelli |  | Lindblad, Perez | Alonso, Bearman | 213.0 |
| Las Vegas GP | Hadjar, Antonelli, Hulkenberg, Lindblad, Perez | Ferrari, Mercedes | Antonelli |  | - | - | 212.6 |
| Qatar GP | Antonelli, Leclerc, Sainz, Albon, Perez | Ferrari, Mercedes | Antonelli |  | Albon, Leclerc, Sainz | Hadjar, Hulkenberg, Lindblad | 213.3 |
| Abu Dhabi GP | Antonelli, Leclerc, Sainz, Albon, Perez | Ferrari, Mercedes | Antonelli |  | - | - | 212.2 |

## Chips available: 3x_booster - expected 1036 points over 6 races

| race | drivers | constructors | boost | chip | transfers in | out | exp. pts |
|---|---|---|---|---|---|---|---|
| US GP | Antonelli, Bearman, Hulkenberg, Lawson, Lindblad | Ferrari, Mercedes | Antonelli |  | Hulkenberg, Lindblad, Mercedes | Colapinto, Hamilton, Racing Bulls | 212.4 |
| Mexico City GP | Antonelli, Bearman, Hulkenberg, Lawson, Perez | Ferrari, Mercedes | Antonelli |  | Perez | Lindblad | 203.1 |
| São Paulo GP | Verstappen, Antonelli, Albon, Perez, Bottas | Ferrari, Mercedes | Antonelli 3x + Verstappen 2x | 3x_booster | Albon, Bottas, Verstappen | Bearman, Hulkenberg, Lawson | 269.0 |
| Las Vegas GP | Verstappen, Antonelli, Albon, Perez, Bottas | Ferrari, Mercedes | Antonelli |  | - | - | 213.0 |
| Qatar GP | Verstappen, Antonelli, Albon, Perez, Bottas | Ferrari, Mercedes | Antonelli |  | - | - | 214.0 |
| Abu Dhabi GP | Antonelli, Leclerc, Albon, Hulkenberg, Perez | Ferrari, Mercedes | Antonelli |  | Hulkenberg, Leclerc | Bottas, Verstappen | 214.7 |

## Chips available: no_negative - expected 1006 points over 6 races

| race | drivers | constructors | boost | chip | transfers in | out | exp. pts |
|---|---|---|---|---|---|---|---|
| US GP | Antonelli, Bearman, Hulkenberg, Lawson, Lindblad | Ferrari, Mercedes | Antonelli | no_negative | Hulkenberg, Lindblad, Mercedes | Colapinto, Hamilton, Racing Bulls | 224.7 |
| Mexico City GP | Hadjar, Antonelli, Alonso, Bearman, Hulkenberg | Ferrari, Mercedes | Antonelli |  | Alonso, Hadjar | Lawson, Lindblad | 209.7 |
| São Paulo GP | Hadjar, Antonelli, Hulkenberg, Lindblad, Perez | Ferrari, Mercedes | Antonelli |  | Lindblad, Perez | Alonso, Bearman | 213.0 |
| Las Vegas GP | Hadjar, Antonelli, Hulkenberg, Lindblad, Perez | Ferrari, Mercedes | Antonelli |  | - | - | 212.6 |
| Qatar GP | Antonelli, Leclerc, Sainz, Albon, Perez | Ferrari, Mercedes | Antonelli |  | Albon, Leclerc, Sainz | Hadjar, Hulkenberg, Lindblad | 213.3 |
| Abu Dhabi GP | Antonelli, Leclerc, Sainz, Albon, Perez | Ferrari, Mercedes | Antonelli |  | - | - | 212.2 |

## Chips available: wildcard - expected 996 points over 6 races

| race | drivers | constructors | boost | chip | transfers in | out | exp. pts |
|---|---|---|---|---|---|---|---|
| US GP | Antonelli, Bearman, Hulkenberg, Lawson, Lindblad | Ferrari, Mercedes | Antonelli |  | Hulkenberg, Lindblad, Mercedes | Colapinto, Hamilton, Racing Bulls | 212.4 |
| Mexico City GP | Hadjar, Antonelli, Alonso, Bearman, Hulkenberg | Ferrari, Mercedes | Antonelli |  | Alonso, Hadjar | Lawson, Lindblad | 209.7 |
| São Paulo GP | Hadjar, Antonelli, Hulkenberg, Lindblad, Perez | Ferrari, Mercedes | Antonelli |  | Lindblad, Perez | Alonso, Bearman | 213.0 |
| Las Vegas GP | Antonelli, Leclerc, Albon, Bearman, Perez | Ferrari, Mercedes | Antonelli | wildcard | Albon, Bearman, Leclerc | Hadjar, Hulkenberg, Lindblad | 213.2 |
| Qatar GP | Antonelli, Leclerc, Albon, Bearman, Perez | Ferrari, Mercedes | Antonelli |  | - | - | 214.4 |
| Abu Dhabi GP | Antonelli, Leclerc, Albon, Bearman, Perez | Ferrari, Mercedes | Antonelli |  | - | - | 213.2 |

## Chips available: limitless - expected 1050 points over 6 races

| race | drivers | constructors | boost | chip | transfers in | out | exp. pts |
|---|---|---|---|---|---|---|---|
| US GP | Antonelli, Bearman, Hulkenberg, Lawson, Lindblad | Ferrari, Mercedes | Antonelli |  | Hulkenberg, Lindblad, Mercedes | Colapinto, Hamilton, Racing Bulls | 212.4 |
| Mexico City GP | Verstappen, Russell, Antonelli, Leclerc, Hamilton | Ferrari, Mercedes | Antonelli | limitless | Hamilton, Leclerc, Russell, Verstappen | Bearman, Hulkenberg, Lawson, Lindblad | 272.3 |
| São Paulo GP | Hadjar, Antonelli, Hulkenberg, Lindblad, Perez | Ferrari, Mercedes | Antonelli |  | Hadjar, Perez | Bearman, Lawson | 213.0 |
| Las Vegas GP | Hadjar, Antonelli, Hulkenberg, Lindblad, Perez | Ferrari, Mercedes | Antonelli |  | - | - | 212.6 |
| Qatar GP | Hadjar, Antonelli, Hulkenberg, Lindblad, Perez | Ferrari, Mercedes | Antonelli |  | - | - | 213.2 |
| Abu Dhabi GP | Hadjar, Antonelli, Hulkenberg, Lindblad, Perez | Ferrari, Mercedes | Antonelli |  | - | - | 212.1 |

## Chips available: wildcard, 3x_booster, no_negative, limitless - expected 1105 points over 6 races

| race | drivers | constructors | boost | chip | transfers in | out | exp. pts |
|---|---|---|---|---|---|---|---|
| US GP | Antonelli, Alonso, Bearman, Hulkenberg, Lawson | Ferrari, Mercedes | Antonelli | no_negative | Alonso, Hulkenberg, Mercedes | Colapinto, Hamilton, Racing Bulls | 221.7 |
| Mexico City GP | Verstappen, Russell, Antonelli, Leclerc, Hamilton | Ferrari, Mercedes | Antonelli | limitless | Hamilton, Leclerc, Russell, Verstappen | Alonso, Bearman, Hulkenberg, Lawson | 272.3 |
| São Paulo GP | Hadjar, Antonelli, Hulkenberg, Lindblad, Perez | Ferrari, Mercedes | Antonelli | wildcard | Hadjar, Lindblad, Perez | Alonso, Bearman, Lawson | 213.0 |
| Las Vegas GP | Verstappen, Antonelli, Albon, Perez, Bottas | Ferrari, Mercedes | Antonelli 3x + Verstappen 2x | 3x_booster | Albon, Bottas, Verstappen | Hadjar, Hulkenberg, Lindblad | 270.3 |
| Qatar GP | Antonelli, Leclerc, Albon, Hulkenberg, Perez | Ferrari, Mercedes | Antonelli |  | Hulkenberg, Leclerc | Bottas, Verstappen | 215.9 |
| Abu Dhabi GP | Antonelli, Leclerc, Albon, Hulkenberg, Perez | Ferrari, Mercedes | Antonelli |  | - | - | 214.7 |
