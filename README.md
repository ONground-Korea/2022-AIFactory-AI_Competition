# 2022-AIFactory-AI_Competition
AI competition to predict price fluctuation rate of agricultural products

### Models
- **Transformers**
- **NLinear**
- **DLinear**
- **NLinear** + **DLinear** (Our Best)

### Summary
Transformers performed poor in the tabular, time series datasets. Linear models performed well but at some point, the performance has not improved any more. So we adopted NLinear's normalizing flow (that last sequence will be significant for predicting future sequences) and added it to the DLinear model structure. We tried to use normalizing flow for both DLinear's decomposed "seasonal and trend" but it underperformed. After trial and errors, we found that fisrt normalize and then decompose it into seasonal and trend overperformed our other methods. And we used the "brand new" Nonlinear function **Mish** and it increased our performance.
