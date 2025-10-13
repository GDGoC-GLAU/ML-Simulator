# Linear Regression Model

## 🏃‍♂️ How to Run
1. Open the simulator and select **Linear Regression** from the model list.
2. Upload your dataset or use the default sample dataset.
3. Adjust parameters if available, then click **Run Simulation**.

## ⚙️ Parameters
| Parameter | Description | Default |
|------------|-------------|----------|
| `fit_intercept` | Whether to calculate the intercept term | True |
| `normalize` | Normalize input features before training | False |
| `test_size` | Proportion of data for testing | 0.2 |

## 📈 Output Plots
- **Scatter Plot:** Shows actual vs. predicted values.
- **Regression Line:** Displays the best-fit line learned by the model.
- **Error Distribution:** Optional plot showing residuals.

![Linear Regression Output](../assets/linear_regression_output.png)

## 🧩 Notes
- Works well for linearly related data.
- Avoid using with categorical or highly nonlinear datasets.
