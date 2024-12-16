import shap
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# تحميل مجموعة بيانات Iris من scikit-learn
data = load_iris()
X = data.data
y = data.target

# تدريب نموذج (RandomForest في هذا المثال)
model = RandomForestClassifier()
model.fit(X, y)

# استخدام SHAP لشرح النموذج
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# حفظ الرسم البياني كـ HTML
shap.summary_plot(shap_values, X, show=False)  # `show=False` لكي لا يتم عرض الرسم
shap.save_html("shap_summary_plot.html")  # حفظ الرسم كـ HTML

# أو حفظه كـ PNG
import matplotlib.pyplot as plt
shap.summary_plot(shap_values, X, show=False)  # `show=False` لكي لا يتم عرض الرسم
plt.savefig("shap_summary_plot.png")  # حفظ الرسم كـ PNG
