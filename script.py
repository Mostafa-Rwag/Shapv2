from flask import Flask, jsonify, send_from_directory
import shap
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

app = Flask(__name__)

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

# حفظ الرسم البياني كـ HTML و PNG
shap.summary_plot(shap_values, X, show=False)
shap.save_html("shap_summary_plot.html")  # حفظ الرسم كـ HTML
plt.savefig("shap_summary_plot.png")  # حفظ الرسم كـ PNG

@app.route('/')
def index():
    return jsonify({"message": "Welcome to the SHAP Flask API!"})

# نقطة النهاية لتحميل ملف HTML الخاص بـ SHAP
@app.route('/api/shap_html', methods=['GET'])
def get_shap_html():
    return send_from_directory('.', 'shap_summary_plot.html')

# نقطة النهاية لتحميل ملف PNG الخاص بـ SHAP
@app.route('/api/shap_png', methods=['GET'])
def get_shap_png():
    return send_from_directory('.', 'shap_summary_plot.png')

# نقطة النهاية للحصول على بيانات SHAP كـ JSON
@app.route('/api/shap_data', methods=['GET'])
def get_shap_data():
    shap_data = shap_values[0]  # استخدام القيم الخاصة بالفئة الأولى (يمكنك تعديلها حسب الحاجة)
    shap_data_json = [dict(zip(["feature_" + str(i), "shap_value"], [i, v])) for i, v in enumerate(shap_data)]
    return jsonify(shap_data_json)

if __name__ == '__main__':
    # تشغيل التطبيق
    app.run(debug=True)
