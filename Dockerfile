# استخدم صورة Python كقاعدة
FROM python:3.9-slim

# تعيين الدليل العامل داخل الحاوية
WORKDIR /app

# نسخ الملفات الخاصة بك إلى الحاوية
COPY . /app

# تثبيت المكتبات اللازمة
RUN pip install --upgrade pip
RUN pip install shap numpy scikit-learn matplotlib flask plot

# نسخ السكربت الخاص بك (تأكد من وجود سكربت Python داخل المجلد الحالي)
COPY script.py /app/script.py

# تشغيل السكربت عند بدء الحاوية
CMD ["python", "script.py"]
