from utils import *  # استيراد الوظائف المساعدة
import os  # استيراد مكتبة إدارة الملفات والنظام
from flask import Flask, render_template, url_for, request, redirect, jsonify  # استيراد وظائف Flask
from werkzeug.utils import secure_filename  # للتعامل مع أسماء الملفات بشكل آمن

# إعداد المتغيرات الرئيسية
MODEL_LIST = ['exp114_model.pt']  # قائمة النماذج المستخدمة للتنبؤ
DEVICE = 'cpu'  # الجهاز المستخدم (CPU في هذه الحالة)
THRESHOLD = 0.5  # العتبة المستخدمة في التنبؤ

# إعداد تطبيق Flask
app = Flask(__name__, static_url_path='/static')

# تحديد الامتدادات المسموح بها للملفات المرفوعة
app.config['ALLOWED_EXTENSIONS'] = ['.dicom', '.dcm', '.png', '.jpg', '.jpeg']
ABS_PATH_TO_STATIC = os.path.abspath("static")  # المسار المطلق لمجلد static

# الدالة الرئيسية لعرض الصفحة الرئيسية
@app.route('/')
def home_func():
    return render_template('index.html')  # عرض قالب الصفحة الرئيسية

# دالة عرض صفحة "حول"
@app.route('/about')
def about_func():
    return render_template('more-info.html')  # عرض قالب صفحة "حول"

# دالة عرض صفحة الأسئلة الشائعة
@app.route('/faq')
def faq_func():
    return render_template('faq.html')  # عرض قالب صفحة "الأسئلة الشائعة"

# دالة رفع الملفات عبر Ajax
@app.route('/upload_ajax', methods=['POST'])
def upload_ajax():
    # حذف المجلدات إذا كانت موجودة
    if os.path.isdir('uploads'):
        shutil.rmtree('uploads')
    if os.path.isdir('static/proc_images_dir'):
        shutil.rmtree('static/proc_images_dir')
    if os.path.isdir('static/pred_images_dir'):
        shutil.rmtree('static/pred_images_dir')

    # إنشاء المجلدات إذا لم تكن موجودة
    if not os.path.isdir('uploads'):
        os.mkdir('uploads')

    # معالجة الملفات المرفوعة
    file_list = request.files.getlist('my_files')
    print(file_list)  # طباعة قائمة الملفات للتتبع
    for item in file_list:
        fname = item.filename
        extension = os.path.splitext(item.filename)[1].lower()
        if extension in app.config['ALLOWED_EXTENSIONS']:
            fname = secure_filename(fname)  # تأمين اسم الملف
            item.save(f'uploads/{fname}')  # حفظ الملف في مجلد uploads

    # قراءة الملفات المرفوعة
    upfile_list = os.listdir('uploads')
    image_list = []
    for item in upfile_list:
        file_ext = item.split('.')[1].lower()
        file_ext_with_dot = f'.{file_ext}'
        if file_ext_with_dot in app.config['ALLOWED_EXTENSIONS']:
            image_list.append(item)

    # معالجة الصور المرفوعة
    process_images(image_list)

    # التنبؤ على جميع الصور
    num_preds_dict, sorted_image_list = predict_on_all_images(MODEL_LIST, DEVICE, image_list, THRESHOLD)
    print('Done')

    # إعداد عرض الصور والنتائج
    for i, item in enumerate(sorted_image_list):
        ext = fname.split('.')[1]
        dicom_ext_list = ['dcm', 'dicom']
        if ext in dicom_ext_list:
            fname = fname.replace(ext, 'png')
        if i == 0:
            image_fin_str = f"""<img class="w3-round unblock" src="/static/pred_images_dir/{item}"  height="580">"""
        else:
            image_fin_str += f"""<img class="w3-round unblock" src="/static/pred_images_dir/{item}"  height="580">"""

    # إنشاء قائمة النتائج
    start_str = "<ul>"
    for i, item in enumerate(sorted_image_list):
        pred_info_tuple = num_preds_dict[item]
        lungs_detected = pred_info_tuple[1]
        num_preds = pred_info_tuple[0]
        if lungs_detected == 'yes':
            if num_preds == 1:
                num_str = str(num_preds) + ' ' + 'opacity detected'
            else:
                num_str = str(num_preds) + ' ' + 'opacities detected'
        elif lungs_detected == 'no':
            num_str = 'Error. Image is not a chest x-ray'
        if i == 0:
            fin_str = start_str + f'<li class="row w3-text-black w3-border-right w3-border-black w3-padding-bottom" onclick="ajaxGetFilename(this.innerHTML)"><a href="#">{num_str}<br>{item}</a></li>'
        else:
            fin_str += f'<li class="row w3-padding-bottom" onclick="ajaxGetFilename(this.innerHTML)"><a href="#">{num_str}<br>{item}</a></li>'

    html_str = fin_str + '</ul>' + """<script>jQuery('li').click(function(event){
                // إزالة جميع الفئات النشطة الحالية
                jQuery('.row').removeClass('w3-text-black w3-border-right w3-border-black');

                // إضافة الفئة النشطة إلى الرابط الذي تم النقر عليه
                jQuery(this).addClass('w3-text-black w3-border-right w3-border-black');
                event.preventDefault();
                 });</script>"""

    # إعداد الصورة الرئيسية
    first_fname = sorted_image_list[0]
    ext = first_fname.split('.')[1]
    dicom_ext_list = ['dcm', 'dicom']
    if ext in dicom_ext_list:
        first_fname = fname.replace(ext, 'png')
    main_image_str = f"""<img id="selected-image"  onclick="get_click_coords(event, this.src)" class="w3-round unblock" src="/static/pred_images_dir/{first_fname}"  height="580" alt="chest x-ray">"""

    # إعداد الاستجابة النهائية
    output_response = {"html_str": html_str, "main_image_str": main_image_str, "image_fin_str": image_fin_str}

    # حذف البيانات التي قدمها المستخدم
    delete_user_submitted_data()

    return jsonify(output_response)

# دالة معالجة طلب Ajax لعرض صورة معينة
@app.route('/process_ajax', methods=['POST'])
def process_ajax():
    fname = request.form.get('file_name')
    print(fname)
    fname = fname.split('<br>')[1]
    fname = fname.replace('</a>', '')
    ext = fname.split('.')[1]
    dicom_ext_list = ['dcm', 'dicom']
    if ext in dicom_ext_list:
        image_fname = fname.split('.')[0] + '.png'
    else:
        image_fname = fname
    print(image_fname)
    info_in_html = f"""<img id="selected-image"  onclick="get_click_coords(event, this.src)" class="w3-round unblock" src="/static/pred_images_dir/{image_fname}"  height="580" alt="chest x-ray">"""
    output = {"output1": info_in_html}
    return jsonify(output)

# دالة معالجة الصور المثبتة مسبقاً
@app.route('/örnek_images', methods=['POST'])
def process_sample_ajax():
    fname = request.form.get('file_name')
    fname = fname.split('<br>')[1]
    fname = fname.replace('</a>', '')
    ext = fname.split('.')[1]
    dicom_ext_list = ['dcm', 'dicom']
    if ext in dicom_ext_list:
        image_fname = fname.split('.')[0] + '.png'
    else:
        image_fname = fname
    info_in_html = f"""<img id="selected-image"  onclick="hide_show_bboxes(this.src, this.id)" class="w3-round unblock" src="/static/örnek_images/{image_fname}"  height="580" alt="chest x-ray">"""
    output = {"output1": info_in_html}
    return jsonify(output)

# دالة معالجة معلومات النقاط التي نُقر عليها
@app.route('/process_click_info', methods=['POST'])
def process_click_info():
    fname = request.form.get('fname')
    image_fname = fname.split('/')[-1:][0]
    if 'no_bboxes_' in image_fname:
        output = show_all_bboxes(image_fname)
    else:
        output = hide_all_bboxes(image_fname)
    return jsonify(output)

# دالة معالجة نقرات الصور المثبتة مسبقاً
@app.route('/process_sample_image_click', methods=['POST'])
def process_sample_image_click():
    fname = request.form.get('fname')
    id = request.form.get('id')
    image_fname = fname.split('/')[-1:][0]
    if 'noboxes_' in image_fname:
        new_fname = fname.replace('noboxes_', '')
        output = {
                'new_fname': new_fname,
                'id': id
              }
    else:
        item_list = fname.split('/')
        image_fname = 'noboxes_' + image_fname
        last_index = len(item_list) - 1
        item_list[last_index] = image_fname
        new_fname = '/'.join(item_list)
        output = {
                'new_fname': new_fname,
                'id': id
              }
    return jsonify(output)

# دالة اختبار التطبيق
@app.route('/test')
def test():
    return 'Bu bir testtir...'

# تشغيل التطبيق
if __name__ == '__main__':
    app.run()
