from PIL import Image
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

import numpy as np
import pandas as pd
import os
import cv2
import shutil
from flask import request

import torch
import torchvision.transforms as T

# قراءة صورة الأشعة السينية من ملف DICOM
# يمكن استخدام VOI LUT وتحويل الصور أحادية اللون إذا كان مطلوبًا
def read_xray(path, voi_lut=True, fix_monochrome=True):
    
    dicom = pydicom.read_file(path)

    # تطبيق VOI LUT إذا كان متاحًا لتحويل البيانات الخام إلى صورة "مفهومة" للبشر
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    # إذا كانت الصورة تبدو مقلوبة بناءً على تفسير الفوتومتريك، نقوم بإصلاحها
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    # تحويل البيانات إلى نطاق [0, 255]
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    return data

# تغيير حجم الصورة إلى أبعاد محددة
# يمكن الحفاظ على نسبة الأبعاد أو تجاهلها
def resize(array, size, keep_ratio=False, resample=Image.LANCZOS):
    
    im = Image.fromarray(array)

    if keep_ratio:
        im.thumbnail((size, size), resample)
    else:
        im = im.resize((size, size), resample)

    return im

# رسم مربعات الإحاطة (Bounding Boxes) على الصورة مع خيار إضافة نص
def draw_bbox(image, xmin, ymin, xmax, ymax, text=None, line_thickness=20):
    
    w = xmax - xmin
    h = ymax - ymin

    # رسم المربع
    start_point = (xmin, ymin)
    end_point = (xmax, ymax)
    bbox_color = (255, 255, 255)  # اللون أبيض
    bbox_thickness = line_thickness

    image = cv2.rectangle(image, start_point, end_point, bbox_color, bbox_thickness)

    # إضافة خلفية النص إذا كان النص موجودًا
    if text:
        text_bground_color = (0, 0, 0)  # اللون أسود
        cv2.rectangle(image, (xmin, ymin - 150), (xmin + w, ymin), text_bground_color, -1)

        # إضافة النص
        text_color = (255, 255, 255)  # اللون أبيض
        font = cv2.FONT_HERSHEY_DUPLEX
        origin = (xmin, ymin - 30)
        fontScale = 3
        thickness = 10

        image = cv2.putText(image, text, origin, font,
                            fontScale, text_color, thickness, cv2.LINE_AA)

    return image

# تحويل الصورة إلى شكل مربع عن طريق تعبئة الجوانب الفارغة
def pad_image_to_square(image):

    shape_tuple = image.shape
    height = image.shape[0]
    width = image.shape[1]

    def pad_image_channel(image_channel, height, width):
        pad_amt = abs(height - width)

        if height == width:
            pad_channel = image_channel
        elif height > width:  # تعبئة يمينًا
            pad_channel = np.pad(image_channel, [(0, 0), (0, pad_amt)], mode='constant')
        else:  # إذا كان العرض أكبر، تعبئة أسفل
            pad_channel = np.pad(image_channel, [(0, pad_amt), (0, 0)], mode='constant')

        return pad_channel

    if len(shape_tuple) == 2:  # صورة رمادية
        padded_image = pad_image_channel(image, height, width)
    elif len(shape_tuple) == 3:  # صورة بأكثر من قناة
        num_channels = image.shape[2]

        for j in range(num_channels):
            image_channel = image[:, :, j]
            padded_channel = pad_image_channel(image_channel, height, width)

            if j == 0:
                padded_image = padded_channel
            else:
                padded_image = np.dstack((padded_image, padded_channel))

    return padded_image

# معالجة قائمة من الصور وتحويلها إلى صور مربعة وحفظها
def process_images(image_file_list):
    
    if not os.path.isdir('static/proc_images_dir'):
        proc_images_dir = os.path.join('static', 'proc_images_dir')
        os.mkdir(proc_images_dir)

    for i, image_fname in enumerate(image_file_list):
        ext = image_fname.split('.')[-1]
        path = 'uploads/' + image_fname

        if ext in ['dcm', 'dicom']:
            image = read_xray(path)
            image = pad_image_to_square(image)
            new_fname = image_fname.replace(ext, 'png')
            new_path = 'static/proc_images_dir/' + new_fname
            cv2.imwrite(new_path, image)
        else:
            image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            image = pad_image_to_square(image)
            new_path = 'static/proc_images_dir/' + image_fname
            cv2.imwrite(new_path, image)

# معالجة التوقعات من نموذج FasterRCNN
# تصفية النتائج بناءً على عتبة الثقة المحددة
def process_fasterrcnn_preds(pred, conf_threshold):
    pred_scores_list = list(pred[0]['scores'].detach().cpu().numpy())
    pred_labels_list = list(pred[0]['labels'].detach().cpu().numpy())
    pred_boxes = pred[0]['boxes'].detach().cpu().numpy()
    pred_boxes_list = [list(item) for item in pred_boxes]

    fin_scores_list = []
    fin_labels_list = []
    fin_boxes_list = []

    if pred_scores_list:
        for i, item in enumerate(pred_scores_list):
            if item > conf_threshold:
                fin_scores_list.append(item)
                fin_labels_list.append(pred_labels_list[i])
                fin_boxes_list.append(pred_boxes_list[i])

    pred_dict = {
        'pred_scores': fin_scores_list,
        'pred_labels': fin_labels_list,
        'pred_boxes': fin_boxes_list
    }

    return pred_dict


def create_pred_dataframe(pred_dict, fname, image_height, image_width):

    num_labels = len(pred_dict['pred_labels'])

    # إذا لم يتنبأ النموذج بأي مربعات، فإن
    # قائمة pred_labels ستكون فارغة، [].
    if num_labels == 0:

        # إنشاء إطار البيانات
        # استخدم رقم مثل 0 وليس سلسلة نصية.
        # السلسلة النصية تغير نوع البيانات في العمود وهذا
        # سيؤدي إلى حدوث أخطاء لاحقاً.
        empty_dict = {
            'xmin': [0],  # إحداثيات x للزاوية العليا اليسرى
            'ymin': [0],  # إحداثيات y للزاوية العليا اليسرى
            'xmax': [0],  # إحداثيات x للزاوية السفلى اليمنى
            'ymax': [0],  # إحداثيات y للزاوية السفلى اليمنى
            'pred_score': [0],  # درجة التنبؤ
            'pred_labels': [2],  # التصنيف المتنبأ به
            'fname': fname,  # اسم الملف
            'orig_image_height': image_height,  # ارتفاع الصورة الأصلية
            'orig_image_width': image_width  # عرض الصورة الأصلية
        }


        df = pd.DataFrame(empty_dict)

    else:

        # إنشاء إطار البيانات
        df1 = pd.DataFrame(pred_dict)
        # إضافة عمود fname
        df1['fname'] = fname

        # إنشاء مصفوفة numpy
        boxes_np = np.array(list(df1['pred_boxes']))

        # استخدام مصفوفة numpy لإنشاء إطار البيانات
        cols = ['xmin', 'ymin', 'xmax', 'ymax']
        df2 = pd.DataFrame(boxes_np, columns=cols)

        # دمج البيانات جنباً إلى جنب
        df = pd.concat([df1, df2], axis=1)
        # إزالة عمود pred_boxes
        df = df.drop('pred_boxes', axis=1)

        # إضافة أعمدة الارتفاع والعرض
        df['orig_image_height'] = image_height
        df['orig_image_width'] = image_width

    # إذا كان التصنيف هو 2 أي أنه طبيعي (دون تعتيم)، فإن
    # نضع جميع الإحداثيات لتلك الصفوف إلى 0.
    xmin_list = []
    ymin_list = []
    xmax_list = []
    ymax_list = []

    for i in range(0, len(df1)):

        pred_label = df.loc[i, 'pred_labels']
        xmin = df.loc[i, 'xmin']
        ymin = df.loc[i, 'ymin']
        xmax = df.loc[i, 'xmax']
        ymax = df.loc[i, 'ymax']

        if pred_label == 2:

            xmin_list.append(0)
            ymin_list.append(0)
            xmax_list.append(0)
            ymax_list.append(0)

        else:
            xmin_list.append(xmin)
            ymin_list.append(ymin)
            xmax_list.append(xmax)
            ymax_list.append(ymax)

    df['xmin'] = xmin_list
    df['ymin'] = ymin_list
    df['xmax'] = xmax_list
    df['ymax'] = ymax_list

    return df


def predict_on_all_images(model_list, device, image_list, threshold):

    # إنشاء مجلد pred_images_dir.
    if os.path.isdir('static/pred_images_dir') == False:
        pred_images_dir = 'static/pred_images_dir'
        os.mkdir(pred_images_dir)

    print('بدأ التنبؤ...')
    print(os.listdir('static/proc_images_dir'))

    model_path_0 = f"TRAINED_MODEL_FOLDER/{model_list[0]}"

    # تحميل النموذج المدرب
    model = torch.load(model_path_0, map_location=torch.device('cpu'))
    # model = torch.load(path_model)

    # وضع النموذج في وضع التقييم
    model.eval()

    # إرسال النموذج إلى الجهاز
    model.to(device)

    num_preds_dict = {}

    for i, fname in enumerate(image_list):

        print(f'التنبؤ بالصورة {i+1} من {len(image_list)}...')

        # الاحتفاظ باسم الملف الأصلي الذي قد يحتوي على امتداد dicom.
        # نحن بحاجة إلى عرض هذا الاسم على الصفحة.
        orig_fname = fname

        # إذا كان الملف يحتوي على امتداد dicom
        # استبدل الامتداد بـ png.
        ext = fname.split('.')[1]
        dicom_ext_list = ['dcm', 'dicom']

        if ext in  dicom_ext_list:
            fname = fname.replace(ext, 'png')


        path = 'static/proc_images_dir/' + fname

        # تحميل الصورة باستخدام PIL
        # لا تحويلها إلى تدرج الرمادي
        #image = Image.open(path)

        # تحميل الصورة باستخدام PIL
        # تحميل وتحويل الصورة إلى تدرج الرمادي
        image = Image.open(path).convert("L")

        # توسيع الصورة إلى مربع.
        # لاحظ أن الصور كانت قد تم توسيعها إلى مربع بالفعل في دالة process_images()

        # الحصول على حجم الصورة

        # تحويل صورة PIL إلى مصفوفة numpy
        image1 = np.array(image)
        # الحصول على الارتفاع والعرض
        image_height = image1.shape[0]
        image_width = image1.shape[1]

        # تحويل الصورة إلى Tensor باستخدام Torch.
        my_transform = T.Compose([T.ToTensor()])
        image = my_transform(image)

        # إرسال الصورة إلى الجهاز
        image = image.to(device)

        # التنبؤ بالصورة
        pred = model([image])

        pred_dict = process_fasterrcnn_preds(pred, threshold)


        # إنشاء إطار البيانات للتنبؤ
        df1 = create_pred_dataframe(pred_dict, fname, image_height, image_width)

        # دمج إطارات البيانات لكل صورة
        if i == 0:
            df_fin = df1
        else:
            df_fin = pd.concat([df_fin, df1], axis=0)



        # رسم مربعات التحديد على الصورة
        # ملاحظة: هنا نحتاج إلى عدم رسم مربعات التحديد على الصور الطبيعية.
        # ---------------------------

        # تحميل الصورة الأصلية
        path = 'static/proc_images_dir/' + fname
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)


        pred_boxes_list = pred_dict['pred_boxes']
        pred_labels_list = pred_dict['pred_labels']

        # 2 هو التصنيف للصور الطبيعية.
        # عندما تكون الصورة طبيعية يتنبأ النموذج بمربع تغطي
        # الصورة بالكامل.
        # هنا سنتأكد من أن مربع التحديد للتنبؤ بالتصنيف 2 لن يتم
        # رسمه على الصورة أو تضمينه في حساب التنبؤات.

        new_label_list = []
        new_bbox_list = []

        # هذه قائمة التصنيفات التي لا نريد رسم مربعات التحديد لها
        # على الصورة.
        # التصنيف 2 يشير إلى مربع حول صورة طبيعية
        # التصنيف 3 يشير إلى مربع حول الرئتين
        # احذف التصنيف 3 من هذه القائمة إذا أردت رؤية
        # المربع حول الرئتين. لكن تذكر أن الكود سيضم هذا المربع في عدد التعتيمات المكتشفة.
        ignore_labels_list = [2, 3] #[2, 3]

        lungs_detected = 'no'

        for j in range(0, len(pred_labels_list)):

            label = pred_labels_list[j]
            bbox = pred_boxes_list[j]

            # تحقق من اكتشاف الرئتين
            if label == 3:
                lungs_detected = 'yes'


            if label not in ignore_labels_list:
                new_label_list.append(label)
                new_bbox_list.append(bbox)

        num_preds = len(new_label_list)


        # إضافة مفتاح وقيمة إلى القاموس
        #num_preds_dict[orig_fname] = num_preds
        num_preds_dict[orig_fname] = (num_preds, lungs_detected)

        for i in range(0, len(new_bbox_list)):

            coords_list = new_bbox_list[i]

            #image = draw_pred_bbox_on_image(image, bbox_coords)

            xmin = int(coords_list[0])
            ymin = int(coords_list[1])
            xmax = int(coords_list[2])
            ymax = int(coords_list[3])

            image = draw_bbox(image, xmin, ymin, xmax, ymax, text=None, line_thickness=2)


        # حفظ الصورة مع مربعات التحديد المرسومة فيها
        dst = os.path.join('static/pred_images_dir/', fname)
        cv2.imwrite(dst, image)

    # حفظ df_fin كملف csv.
    # يمكننا استخدام هذا الملف لاحقاً للتحكم
    # بما يحدث عندما ينقر المستخدم على الصورة.
    path = 'df_fin_preds.csv'
    df_fin.to_csv(path, index=False)

    print('تم التنبؤ.')

    # ترتيب الصور التي سيتم عرض التنبؤات لها.
    # الصور التي تحتوي على تعتيم يجب عرضها أولاً.

    list1 = []
    list2 = []

    for key, value in num_preds_dict.items():

        # القيمة هي زوج: (عدد التنبؤات، اكتشاف الرئتين)
        if value[0] > 0:
            list1.append(key)
        else:
            list2.append(key)

    # دمج القوائم في قائمة واحدة
    sorted_image_list = list1 + list2

    assert len(sorted_image_list) == len(image_list)

    return num_preds_dict, sorted_image_list




def delete_user_submitted_data():

    """
    ملاحظة:
    هذه الدالة لا تحذف الصور في مجلد 'static/pred_images_dir'.
    التطبيق يحتاج هذه الصور لعرضها في الصفحة الرئيسية.
    يتم مسح مجلد 'static/pred_images_dir' في كل مرة يرسل المستخدم ملفات جديدة.
"""

    # حذف المجلدات ومحتوياتها.
    # هذا من أجل أمان البيانات.
    if os.path.isdir('uploads') == True:
        shutil.rmtree('uploads')

    # حذف المجلدات ومحتوياتها.
    # هذا من أجل أمان البيانات.
    #if os.path.isdir('static/proc_images_dir') == True:
        #shutil.rmtree('static/proc_images_dir')


    # حذف المجلدات ومحتوياتها.
    # هذا من أجل أمان البيانات.
    #if os.path.isdir('static/pred_images_dir') == True:
        #shutil.rmtree('static/pred_images_dir')



# عند نقر المستخدم على الصورة
# هذه الدالة تجعل مربعات التحديد تختفي.
def hide_all_bboxes(image_fname):

    # حذف مجلد صور التحليل إذا كان موجوداً.
    if os.path.isdir('static/analysis_images_dir') == True:
        shutil.rmtree('static/analysis_images_dir')
        print('تم حذف المجلد.')

    # إنشاء مجلد analysis_images_dir.
    if os.path.isdir('static/analysis_images_dir') == False:
        analysis_images_dir = 'static/analysis_images_dir'
        os.mkdir(analysis_images_dir)


    # تحميل صورة جديدة دون النقاط أو مربعات التحديد
    path = os.path.join('static/proc_images_dir', image_fname)
    image = cv2.imread(path)

    # تغيير اسم الصورة للإشارة إلى أن مربعات التحديد قد اختفت
    image_fname = 'no_bboxes_' + image_fname


    # سوف نستخدم هذا لإنشاء اسم مجلد جديد.
    k = str(99)

    # المشكلة:
    # نريد عرض نفس الصورة في كل مرة مع رسم مربع التحديد في مكان مختلف.
    # ولكن في الكود الجديد التالي، المتصفح لن يغير الصورة المعروضة
    # إذا كان مسار src للصورة الجديدة هو نفسه مثل الصورة السابقة.
    # الحل:
    # سنخزن كل صورة معدلة في مجلد مختلف. هذا سيغير مسار src بينما
    # يبقى نفس image_fname. نحتاج إلى أن يبقى اسم الملف كما هو لأننا في كل مرة
    # نحتاج إلى تحميل الصورة التي أرسلها المستخدم، وهي محفوظة في مجلد png_images_dir.

    # يمكننا تغيير اسم المجلد في كل مرة لأن
    # نحتاج فقط إلى اسم الملف الذي في نهاية سمة src.
    # طالما أن اسم الملف يبقى كما هو كل مرة، كل شيء سيعمل.
    new_image_str = f"""<img id="selected-image" onclick="get_click_coords(event, this.src)"  class="w3-round unblock" src="/static/analysis_images_dir/{k}/{image_fname}"  height="580" alt="Wheat">"""


    # فقط إذا نقر المستخدم داخل مربع التحديد
    if new_image_str != 'None':

        print('قام المستخدم بالنقر على الصورة.')

        # إنشاء مجلد analysis_images_dir.
        if os.path.isdir(f'static/analysis_images_dir/{k}') == False:
            analysis_images_dir = f'static/analysis_images_dir/{k}'
            os.mkdir(analysis_images_dir)

        # حفظ الصورة
        dst = os.path.join(f'static/analysis_images_dir/{k}', image_fname)
        cv2.imwrite(dst, image)


    # إذا لم ينقر المستخدم داخل مربع التحديد، فإن
    # new_image_str == 'None'.
    # ثم لن يغير كود الجافا سكربت الصورة على الصفحة.
    # ستظل الصورة الحالية كما هي.
    output = {
                'new_image_str': new_image_str
              }

    return output



# عند نقر المستخدم على صورة
# هذه الدالة تجعل مربعات التحديد تظهر.
def show_all_bboxes(image_fname):

    # إزالة اسم 'no_bboxes_' من الاسم، إذا كان موجوداً
    image_fname = image_fname.replace('no_bboxes_', '')

    # تحميل صورة جديدة دون النقاط أو مربعات التحديد
    #path = os.path.join('static/pred_images_dir', image_fname)
    #image = cv2.imread(path)

    # تحميل الصورة المتنبأ بها والتي تم رسم مربعات التحديد عليها
    new_image_str = f"""<img id="selected-image" onclick="get_click_coords(event, this.src)"  class="w3-round unblock" src="/static/pred_images_dir/{image_fname}"  height="580" alt="Wheat">"""

    # فقط إذا نقر المستخدم داخل مربع التحديد
    if new_image_str != 'None':
        print('قام المستخدم بالنقر على الصورة.')
        output = {
            'new_image_str': new_image_str
        }
    else:
        output = {
            'new_image_str': 'None'
        }
    return output


# هذه الدالة ترسم مربعات التحديد على الصورة
def draw_bbox(image, xmin, ymin, xmax, ymax, text=None, line_thickness=2):
    """
    هذه الدالة ترسم مربعات التحديد على الصورة.
    """
    # رسم مستطيل على الصورة باستخدام OpenCV
    color = (255, 0, 0)  # اللون الأزرق
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, line_thickness)

    return image
