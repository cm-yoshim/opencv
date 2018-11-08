'''
こちらを参考に
https://developers.cyberagent.co.jp/blog/archives/12666/
http://pynote.hatenablog.com/entry/opencv-findcontours
'''

import cv2
import time
import numpy as np

# 輪郭を絞り込む関数（サイズで絞り込み）
def extract_contours(contours, size):
    '''
    contours:領域の四点のx,y座標。
    size:どのくらいのサイズ以上だったら抽出するのか、という閾値。小さすぎると腕以外のものも検出してしまう。

    返り値:「size」で指定した面積以上の領域をリスト形式で返す。
    '''
    area = 0
    list_extracted_contours = []
    for i in contours:
        area = cv2.contourArea(i)
        if area >= size:
            list_extracted_contours.append(i)
    
    return list_extracted_contours


# 輪郭（長方形）を抽出し、画像に出力する関数
def get_rect(img, contours):
    '''
    img:画像。0でない部分は1とした2値画像。
    contours:輪郭のx,y座標の情報。リスト形式を想定。

    返り値:検出された「腕の座標」。
    '''

    dict_box = {}
    for i in range(len(contours)):
        cnt = contours[i]
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        extract_rect = cv2.drawContours(img,[box],0,(0,0,255),2)       
        dict_box[i] = box
        cv2.imshow('extract_rect', extract_rect)

    return dict_box


# 腕がどこからきたのかを判別する関数
def get_origin_of_arm(dict_box):
    dict_output = {}
    '''
    dict_box:抽出された短径の座標が格納された辞書型変数。
            下記のような変数を期待しています。
            {0: array([[182, 424],
                    [128,   0],
                    [322, -24],
                    [376, 399]]), 
            1: array([[148, 418],
                    [  0, 418],
                    [  0,   0],
                    [148,   0]])}

    返り値:各領域が「左右正面」のいずれから出てきたかの辞書
    '''
    if len(dict_box) != 0:
        # 抽出された領域全てに対してループ
        for i in dict_box:
            
            # 4角の座標位置から、腕がどこからきているかを評価
            for j in range(len(dict_box[i])):
                if dict_box[i][j][0] <= 0: # いずれかのx座標が0以下なら、右から手が出ている(カメラで上から見ていることを想定)
                    dict_output[i] = 'right'
                    break
                elif dict_box[i][j][0] >= width_of_img: # いずれかのx座標が設定したサイズの最大値以上なら、左から手が出ている(カメラで上から見ていることを想定)
                    dict_output[i] = 'left'
                    break
                elif dict_box[i][j][1] <= 0: # いずれかのy座標が0以下なら、正面から手が出ている(カメラで上から見ていることを想定)
                    dict_output[i] = 'front'
                else:
                    pass

                # 左右正面のいずれでもない場合は...「わかりませんでした」という結果を返そう
                if j + 1 == len(dict_box[i]) and i not in dict_output:
                    dict_output[i] = 'unknown!!!'
                
    return dict_output

# 画像は全てグレースケールで処理する
width_of_img = 960
height_of_img = 780
fps = 30

cap = cv2.VideoCapture(1)
cap.set(3,width_of_img) # WIDTH
cap.set(4,height_of_img) # HEIGHT
cap.set(5,fps) # FPS

avg = None

while(True):
    ret, frame = cap.read()    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 前フレームを保存
    if avg is None:
        avg = gray.copy().astype("float")
        continue

    # 現在のフレームと移動平均との間の差を計算する
    # accumulateWeighted関数の第三引数は「どれくらいの早さで以前の画像を忘れるか」。小さければ小さいほど「最新の画像」を重視する。
    # http://opencv.jp/opencv-2svn/cpp/imgproc_motion_analysis_and_object_tracking.html
    # 小さくしないと前のフレームの残像が残る
    # 重みは蓄積し続ける。
    cv2.accumulateWeighted(gray, avg, 0.00001)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

    # 閾値を設定し、フレームを2値化
    thresh = cv2.threshold(frameDelta, 50, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite('./thresh.jpg', thresh)

    # 輪郭を見つける
    _, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # 輪郭を「ある程度以上の大きさのものだけ」に絞り込み
    size = 2000
    list_extracted_contours = extract_contours(contours, size)

    # 輪郭を見つけて画像に出力
    thresh_img = cv2.imread('./thresh.jpg')
    dict_box = get_rect(thresh_img, list_extracted_contours)

    # 比較用に普通の画像も表示
    cv2.imshow('raw', frame)

    # 腕の出どころを取得
    origin_of_arm = get_origin_of_arm(dict_box)
    for i in origin_of_arm.values():
        if i != 'unknown!!!':
            print(i)
    
    # avg = None # 前フレームと比較するならコメントアウトを外す。

    if cv2.waitKey(1) != -1:
        break
 
cap.release()
cv2.destroyAllWindows()
