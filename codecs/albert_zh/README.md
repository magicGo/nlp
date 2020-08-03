# albert_zh

## 1. classifier task
### 1.1 data format:
Heater	wc热水器关一下加强智能增压<br>
Heater	wc的热水器取消加强智能增压<br>
Heater	一层主卧室热水器关闭加强智能增压功能<br>

## 2. ner task
### 2.1 data format
一 层 浴 霸 关 上 [SEP] B-floor E-floor B-deviceName E-deviceName O O<br>
一 层 浴 霸 关 掉 [SEP] B-floor E-floor B-deviceName E-deviceName O O<br>

## 3. joint classifier and ner task
### 3.1 data format
closeDevice [CLS] 一 层 浴 霸 关 上 [SEP] B-floor E-floor B-deviceName E-deviceName O O<br>
closeDevice [CLS] 一 层 浴 霸 关 掉 [SEP] B-floor E-floor B-deviceName E-deviceName O O<br>
