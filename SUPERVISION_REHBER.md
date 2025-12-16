# 📚 Supervision Kütüphanesi Rehberi

Bu rehber, futbol analiz projesinde kullanılan **Supervision** kütüphanesi tekniklerini açıklar.

> **Supervision**, Roboflow tarafından geliştirilen, bilgisayarla görme projelerinde tespit sonuçlarını işlemek ve görselleştirmek için kullanılan güçlü bir Python kütüphanesidir.

---

## 📦 Kurulum

```bash
pip install supervision
```

```python
import supervision as sv
```

---

## 1️⃣ Detections (Tespitler)

### sv.Detections Sınıfı

Tespit edilen nesnelerin tüm bilgilerini tutan ana veri yapısı.

```python
# YOLO modelinden Supervision formatına dönüştürme
from ultralytics import YOLO

model = YOLO("model.pt")
results = model(frame)[0]

# Supervision Detections'a dönüştür
detections = sv.Detections.from_ultralytics(results)
```

### Detections Özellikleri

| Özellik | Açıklama | Örnek Değer |
|---------|----------|-------------|
| `xyxy` | Bounding box koordinatları [x1, y1, x2, y2] | `[[100, 200, 300, 400], ...]` |
| `confidence` | Güven skorları (0-1) | `[0.95, 0.87, 0.92, ...]` |
| `class_id` | Sınıf ID'leri | `[0, 1, 0, 2, ...]` |
| `tracker_id` | Takip ID'leri (tracking aktifse) | `[1, 2, 5, 7, ...]` |
| `data` | Ek veriler (class_name vb.) | `{'class_name': ['person', ...]}` |

### Detections Kullanım Örnekleri

```python
# Toplam tespit sayısı
print(len(detections))  # 10

# İlk tespitin bbox'ı
print(detections.xyxy[0])  # [100, 200, 300, 400]

# Tüm güven skorları
print(detections.confidence)  # [0.95, 0.87, ...]

# Sınıf isimleri
print(detections.data['class_name'])  # ['person', 'ball', ...]

# Belirli bir tespiti seçme (slicing)
tek_tespit = detections[0:1]  # İlk tespit

# Filtreleme (confidence > 0.8)
filtreli = detections[detections.confidence > 0.8]
```

### Manuel Detections Oluşturma

```python
detections = sv.Detections(
    xyxy=np.array([[100, 100, 200, 200], [300, 300, 400, 400]]),
    confidence=np.array([0.9, 0.85]),
    class_id=np.array([0, 1])
)
```

---

## 2️⃣ Tracking (Takip)

### ByteTrack

Yüksek performanslı çoklu nesne takip algoritması.

```python
# Tracker oluşturma
tracker = sv.ByteTrack(
    track_activation_threshold=0.25,  # Yeni track için min güven
    lost_track_buffer=50,             # Kayıp track toleransı (frame)
    minimum_matching_threshold=0.8,   # Eşleştirme IoU eşiği
    frame_rate=30                     # Video FPS
)

# Tracking uygulama
detections = tracker.update_with_detections(detections)

# Artık tracker_id kullanılabilir
print(detections.tracker_id)  # [1, 2, 3, 5, ...]
```

### ByteTrack Parametreleri

| Parametre | Açıklama | Varsayılan |
|-----------|----------|------------|
| `track_activation_threshold` | Yeni track başlatma eşiği | 0.25 |
| `lost_track_buffer` | Kayıp nesne bekle (frame) | 30 |
| `minimum_matching_threshold` | IoU eşleştirme eşiği | 0.8 |
| `frame_rate` | Video FPS değeri | 30 |

---

## 3️⃣ Annotators (Görselleştiriciler)

### 3.1 BoxAnnotator - Klasik Kutu

```python
box_annotator = sv.BoxAnnotator(
    thickness=2,           # Çizgi kalınlığı
    color=sv.Color.RED     # Kutu rengi
)

annotated_frame = box_annotator.annotate(
    scene=frame,
    detections=detections
)
```

### 3.2 RoundBoxAnnotator - Yuvarlak Köşeli Kutu ⭐

```python
round_box_annotator = sv.RoundBoxAnnotator(
    thickness=2,       # Çizgi kalınlığı
    roundness=0.6,     # Köşe yuvarlaklığı (0-1)
    color=sv.Color.BLUE
)

annotated_frame = round_box_annotator.annotate(
    scene=frame,
    detections=detections
)
```

### 3.3 BoxCornerAnnotator - Köşe Stili

```python
corner_annotator = sv.BoxCornerAnnotator(
    thickness=2,
    corner_length=15,      # Köşe çizgi uzunluğu
    color=sv.Color.WHITE
)

annotated_frame = corner_annotator.annotate(
    scene=frame,
    detections=detections
)
```

### 3.4 EllipseAnnotator - Elips (Ayak Altı)

```python
ellipse_annotator = sv.EllipseAnnotator(
    thickness=2,
    start_angle=-45,   # Başlangıç açısı
    end_angle=225      # Bitiş açısı
)

annotated_frame = ellipse_annotator.annotate(
    scene=frame,
    detections=detections
)
```

### 3.5 CircleAnnotator - Daire

```python
circle_annotator = sv.CircleAnnotator(
    thickness=2,
    color=sv.Color.GREEN
)

annotated_frame = circle_annotator.annotate(
    scene=frame,
    detections=detections
)
```

### 3.6 LabelAnnotator - Etiket Metni ⭐

```python
label_annotator = sv.LabelAnnotator(
    text_scale=0.5,                      # Metin boyutu
    text_padding=5,                      # Metin kenar boşluğu
    text_position=sv.Position.TOP_CENTER,# Metin konumu
    color=sv.Color.BLACK,                # Arka plan rengi
    text_color=sv.Color.WHITE            # Metin rengi
)

# Etiket listesi hazırla
labels = ["Oyuncu #1", "Oyuncu #2", "Top"]

annotated_frame = label_annotator.annotate(
    scene=frame,
    detections=detections,
    labels=labels
)
```

### 3.7 TraceAnnotator - Hareket İzi ⭐

```python
trace_annotator = sv.TraceAnnotator(
    trace_length=40,                     # İz uzunluğu (frame)
    thickness=2,                         # İz kalınlığı
    position=sv.Position.BOTTOM_CENTER   # İzin çıkış noktası
)

annotated_frame = trace_annotator.annotate(
    scene=frame,
    detections=detections
)
```

### 3.8 HeatMapAnnotator - Isı Haritası

```python
heatmap_annotator = sv.HeatMapAnnotator(
    radius=40,           # Isı yarıçapı
    kernel_size=25,      # Kernel boyutu
    top_hue=0,           # Üst renk tonu (kırmızı)
    low_hue=125          # Alt renk tonu (mavi)
)

annotated_frame = heatmap_annotator.annotate(
    scene=frame,
    detections=detections
)
```

---

## 4️⃣ Position (Konum Sabitleri)

Annotator'larda kullanılan konum sabitleri:

```python
sv.Position.CENTER           # Merkez
sv.Position.TOP_CENTER       # Üst merkez
sv.Position.TOP_LEFT         # Sol üst
sv.Position.TOP_RIGHT        # Sağ üst
sv.Position.BOTTOM_CENTER    # Alt merkez ⭐ (ayak pozisyonu için)
sv.Position.BOTTOM_LEFT      # Sol alt
sv.Position.BOTTOM_RIGHT     # Sağ alt
```

---

## 5️⃣ Color (Renk Sistemi)

### Hazır Renkler

```python
sv.Color.RED
sv.Color.GREEN
sv.Color.BLUE
sv.Color.YELLOW
sv.Color.WHITE
sv.Color.BLACK
sv.Color.CYAN
sv.Color.MAGENTA
```

### Özel Renk Tanımlama

```python
# HEX kodundan
turuncu = sv.Color.from_hex("#FF6B35")
mavi = sv.Color.from_hex("#3498DB")

# RGB'den
kirmizi = sv.Color.from_rgb_tuple((255, 0, 0))

# BGR'den (OpenCV formatı)
yesil = sv.Color.from_bgr_tuple((0, 255, 0))
```

---

## 6️⃣ Video İşleme Araçları

### VideoInfo - Video Bilgileri

```python
video_info = sv.VideoInfo.from_video_path("video.mp4")

print(video_info.width)         # 1920
print(video_info.height)        # 1080
print(video_info.fps)           # 30.0
print(video_info.total_frames)  # 9000
```

### Frame Generator - Frame Okuma

```python
frame_generator = sv.get_video_frames_generator(
    source_path="video.mp4",
    stride=1,      # Her frame (2 = her 2'de 1)
    start=0,       # Başlangıç frame'i
    end=None       # Bitiş frame'i (None = son)
)

for frame in frame_generator:
    # frame işle
    pass
```

### VideoSink - Video Yazma

```python
video_info = sv.VideoInfo.from_video_path("input.mp4")

with sv.VideoSink(target_path="output.mp4", video_info=video_info) as sink:
    for frame in frame_generator:
        # Frame'i işle
        annotated_frame = process(frame)
        
        # Dosyaya yaz
        sink.write_frame(annotated_frame)
```

---

## 7️⃣ Zone (Bölge) Sistemi

### LineZone - Çizgi Geçiş Sayacı

```python
# Çizgi tanımla
line_zone = sv.LineZone(
    start=sv.Point(0, 500),       # Başlangıç noktası
    end=sv.Point(1920, 500)       # Bitiş noktası
)

# Geçişleri tetikle
line_zone.trigger(detections)

# Sonuçlar
print(line_zone.in_count)   # İçeri geçiş sayısı
print(line_zone.out_count)  # Dışarı geçiş sayısı
```

### LineZoneAnnotator

```python
line_annotator = sv.LineZoneAnnotator(
    thickness=2,
    text_thickness=2,
    text_scale=1.0,
    text_offset=10,
    color=sv.Color.YELLOW
)

annotated_frame = line_annotator.annotate(
    frame=frame,
    line_counter=line_zone
)
```

### PolygonZone - Çokgen Alan Sayacı

```python
# Çokgen köşeleri tanımla
polygon = np.array([
    [100, 100],
    [500, 100],
    [500, 400],
    [100, 400]
])

# Zone oluştur
polygon_zone = sv.PolygonZone(
    polygon=polygon,
    triggering_anchors=[sv.Position.BOTTOM_CENTER]
)

# İçeride mi kontrol et
is_inside = polygon_zone.trigger(detections)
inside_count = np.sum(is_inside)
```

### PolygonZoneAnnotator

```python
zone_annotator = sv.PolygonZoneAnnotator(
    zone=polygon_zone,
    color=sv.Color.RED,
    thickness=2,
    text_thickness=2,
    text_scale=1.0
)

annotated_frame = zone_annotator.annotate(scene=frame)
```

---

## 8️⃣ Tam Kullanım Örneği

```python
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

# 1. Model ve Tracker
model = YOLO("model.pt")
tracker = sv.ByteTrack()

# 2. Annotator'lar
box_annotator = sv.RoundBoxAnnotator(thickness=2, roundness=0.6)
label_annotator = sv.LabelAnnotator(text_scale=0.5)
trace_annotator = sv.TraceAnnotator(trace_length=30)

# 3. Video bilgileri
video_info = sv.VideoInfo.from_video_path("input.mp4")
frame_generator = sv.get_video_frames_generator("input.mp4")

# 4. İşleme döngüsü
with sv.VideoSink("output.mp4", video_info) as sink:
    for frame in frame_generator:
        # Tespit
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # Tracking
        detections = tracker.update_with_detections(detections)
        
        # Etiketler
        labels = [f"#{tid}" for tid in detections.tracker_id]
        
        # Görselleştirme
        frame = trace_annotator.annotate(frame, detections)
        frame = box_annotator.annotate(frame, detections)
        frame = label_annotator.annotate(frame, detections, labels)
        
        # Kaydet
        sink.write_frame(frame)
```

---

## 📌 İpuçları

1. **Annotator Sırası Önemli**: Altta kalmasını istediğin annotator'ı önce çağır
   ```python
   frame = trace_annotator.annotate(frame, detections)  # En altta
   frame = box_annotator.annotate(frame, detections)    # Ortada
   frame = label_annotator.annotate(frame, detections)  # En üstte
   ```

2. **Performans**: Büyük videolarda `stride` parametresi kullan
   ```python
   frame_generator = sv.get_video_frames_generator("video.mp4", stride=2)
   ```

3. **Filtreleme**: Detections üzerinde NumPy maskeleri kullan
   ```python
   # Sadece yüksek güvenli tespitler
   high_conf = detections[detections.confidence > 0.7]
   
   # Sadece belirli sınıf
   persons = detections[detections.class_id == 0]
   ```

4. **Renk Paleti**: ColorPalette kullan
   ```python
   palette = sv.ColorPalette.from_hex(["#FF0000", "#00FF00", "#0000FF"])
   ```

---

## 🔗 Kaynaklar

- [Supervision GitHub](https://github.com/roboflow/supervision)
- [Supervision Dokümantasyon](https://supervision.roboflow.com/)
- [Roboflow Blog](https://blog.roboflow.com/)

---

*Bu rehber, futbol_player_analizi projesi için hazırlanmıştır.*

