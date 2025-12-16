"""
===============================================================================
⚽ FUTBOL OYUNCU ANALİZ SİSTEMİ ⚽
===============================================================================
"""

# =============================================================================
# KÜTÜPHANE İMPORTLARI
# =============================================================================

import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# SUPERVISION KÜTÜPHANESİ
# -----------------------------------------------------------------------------

import supervision as sv

# =============================================================================
# MODEL YOLU
# =============================================================================

MODEL_PATH = "C:/Users/Batu/Desktop/futbol_player_analizi/futbol_tespit_projesi/yolov8s_v1/weights/best.pt"

# =============================================================================
# FUTBOL OYUNCU ANALİZ SİSTEMİ
# =============================================================================

class FutbolAnalizSistemi:
    """Futbol oyuncu ve hakem tespiti, takip ve sayma sistemi."""
    
    def __init__(self, model_path: str, confidence: float = 0.35, iou: float = 0.5):
        """Sistemi başlatır."""
        
        # ---------------------------------------------------------------------
        # MODEL
        # ---------------------------------------------------------------------
        print("🔄 Model yükleniyor...")
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou = iou
        self.class_names = self.model.names
        print(f"✅ Model yüklendi! Sınıflar: {self.class_names}")
        
        # ---------------------------------------------------------------------
        # TRACKER
        # ---------------------------------------------------------------------
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30
        )
        
        # ---------------------------------------------------------------------
        # ANNOTATORS
        # ---------------------------------------------------------------------
        self.box_annotator = sv.RoundBoxAnnotator(
            thickness=2,
            roundness=0.6,
            color_lookup=sv.ColorLookup.CLASS
        )
        
        self.corner_annotator = sv.BoxCornerAnnotator(
            thickness=3,
            corner_length=15,
            color_lookup=sv.ColorLookup.CLASS
        )
        
        self.label_annotator = sv.LabelAnnotator(
            text_scale=0.5,
            text_padding=5,
            text_position=sv.Position.TOP_CENTER,
            color_lookup=sv.ColorLookup.CLASS,
            text_color=sv.Color.BLACK
        )
        
        self.trace_annotator = sv.TraceAnnotator(
            thickness=2,
            trace_length=50,
            position=sv.Position.BOTTOM_CENTER,
            color_lookup=sv.ColorLookup.CLASS
        )
        
        # ---------------------------------------------------------------------
        # RENK PALETİ
        # ---------------------------------------------------------------------
        self.color_palette = sv.ColorPalette.from_hex([
            "#FFFFFF",  # player - beyaz
            "#00FFFF",  # goalkeeper - cyan
            "#FFFF00",  # referee - sarı
            "#00FF00",  # ball - yeşil
        ])
        
        # ---------------------------------------------------------------------
        # ISI HARİTASI
        # ---------------------------------------------------------------------
        self.heatmap_data = None
        self.frame_shape = None
        
        # ---------------------------------------------------------------------
        # SAYAÇ
        # ---------------------------------------------------------------------
        self.class_counts = {}
        
        print("=" * 50)
        print("⚽ SİSTEM HAZIR!")
        print("=" * 50)
    
    # =========================================================================
    # TESPİT
    # =========================================================================
    
    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Frame üzerinde nesne tespiti yapar."""
        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou,
            verbose=False,
            agnostic_nms=True
        )[0]
        return sv.Detections.from_ultralytics(results)
    
    # =========================================================================
    # TAKİP
    # =========================================================================
    
    def track(self, detections: sv.Detections) -> sv.Detections:
        """Tespitlere takip ID'si atar."""
        return self.tracker.update_with_detections(detections)
    
    # =========================================================================
    # ISI HARİTASI
    # =========================================================================
    
    def update_heatmap(self, frame: np.ndarray, detections: sv.Detections):
        """Isı haritası verilerini günceller."""
        if self.heatmap_data is None or self.frame_shape != frame.shape[:2]:
            self.frame_shape = frame.shape[:2]
            self.heatmap_data = np.zeros(self.frame_shape, dtype=np.float32)
        
        for bbox in detections.xyxy:
            cx, cy = int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)
            cv2.circle(self.heatmap_data, (cx, cy), 25, 1, -1)
    
    def draw_heatmap(self, frame: np.ndarray) -> np.ndarray:
        """Isı haritasını frame üzerine çizer."""
        if self.heatmap_data is None:
            return frame
        
        heatmap = self.heatmap_data.copy()
        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
        else:
            heatmap = heatmap.astype(np.uint8)
        
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 0.85, heatmap_colored, 0.15, 0)
    
    # =========================================================================
    # ETİKET OLUŞTUR
    # =========================================================================
    
    def create_labels(self, detections: sv.Detections) -> list:
        """Sınıf adı ve başarı oranı içeren etiketler oluşturur."""
        labels = []
        for i in range(len(detections)):
            class_id = detections.class_id[i]
            confidence = detections.confidence[i]
            class_name = self.class_names.get(class_id, "unknown")
            labels.append(f"{class_name} {confidence:.0%}")
        return labels
    
    # =========================================================================
    # SAYIM
    # =========================================================================
    
    def count_classes(self, detections: sv.Detections) -> dict:
        """Sınıf bazlı sayım yapar."""
        counts = {}
        for class_id in detections.class_id:
            class_name = self.class_names.get(class_id, "unknown")
            counts[class_name] = counts.get(class_name, 0) + 1
        return counts
    
    # =========================================================================
    # BİLGİ PANELİ
    # =========================================================================
    
    def draw_info_panel(self, frame: np.ndarray, counts: dict, total: int) -> np.ndarray:
        """Sol üst köşeye bilgi paneli çizer."""
        lines = [f"Toplam: {total}"] + [f"{k}: {v}" for k, v in counts.items()]
        
        h = 20 + len(lines) * 25
        w = 180
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (w, h), (40, 160, 240), -1)
        cv2.rectangle(overlay, (8, 8), (w, h), (20, 120, 200), 2)
        frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
        
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (15, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        
        return frame
    
    # =========================================================================
    # FRAME İŞLE
    # =========================================================================
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Tek frame'i işler ve görselleştirir."""
        
        # Tespit
        detections = self.detect(frame)
        
        if len(detections) == 0:
            return frame
        
        # Takip
        detections = self.track(detections)
        
        # Isı haritası güncelle
        self.update_heatmap(frame, detections)
        
        # Sayım
        counts = self.count_classes(detections)
        
        # Etiketler
        labels = self.create_labels(detections)
        
        # Görselleştirme
        annotated = frame.copy()
        annotated = self.draw_heatmap(annotated)
        annotated = self.trace_annotator.annotate(annotated, detections)
        annotated = self.box_annotator.annotate(annotated, detections)
        annotated = self.corner_annotator.annotate(annotated, detections)
        annotated = self.label_annotator.annotate(annotated, detections, labels)
        annotated = self.draw_info_panel(annotated, counts, len(detections))
        
        return annotated
    
    # =========================================================================
    # VİDEO İŞLE
    # =========================================================================
    
    def process_video(self, video_path: str, output_path: str = None, show: bool = True):
        """Video dosyasını işler."""
        
        video_info = sv.VideoInfo.from_video_path(video_path)
        print(f"\n📹 Video: {video_info.width}x{video_info.height} @ {video_info.fps}fps")
        print(f"📊 Toplam: {video_info.total_frames} frame\n")
        
        generator = sv.get_video_frames_generator(video_path)
        
        from tqdm import tqdm
        
        if output_path:
            with sv.VideoSink(output_path, video_info) as sink:
                for frame in tqdm(generator, total=video_info.total_frames):
                    result = self.process_frame(frame)
                    sink.write_frame(result)
                    
                    if show:
                        cv2.imshow("Futbol Analiz", result)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
        else:
            for frame in tqdm(generator, total=video_info.total_frames):
                result = self.process_frame(frame)
                
                if show:
                    cv2.imshow("Futbol Analiz", result)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        cv2.destroyAllWindows()
        print("\n✅ Tamamlandı!")

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Ana fonksiyon."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Futbol Analiz Sistemi")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--model", type=str, default=MODEL_PATH)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.5)
    
    args = parser.parse_args()
    
    print("\n" + "=" * 50)
    print("⚽ FUTBOL ANALİZ SİSTEMİ")
    print("=" * 50 + "\n")
    
    sistem = FutbolAnalizSistemi(args.model, args.conf, args.iou)
    sistem.process_video(args.video, args.output)

# =============================================================================
# ÇALIŞTIR
# =============================================================================

if __name__ == "__main__":
    main()
