import cv2
import os
from pathlib import Path

def process_videos():
    src_dir = Path("data/train/violence")
    print("🎬 Нарезка видео на кадры для обучения...")
    
    # Ищем все видео
    videos = list(src_dir.glob("*.mp4"))
    if not videos:
        print("❌ Видео не найдены. Сначала верни .mp4 в папку violence!")
        return

    for video_path in videos:
        cap = cv2.VideoCapture(str(video_path))
        v_name = video_path.stem
        count = 0
        success, image = cap.read()
        
        # Берем каждый 15-й кадр, чтобы данные не были слишком одинаковыми
        frame_idx = 0
        while success:
            if frame_idx % 15 == 0:
                # Сохраняем как JPG
                frame_name = f"frame_{v_name}_{count}.jpg"
                cv2.imwrite(str(src_dir / frame_name), image)
                count += 1
            
            success, image = cap.read()
            frame_idx += 1
            
            # Ограничение: не более 30 кадров с одного видео для баланса
            if count >= 30: break
            
        cap.release()
        print(f"✅ Извлечено {count} кадров из {video_path.name}")
        os.remove(video_path) # Удаляем видео, оставляем только картинки

if __name__ == "__main__":
    process_videos()