import time
import torch
from pathlib import Path
from collections import defaultdict
from app.services.horse_cascade_service import HorseCascadeService

def main():
    service = HorseCascadeService(
        detector_path="/home/maks/Moderator-CNN/models/horse/yolov8_horse_head.pt",
        top_k=3,
        match_threshold=14
    )

    device_name = service.pipeline.biometrics_service.device
    print(f"⚙️ [Device Check] PyTorch CUDA available: {torch.cuda.is_available()} | Active Device: {device_name}")

    val_dir = Path("/home/maks/Moderator-CNN/data/Horse")
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    image_paths = sorted([p for p in val_dir.rglob("*") if p.suffix.lower() in valid_exts])

    horses_dict = defaultdict(list)
    for p in image_paths:
        horses_dict[p.parent.name].append(str(p))

    print("🚀 [Tight Crop Enrollment] Индексация...")
    gallery_count = 0
    query_paths = []

    for horse_id, paths in horses_dict.items():
        enroll_paths = paths[:2]
        test_paths = paths[2:] if len(paths) > 2 else paths[1:]

        for ep in enroll_paths:
            if service.enroll_horse(horse_id, ep):
                gallery_count += 1

        for tp in test_paths:
            query_paths.append((horse_id, tp))

    print(f"✅ Индексировано ракурсов: {gallery_count}")
    print(f"🧪 Поиск 1:N (Threshold=14) для {len(query_paths)} запросов...\n")

    correct_matches = 0
    false_matches = 0
    missed_matches = 0
    errors_count = 0
    total_time = 0.0

    for true_id, q_path in query_paths:
        t0 = time.time()
        result = service.search_1_to_n(q_path)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.time() - t0) * 1000
        total_time += elapsed

        status = result.get("status", "ERROR")
        matched_id = result.get("matched_horse_id", None)
        inliers = result.get("max_inliers", 0)
        bd = result.get("breakdown_ms", {})

        if status == "MATCH":
            if matched_id == true_id:
                correct_matches += 1
                tag = "✅ CORRECT"
            else:
                false_matches += 1
                tag = "❌ FALSE MATCH"
        elif status == "NO_MATCH":
            missed_matches += 1
            tag = "⚠️ MISSED"
        else:
            errors_count += 1
            tag = f"⛔ ERROR: {result.get('message', 'Unknown error')}"

        print(f"  Query: {true_id:4s} | Match: {str(matched_id):4s} | Inliers: {inliers:2d} | Time: {elapsed:5.0f}ms (YOLO:{bd.get('yolo',0)}ms DINO:{bd.get('dino',0)}ms LoFTR:{bd.get('loftr',0)}ms) | {tag}")

    avg_latency = total_time / max(1, len(query_paths))
    print("\n=== ИТОГИ OPTIMIZED TIGHT CROP SEARCH (Th=14) ===")
    print(f"  • Всего тестовых запросов : {len(query_paths)}")
    print(f"  • Верных распознаваний    : {correct_matches}")
    print(f"  • Ложных срабатываний (FAR): {false_matches}")
    print(f"  • Пропущенных (FRR)        : {missed_matches}")
    print(f"  • Ошибок детекции          : {errors_count}")
    print(f"  • Средняя задержка поиска : {avg_latency:.2f} ms")

if __name__ == "__main__":
    main()